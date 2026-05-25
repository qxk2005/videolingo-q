import os
import json
import time
import shutil
import subprocess
from typing import Tuple

import pandas as pd
from pydub import AudioSegment
from rich.console import Console
from rich.progress import Progress
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.utils import *
from core.utils.models import *
from core.asr_backend.audio_preprocess import get_audio_duration
from core.tts_backend.tts_main import tts_main

console = Console()

TEMP_FILE_TEMPLATE = f"{_AUDIO_TMP_DIR}/{{}}_temp.wav"
OUTPUT_FILE_TEMPLATE = f"{_AUDIO_SEGS_DIR}/{{}}.wav"
WARMUP_SIZE = 5
TTS_SETTINGS_FILE = "output/audio/.tts_settings"

def get_tts_fingerprint():
    """Get a string fingerprint of current TTS settings to detect changes."""
    tts_method = load_key("tts_method")
    settings = {"tts_method": tts_method}
    try:
        if tts_method == "edge_tts":
            settings["voice"] = load_key("edge_tts.voice")
        elif tts_method == "openai_tts":
            settings["voice"] = load_key("openai_tts.voice")
        elif tts_method == "azure_tts":
            settings["voice"] = load_key("azure_tts.voice")
        elif tts_method == "fish_tts":
            settings["character"] = load_key("fish_tts.character")
        elif tts_method == "sf_fish_tts":
            settings["voice"] = load_key("sf_fish_tts.voice")
        elif tts_method == "gpt_sovits":
            settings["character"] = load_key("gpt_sovits.character")
    except Exception:
        pass
    return json.dumps(settings, sort_keys=True)

def check_and_clear_audio_cache():
    """If TTS settings changed since last run, clear cached audio files."""
    current = get_tts_fingerprint()
    if os.path.exists(TTS_SETTINGS_FILE):
        with open(TTS_SETTINGS_FILE, 'r', encoding='utf-8') as f:
            stored = f.read().strip()
        if stored != current:
            rprint(f"[yellow]⚠️ TTS settings changed ({stored} → {current}), clearing audio cache...[/yellow]")
            for d in [_AUDIO_TMP_DIR, _AUDIO_SEGS_DIR]:
                if os.path.exists(d):
                    shutil.rmtree(d)
    settings_dir = os.path.dirname(TTS_SETTINGS_FILE)
    if settings_dir:
        os.makedirs(settings_dir, exist_ok=True)
    with open(TTS_SETTINGS_FILE, 'w', encoding='utf-8') as f:
        f.write(current)

def parse_df_srt_time(time_str: str) -> float:
    """Convert SRT time format to seconds"""
    hours, minutes, seconds = time_str.strip().split(':')
    seconds, milliseconds = seconds.split('.')
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000

def adjust_audio_speed(input_file: str, output_file: str, speed_factor: float) -> None:
    """Adjust audio speed and handle edge cases"""
    # If the speed factor is close to 1, directly copy the file
    if abs(speed_factor - 1.0) < 0.001:
        shutil.copy2(input_file, output_file)
        return
    
    # Check for invalid speed factors
    if speed_factor <= 0:
        rprint(f"[red]❌ Error: Invalid speed factor {speed_factor} (must be positive)[/red]")
        rprint(f"[yellow]⚠️ Copying file unchanged due to invalid speed factor[/yellow]")
        shutil.copy2(input_file, output_file)
        return
    
    # Clamp speed factor to FFmpeg atempo limits (0.5 to 100.0)
    # For extreme cases, we'll clamp to reasonable bounds
    min_speed, max_speed = 0.5, 4.0
    if speed_factor < min_speed or speed_factor > max_speed:
        rprint(f"[yellow]⚠️ Speed factor {speed_factor:.3f} is outside safe range [{min_speed}-{max_speed}], clamping[/yellow]")
        speed_factor = max(min_speed, min(speed_factor, max_speed))
        
    atempo = speed_factor
    cmd = ['ffmpeg', '-i', input_file, '-filter:a', f'atempo={atempo}', '-y', output_file]
    input_duration = get_audio_duration(input_file)
    max_retries = 2
    for attempt in range(max_retries):
        try:
            subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
            output_duration = get_audio_duration(output_file)
            expected_duration = input_duration / speed_factor
            diff = output_duration - expected_duration
            # If the output duration exceeds the expected duration, but the input audio is less than 3 seconds, and the error is within 0.1 seconds, truncate to the expected length
            if output_duration >= expected_duration * 1.02 and input_duration < 3 and diff <= 0.1:
                audio = AudioSegment.from_wav(output_file)
                trimmed_audio = audio[:(expected_duration * 1000)]  # pydub uses milliseconds
                trimmed_audio.export(output_file, format="wav")
                print(f"✂️ Trimmed to expected duration: {expected_duration:.2f} seconds")
                return
            elif output_duration >= expected_duration * 1.02:
                raise Exception(f"Audio duration abnormal: input file={input_file}, output file={output_file}, speed factor={speed_factor}, input duration={input_duration:.2f}s, output duration={output_duration:.2f}s")
            return
        except subprocess.CalledProcessError as e:
            if attempt < max_retries - 1:
                rprint(f"[yellow]⚠️ Audio speed adjustment failed, retrying in 1s ({attempt + 1}/{max_retries})[/yellow]")
                time.sleep(1)
            else:
                rprint(f"[red]❌ Audio speed adjustment failed, max retries reached ({max_retries})[/red]")
                raise e

def trim_tts_silence(input_file: str) -> None:
    """Trim leading and trailing silence from TTS audio in-place.

    TTS engines (especially edge_tts) often add 100-300ms of silence at start/end.
    Removing it reduces real_dur, which in turn lowers the required speed-up factor.
    """
    temp_file = input_file.replace('.wav', '_trimmed.wav')
    cmd = [
        'ffmpeg', '-i', input_file,
        '-af', (
            'silenceremove='
            'start_periods=1:start_silence=0.05:start_threshold=-45dB:'
            'stop_periods=-1:stop_silence=0.1:stop_threshold=-45dB'
        ),
        '-y', temp_file
    ]
    try:
        subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
        new_dur = get_audio_duration(temp_file)
        orig_dur = get_audio_duration(input_file)
        # Only replace if trimming actually reduced duration (sanity check)
        if 0 < new_dur < orig_dur:
            shutil.move(temp_file, input_file)
        elif os.path.exists(temp_file):
            os.remove(temp_file)
    except Exception:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def process_row(row: pd.Series, tasks_df: pd.DataFrame) -> Tuple[int, float]:
    """Helper function for processing single row data with continuous dubbing optimization"""
    number = row['number']
    lines = eval(row['lines']) if isinstance(row['lines'], str) else row['lines']
    real_dur = 0
    
    tts_method = load_key("tts_method")
    success_continuous = False
    
    # Apply continuous dubbing optimization for multi-line chunks in doubao_tts
    if len(lines) > 1 and tts_method == "doubao_tts":
        try:
            import tempfile
            from pydub.silence import detect_nonsilent
            
            # Combine sentences with terminal punctuation and double spaces for natural speech pauses
            joined_lines = []
            display_lang = load_key("display_language")
            for l in lines:
                l_strip = l.strip()
                if not l_strip:
                    continue
                # Ensure each line ends with a sentence terminator for natural prosody
                if l_strip[-1] not in ['.', '!', '?', '。', '！', '？', ',', '，']:
                    l_strip += '。' if display_lang == "zh-CN" else '.'
                joined_lines.append(l_strip)
                
            combined_text = "  ".join(joined_lines)
            
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
                combined_temp_path = tmp_file.name
            
            # Generate continuous paragraph audio
            tts_main(combined_text, combined_temp_path, number, tasks_df)
            
            # Load the continuous audio
            combined_audio = AudioSegment.from_file(combined_temp_path)
            
            # Adaptive threshold scan to identify exactly len(lines) nonsilent segments
            best_ranges = None
            for thresh in range(-45, -15, 2):
                ranges = detect_nonsilent(combined_audio, min_silence_len=200, silence_thresh=thresh)
                # Filter out extremely short nonsilent chunks (e.g. less than 100ms) which might be noise
                ranges = [r for r in ranges if (r[1] - r[0]) > 100]
                if len(ranges) == len(lines):
                    best_ranges = ranges
                    break
            
            if best_ranges is not None:
                # Successfully identified and split all sentence segments!
                rprint(f"[bold green]✨ Continuous Dubbing Success: Chunk {number} with {len(lines)} lines successfully aligned and split on silence![/bold green]")
                for line_index, (start_ms, end_ms) in enumerate(best_ranges):
                    temp_file = TEMP_FILE_TEMPLATE.format(f"{number}_{line_index}")
                    # Extract segment with 100ms padding for smooth start/end transition
                    pad = 100
                    start_pad = max(0, start_ms - pad)
                    end_pad = min(len(combined_audio), end_ms + pad)
                    segment = combined_audio[start_pad:end_pad]
                    segment.export(temp_file, format="wav")
                    
                    trim_tts_silence(temp_file)
                    dur = get_audio_duration(temp_file)
                    real_dur += dur
                
                success_continuous = True
            else:
                rprint(f"[yellow]⚠️ Continuous split fallback: Expected {len(lines)} segments but found different counts. Falling back to individual generation.[/yellow]")
            
            if os.path.exists(combined_temp_path):
                os.remove(combined_temp_path)
                
        except Exception as e:
            rprint(f"[yellow]⚠️ Continuous Dubbing failed: {e}. Falling back to standard sentence-by-sentence generation.[/yellow]")
            
    # Fallback to original sentence-by-sentence generation
    if not success_continuous:
        real_dur = 0
        for line_index, line in enumerate(lines):
            temp_file = TEMP_FILE_TEMPLATE.format(f"{number}_{line_index}")
            tts_main(line, temp_file, number, tasks_df)
            trim_tts_silence(temp_file)  # Remove TTS leading/trailing silence before measuring
            dur = get_audio_duration(temp_file)
            if dur <= 0 and os.path.exists(temp_file):
                os.remove(temp_file)
                tts_main(line, temp_file, number, tasks_df)
                trim_tts_silence(temp_file)
                dur = get_audio_duration(temp_file)
            real_dur += dur
            
    return number, real_dur

def generate_tts_audio(tasks_df: pd.DataFrame) -> pd.DataFrame:
    """Generate TTS audio sequentially and calculate actual duration"""
    tasks_df['real_dur'] = 0
    rprint("[bold green]🎯 Starting TTS audio generation...[/bold green]")
    
    from core.utils.progress_utils import get_progress, update_st_progress
    progress = get_progress()
    is_internal = not progress.live.is_started
    if is_internal: progress.start()

    task_desc = "🔄 正在生成 TTS 配音音频..."
    task = progress.add_task(task_desc, total=len(tasks_df))
    
    # Use user-defined tts_max_workers. For gpt_sovits, force 1 to avoid conflicts.
    tts_method = load_key("tts_method")
    tts_max_workers = load_key("tts_max_workers")
    
    if tts_method == "gpt_sovits":
        tts_max_workers = 1
    
    # Specialized handling for Gemini TTS to prevent rate limits
    if tts_method == "gemini_tts":
        rprint(f"[yellow]⏳ Gemini TTS mode: Using batch-sequential processing (Batch size: {tts_max_workers}) to prevent rate limits...[/yellow]")
        # Process in batches of tts_max_workers
        for i in range(0, len(tasks_df), tts_max_workers):
            batch = tasks_df.iloc[i:i + tts_max_workers]
            with ThreadPoolExecutor(max_workers=tts_max_workers) as executor:
                futures = [
                    executor.submit(process_row, row, tasks_df.copy())
                    for _, row in batch.iterrows()
                ]
                for future in as_completed(futures):
                    try:
                        number, real_dur = future.result()
                        tasks_df.loc[tasks_df['number'] == number, 'real_dur'] = real_dur
                        progress.advance(task)
                    except Exception as e:
                        rprint(f"[red]❌ Error in Gemini batch TTS: {str(e)}[/red]")
                        raise e
            # Small wait between batches if not the last batch
            if i + tts_max_workers < len(tasks_df):
                wait_time = 2  # Adjust wait time as needed
                rprint(f"[dim]Waiting {wait_time}s for next Gemini batch...[/dim]")
                time.sleep(wait_time)
            update_st_progress(min(i + tts_max_workers, len(tasks_df)), len(tasks_df), task_desc)
    
    elif tts_max_workers == 1:
        # Sequential processing
        for i, (_, row) in enumerate(tasks_df.iterrows()):
            try:
                number, real_dur = process_row(row, tasks_df)
                tasks_df.loc[tasks_df['number'] == number, 'real_dur'] = real_dur
                progress.advance(task)
                update_st_progress(i + 1, len(tasks_df), task_desc)
            except Exception as e:
                rprint(f"[red]❌ Error in sequential TTS: {str(e)}[/red]")
                raise e
    else:
        # Parallel processing using the specified batch size (max_workers)
        with ThreadPoolExecutor(max_workers=tts_max_workers) as executor:
            futures = [
                executor.submit(process_row, row, tasks_df.copy())
                for _, row in tasks_df.iterrows()
            ]
            
            completed_count = 0
            for future in as_completed(futures):
                try:
                    number, real_dur = future.result()
                    tasks_df.loc[tasks_df['number'] == number, 'real_dur'] = real_dur
                    progress.advance(task)
                    completed_count += 1
                    update_st_progress(completed_count, len(tasks_df), task_desc)
                except Exception as e:
                    rprint(f"[red]❌ Error in parallel TTS: {str(e)}[/red]")
                    raise e

    if is_internal:
        progress.stop()
    else:
        progress.remove_task(task)

    rprint("[bold green]✨ TTS audio generation completed![/bold green]")
    return tasks_df

def process_chunk(chunk_df: pd.DataFrame, accept: float, min_speed: float) -> tuple[float, bool]:
    """Process audio chunk and calculate speed factor"""
    chunk_durs = chunk_df['real_dur'].sum()
    tol_durs = chunk_df['tol_dur'].sum()
    durations = tol_durs - chunk_df.iloc[-1]['tolerance']
    all_gaps = chunk_df['gap'].sum() - chunk_df.iloc[-1]['gap']
    
    keep_gaps = True
    speed_var_error = 0.1
    
    # Handle negative tol_durs or durations - this indicates timing constraints are impossible
    if tol_durs <= 0 or durations <= 0:
        rprint(f"[yellow]⚠️ Warning: Negative time constraints (tol_durs: {tol_durs:.3f}s, durations: {durations:.3f}s), using minimum speed[/yellow]")
        return min_speed, False
    
    # Ensure we don't divide by negative or very small numbers
    safe_durations = max(durations - speed_var_error, 0.01)
    safe_tol_durs = max(tol_durs - speed_var_error, 0.01)

    if (chunk_durs + all_gaps) / accept < durations:
        speed_factor = max(min_speed, (chunk_durs + all_gaps) / safe_durations)
    elif chunk_durs / accept < durations:
        speed_factor = max(min_speed, chunk_durs / safe_durations)
        keep_gaps = False
    elif (chunk_durs + all_gaps) / accept < tol_durs:
        speed_factor = max(min_speed, (chunk_durs + all_gaps) / safe_tol_durs)
    else:
        speed_factor = chunk_durs / safe_tol_durs
        keep_gaps = False
    
    # Additional safety check to ensure speed factor is reasonable
    speed_factor = max(min_speed, min(speed_factor, 4.0))
        
    return round(speed_factor, 3), keep_gaps

def _apply_speed_to_chunk(chunk_df: pd.DataFrame, speed_factor: float, keep_gaps: bool, chunk_start_time: float):
    """Apply speed adjustment to every audio file in a chunk.
    Returns (final_cur_time, {number: new_sub_times})."""
    cur_time = chunk_start_time
    number_to_sub_times = {}
    for i, row in chunk_df.iterrows():
        if i != 0 and keep_gaps:
            cur_time += chunk_df.iloc[i - 1]['gap'] / speed_factor
        number = row['number']
        lines = eval(row['lines']) if isinstance(row['lines'], str) else row['lines']
        new_sub_times = []
        for line_index, _ in enumerate(lines):
            temp_file = TEMP_FILE_TEMPLATE.format(f"{number}_{line_index}")
            output_file = OUTPUT_FILE_TEMPLATE.format(f"{number}_{line_index}")
            adjust_audio_speed(temp_file, output_file, speed_factor)
            ad_dur = get_audio_duration(output_file)
            new_sub_times.append([cur_time, cur_time + ad_dur])
            cur_time += ad_dur
        number_to_sub_times[number] = new_sub_times
    return cur_time, number_to_sub_times


def _truncate_chunk_tail(tasks_df: pd.DataFrame, index: int, chunk_end_time: float) -> None:
    """Truncate the last audio segment of a chunk so it ends at chunk_end_time."""
    last_number = tasks_df.iloc[index]['number']
    last_lines = eval(tasks_df.iloc[index]['lines']) if isinstance(tasks_df.iloc[index]['lines'], str) else tasks_df.iloc[index]['lines']
    last_file = OUTPUT_FILE_TEMPLATE.format(f"{last_number}_{len(last_lines) - 1}")
    audio = AudioSegment.from_wav(last_file)
    cur_end = tasks_df.at[index, 'new_sub_times'][-1][1]
    keep_ms = max(0, (len(audio) / 1000 - (cur_end - chunk_end_time)) * 1000)
    audio[:keep_ms].export(last_file, format="wav")
    last_times = tasks_df.at[index, 'new_sub_times']
    last_times[-1][1] = chunk_end_time
    tasks_df.at[index, 'new_sub_times'] = last_times


def merge_chunks(tasks_df: pd.DataFrame) -> pd.DataFrame:
    """Merge audio chunks and adjust timeline"""
    rprint("[bold blue]🔄 Starting audio chunks processing...[/bold blue]")
    accept = load_key("speed_factor.accept")
    min_speed = load_key("speed_factor.min")
    chunk_start = 0

    tasks_df['new_sub_times'] = None

    for index, row in tasks_df.iterrows():
        if row['cut_off'] != 1:
            continue

        chunk_df = tasks_df.iloc[chunk_start:index + 1].reset_index(drop=True)
        speed_factor, keep_gaps = process_chunk(chunk_df, accept, min_speed)

        chunk_start_time = parse_df_srt_time(chunk_df.iloc[0]['start_time'])
        chunk_end_time = parse_df_srt_time(chunk_df.iloc[-1]['end_time']) + chunk_df.iloc[-1]['tolerance']

        # First pass: apply speed adjustment
        cur_time, sub_times = _apply_speed_to_chunk(chunk_df, speed_factor, keep_gaps, chunk_start_time)

        # If actual audio overflows by more than 0.6 s, recalculate speed from real durations
        if cur_time > chunk_end_time + 0.6:
            available = chunk_end_time - chunk_start_time
            actual = cur_time - chunk_start_time
            
            # Prevent invalid calculations
            if available <= 0 or actual <= 0:
                rprint(f"[red]❌ Error: Invalid time calculations - available: {available:.3f}s, actual: {actual:.3f}s[/red]")
                rprint(f"[yellow]⚠️ Skipping speed correction for chunk {chunk_start}–{index}[/yellow]")
            else:
                corrected_sf = speed_factor * actual / available
                # Clamp corrected speed factor to reasonable bounds
                corrected_sf = max(0.5, min(corrected_sf, 4.0))
                rprint(f"[yellow]⚠️ Chunk {chunk_start}–{index} overflowed by {cur_time - chunk_end_time:.2f}s "
                       f"(sf {speed_factor:.3f}→{corrected_sf:.3f}), retrying...[/yellow]")
                cur_time, sub_times = _apply_speed_to_chunk(chunk_df, corrected_sf, keep_gaps, chunk_start_time)
                speed_factor = corrected_sf

        # Write sub_times back to tasks_df
        emoji = "⚡" if speed_factor <= accept else "⚠️"
        for number, new_sub_times in sub_times.items():
            main_df_idx = tasks_df[tasks_df['number'] == number].index[0]
            tasks_df.at[main_df_idx, 'new_sub_times'] = new_sub_times
        rprint(f"[cyan]{emoji} Processed chunk {chunk_start} to {index} with speed factor {speed_factor:.3f}[/cyan]")

        # Final overflow check: truncate tail if still slightly over
        if cur_time > chunk_end_time:
            time_diff = cur_time - chunk_end_time
            rprint(f"[yellow]⚠️ Chunk {chunk_start}–{index} exceeds by {time_diff:.3f}s, truncating last audio[/yellow]")
            _truncate_chunk_tail(tasks_df, index, chunk_end_time)

        chunk_start = index + 1

    rprint("[bold green]✅ Audio chunks processing completed![/bold green]")
    return tasks_df

def gen_audio() -> None:
    """Main function: Generate audio and process timeline"""
    rprint("[bold magenta]🚀 Starting audio generation process...[/bold magenta]")
    
    # 🎯 Step0: Check if TTS settings changed; clear cache if so
    check_and_clear_audio_cache()

    # 🎯 Step1: Create necessary directories
    os.makedirs(_AUDIO_TMP_DIR, exist_ok=True)
    os.makedirs(_AUDIO_SEGS_DIR, exist_ok=True)
    
    # 📝 Step2: Load task file
    tasks_df = pd.read_excel(_8_1_AUDIO_TASK)
    rprint("[green]📊 Loaded task file successfully[/green]")
    
    # 🔊 Step3: Generate TTS audio
    tasks_df = generate_tts_audio(tasks_df)
    
    # 🔄 Step4: Merge audio chunks
    tasks_df = merge_chunks(tasks_df)
    
    # 💾 Step5: Save results
    tasks_df.to_excel(_8_1_AUDIO_TASK, index=False)
    rprint("[bold green]🎉 Audio generation completed successfully![/bold green]")

if __name__ == "__main__":
    gen_audio()
