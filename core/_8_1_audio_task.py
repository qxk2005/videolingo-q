import datetime
import os
import re
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from core.prompts import get_subtitle_trim_prompt, get_batch_trim_prompt
from core.tts_backend.estimate_duration import init_estimator, estimate_duration
from core.utils import *
from core.utils.models import *

console = Console()
speed_factor = load_key("speed_factor")

TRANS_SUBS_FOR_AUDIO_FILE = 'output/audio/trans_subs_for_audio.srt'
SRC_SUBS_FOR_AUDIO_FILE = 'output/audio/src_subs_for_audio.srt'
ESTIMATOR = None

def check_len_then_trim(text, duration):
    global ESTIMATOR
    if ESTIMATOR is None:
        ESTIMATOR = init_estimator()
    estimated_duration = estimate_duration(text, ESTIMATOR) / speed_factor['max']
    
    console.print(f"Subtitle text: {text}, "
                  f"[bold green]Estimated reading duration: {estimated_duration:.2f} seconds[/bold green]")

    if estimated_duration > duration:
        rprint(Panel(f"Estimated reading duration {estimated_duration:.2f} seconds exceeds given duration {duration:.2f} seconds, shortening...", title="Processing", border_style="yellow"))
        original_text = text
        prompt = get_subtitle_trim_prompt(text, duration)
        def valid_trim(response):
            if 'result' not in response:
                return {'status': 'error', 'message': 'No result in response'}
            return {'status': 'success', 'message': ''}
        try:    
            response = ask_gpt(prompt, resp_type='json', log_title='sub_trim', valid_def=valid_trim)
            shortened_text = response['result']
        except Exception:
            rprint("[bold red]🚫 AI refused to answer due to sensitivity, so manually remove punctuation[/bold red]")
            shortened_text = re.sub(r'[,.!?;:，。！？；：]', ' ', text).strip()
        rprint(Panel(f"Subtitle before shortening: {original_text}\nSubtitle after shortening: {shortened_text}", title="Subtitle Shortening Result", border_style="green"))
        return shortened_text
    else:
        return text

def batch_check_len_then_trim(texts, durations):
    """Batch version of check_len_then_trim. Returns list of (possibly shortened) texts."""
    global ESTIMATOR
    if ESTIMATOR is None:
        ESTIMATOR = init_estimator()

    needs_trim = []  # list of (original_index, text, duration)
    results = list(texts)  # start with originals

    for idx, (text, duration) in enumerate(zip(texts, durations)):
        estimated = estimate_duration(text, ESTIMATOR) / speed_factor['max']
        console.print(f"Subtitle text: {text}, "
                      f"[bold green]Estimated reading duration: {estimated:.2f} seconds[/bold green]")
        if estimated > duration:
            rprint(Panel(f"Estimated reading duration {estimated:.2f}s exceeds {duration:.2f}s, queued for shortening...",
                         title="Processing", border_style="yellow"))
            needs_trim.append((idx, text, duration))

    if not needs_trim:
        return results

    items = [(text, dur) for _, text, dur in needs_trim]
    prompt = get_batch_trim_prompt(items)

    def valid_batch_trim(response):
        for i in range(1, len(items) + 1):
            k = str(i)
            if k not in response or 'result' not in response.get(k, {}):
                return {'status': 'error', 'message': f'Missing key {k} or result field'}
        return {'status': 'success', 'message': ''}

    try:
        response = ask_gpt(prompt, resp_type='json', log_title='batch_sub_trim', valid_def=valid_batch_trim)
        for local_i, (orig_idx, original_text, _) in enumerate(needs_trim, 1):
            shortened = response[str(local_i)]['result']
            rprint(Panel(f"Before: {original_text}\nAfter:  {shortened}", title="Subtitle Shortening Result", border_style="green"))
            results[orig_idx] = shortened
    except Exception as e:
        rprint(f"[bold yellow]⚠️ Batch trim failed ({e}), falling back to punctuation removal[/bold yellow]")
        for orig_idx, text, _ in needs_trim:
            results[orig_idx] = re.sub(r'[,.!?;:，。！？；：]', ' ', text).strip()

    return results


def time_diff_seconds(t1, t2, base_date):
    """Calculate the difference in seconds between two time objects"""
    dt1 = datetime.datetime.combine(base_date, t1)
    dt2 = datetime.datetime.combine(base_date, t2)
    return (dt2 - dt1).total_seconds()

def process_srt():
    """Process srt file, generate audio tasks"""
    
    # Defensive check: if subtitle files are missing, try to regenerate them
    if not os.path.exists(TRANS_SUBS_FOR_AUDIO_FILE) or not os.path.exists(SRC_SUBS_FOR_AUDIO_FILE):
        print(f"Subtitle files missing, attempting to regenerate...")
        try:
            from core._6_gen_sub import align_timestamp_main
            align_timestamp_main()
        except Exception as e:
            raise FileNotFoundError(f"Subtitle files missing and regeneration failed: {e}")

    with open(TRANS_SUBS_FOR_AUDIO_FILE, 'r', encoding='utf-8') as file:
        content = file.read()
    
    with open(SRC_SUBS_FOR_AUDIO_FILE, 'r', encoding='utf-8') as src_file:
        src_content = src_file.read()
    
    subtitles = []
    src_subtitles = {}
    
    for block in src_content.strip().split('\n\n'):
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        if len(lines) < 3:
            continue
        
        number = int(lines[0])
        src_text = ' '.join(lines[2:])
        src_subtitles[number] = src_text
    
    for block in content.strip().split('\n\n'):
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        if len(lines) < 3:
            continue
        
        try:
            number = int(lines[0])
            start_time, end_time = lines[1].split(' --> ')
            start_time = datetime.datetime.strptime(start_time, '%H:%M:%S,%f').time()
            end_time = datetime.datetime.strptime(end_time, '%H:%M:%S,%f').time()
            duration = time_diff_seconds(start_time, end_time, datetime.date.today())
            text = ' '.join(lines[2:])
            # Remove content within parentheses (including English and Chinese parentheses) and brackets
            text = re.sub(r'\([^)]*\)', '', text).strip()
            text = re.sub(r'（[^）]*）', '', text).strip()
            text = re.sub(r'\[[^\]]*\]', '', text).strip()
            # Remove '-' character, can continue to add illegal characters that cause errors
            text = text.replace('-', '')

            # Add the original text from src_subs_for_audio.srt
            origin = src_subtitles.get(number, '')

        except ValueError as e:
            rprint(Panel(f"Unable to parse subtitle block '{block}', error: {str(e)}, skipping this subtitle block.", title="Error", border_style="red"))
            continue
        
        subtitles.append({'number': number, 'start_time': start_time, 'end_time': end_time, 'duration': duration, 'text': text, 'origin': origin})
    
    df = pd.DataFrame(subtitles)
    
    i = 0
    MIN_SUB_DUR = load_key("min_subtitle_duration")
    while i < len(df):
        today = datetime.date.today()
        if df.loc[i, 'duration'] < MIN_SUB_DUR:
            if i < len(df) - 1 and time_diff_seconds(df.loc[i, 'start_time'],df.loc[i+1, 'start_time'],today) < MIN_SUB_DUR:
                rprint(f"[bold yellow]Merging subtitles {i+1} and {i+2}[/bold yellow]")
                df.loc[i, 'text'] += ' ' + df.loc[i+1, 'text']
                df.loc[i, 'origin'] += ' ' + df.loc[i+1, 'origin']
                df.loc[i, 'end_time'] = df.loc[i+1, 'end_time']
                df.loc[i, 'duration'] = time_diff_seconds(df.loc[i, 'start_time'],df.loc[i, 'end_time'],today)
                df = df.drop(i+1).reset_index(drop=True)
            else:
                if i < len(df) - 1:  # Not the last audio
                    rprint(f"[bold blue]Extending subtitle {i+1} duration to {MIN_SUB_DUR} seconds[/bold blue]")
                    df.loc[i, 'end_time'] = (datetime.datetime.combine(today, df.loc[i, 'start_time']) + 
                                            datetime.timedelta(seconds=MIN_SUB_DUR)).time()
                    df.loc[i, 'duration'] = MIN_SUB_DUR
                else:
                    rprint(f"[bold red]The last subtitle {i+1} duration is less than {MIN_SUB_DUR} seconds, but not extending[/bold red]")
                i += 1
        else:
            i += 1
    
    df['start_time'] = df['start_time'].apply(lambda x: x.strftime('%H:%M:%S.%f')[:-3])
    df['end_time'] = df['end_time'].apply(lambda x: x.strftime('%H:%M:%S.%f')[:-3])

    # Save original text before trimming so gen_dub_chunks can match against trans.srt.
    df['orig_text'] = df['text'].copy()
    # AI-assisted text compression before TTS generation.
    efficiency_mode = load_key("efficiency_mode")
    from core.utils.progress_utils import get_progress, update_st_progress
    progress = get_progress()
    is_internal = not progress.live.is_started
    if is_internal: progress.start()

    task_desc = "✂️ 正在根据预估语速压缩字幕..."
    if efficiency_mode:
        df['text'] = batch_check_len_then_trim(df['text'].tolist(), df['duration'].tolist())
    else:
        task = progress.add_task(task_desc, total=len(df))
        def process_with_progress(row):
            res = check_len_then_trim(row['text'], row['duration'])
            progress.advance(task)
            update_st_progress(int(progress.tasks[task].completed), len(df), task_desc)
            return res
        df['text'] = df.apply(process_with_progress, axis=1)
        if not is_internal: progress.remove_task(task)

    if is_internal: progress.stop()

    # Apply video slow-down factor: scale all timestamps so that each subtitle window is
    # proportionally wider. Enables the audio to fit with less speed-up.
    # The output video must be slowed by the same factor in _12_dub_to_vid.py.
    try:
        video_slow_factor = load_key("video_slow_factor")
    except KeyError:
        video_slow_factor = 1.0
    if video_slow_factor != 1.0:
        def _scale_time_str(time_str: str) -> str:
            t = datetime.datetime.strptime(time_str, '%H:%M:%S.%f').time()
            total_secs = t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6
            scaled = total_secs / video_slow_factor
            td = datetime.timedelta(seconds=scaled)
            total_us = int(td.total_seconds() * 1_000_000)
            h = total_us // 3_600_000_000
            m = (total_us % 3_600_000_000) // 60_000_000
            s = (total_us % 60_000_000) // 1_000_000
            ms = (total_us % 1_000_000) // 1_000
            return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

        df['start_time'] = df['start_time'].apply(_scale_time_str)
        df['end_time'] = df['end_time'].apply(_scale_time_str)
        df['duration'] = df['duration'] / video_slow_factor
        rprint(f"[bold cyan]🎬 Timestamps scaled by 1/{video_slow_factor:.2f} (video_slow_factor={video_slow_factor})[/bold cyan]")

    return df

def gen_audio_task_main():
    # Auto-invalidate stale task file if subtitle SRT was regenerated after it
    if os.path.exists(_8_1_AUDIO_TASK) and os.path.exists(TRANS_SUBS_FOR_AUDIO_FILE):
        if os.path.getmtime(TRANS_SUBS_FOR_AUDIO_FILE) > os.path.getmtime(_8_1_AUDIO_TASK):
            os.remove(_8_1_AUDIO_TASK)
            rprint("[yellow]🔄 Subtitle SRT is newer than audio task file, regenerating...[/yellow]")
    if os.path.exists(_8_1_AUDIO_TASK):
        rprint(f"[yellow]⚠️ File <{_8_1_AUDIO_TASK}> already exists, skip <gen_audio_task_main> step.[/yellow]")
        return
    df = process_srt()
    console.print(df)
    df.to_excel(_8_1_AUDIO_TASK, index=False)
    rprint(Panel(f"Successfully generated {_8_1_AUDIO_TASK}", title="Success", border_style="green"))

if __name__ == '__main__':
    gen_audio_task_main()