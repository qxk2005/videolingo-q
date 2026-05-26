import os
import pandas as pd
import subprocess
from pydub import AudioSegment
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.console import Console
from core.utils import *
from core.utils.models import *
from core.tts_backend.tts_main import tts_main, is_audio_valid
from core._10_gen_audio import adjust_audio_speed, TEMP_FILE_TEMPLATE
from core.asr_backend.audio_preprocess import get_audio_duration
console = Console()

DUB_VOCAL_FILE = 'output/dub.mp3'

DUB_SUB_FILE = 'output/dub.srt'
OUTPUT_FILE_TEMPLATE = f"{_AUDIO_SEGS_DIR}/{{}}.wav"

def load_and_flatten_data(excel_file):
    """Load and flatten Excel data"""
    df = pd.read_excel(excel_file)
    lines = [eval(line) if isinstance(line, str) else line for line in df['lines'].tolist()]
    lines = [item for sublist in lines for item in sublist]
    
    new_sub_times = [eval(time) if isinstance(time, str) else time for time in df['new_sub_times'].tolist()]
    new_sub_times = [item for sublist in new_sub_times for item in sublist]
    
    return df, lines, new_sub_times

def get_audio_files(df):
    """Generate a list of audio file paths"""
    audios = []
    for index, row in df.iterrows():
        number = row['number']
        line_count = len(eval(row['lines']) if isinstance(row['lines'], str) else row['lines'])
        for line_index in range(line_count):
            temp_file = OUTPUT_FILE_TEMPLATE.format(f"{number}_{line_index}")
            audios.append(temp_file)
    return audios

def process_audio_segment(audio_file):
    """Process a single audio segment with MP3 compression"""
    # Check if the file is empty or too small (e.g. 44-byte empty WAV header) to bypass directly to silence
    if os.path.exists(audio_file) and os.path.getsize(audio_file) < 100:
        console.print(f"[yellow]⚠️ Audio file {audio_file} is empty/too small, creating silent segment[/yellow]")
        return AudioSegment.silent(duration=100, frame_rate=16000)
        
    # First check if the audio file is valid by trying to get its info
    try:
        # Quick check using ffmpeg to see if file is readable
        check_cmd = ['ffmpeg', '-i', audio_file, '-f', 'null', '-']
        result = subprocess.run(check_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
        
        # If ffmpeg can't process the file, create a silent segment
        if result.returncode != 0:
            console.print(f"[yellow]⚠️ Audio file {audio_file} is corrupted/empty, creating silent segment[/yellow]")
            return AudioSegment.silent(duration=100, frame_rate=16000)  # 100ms of silence
            
    except (subprocess.TimeoutExpired, Exception) as e:
        console.print(f"[yellow]⚠️ Error checking audio file {audio_file}: {e}, creating silent segment[/yellow]")
        return AudioSegment.silent(duration=100, frame_rate=16000)
    
    temp_file = f"{os.path.splitext(audio_file)[0]}_temp.mp3"
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-i', audio_file,
        '-ar', '16000',
        '-ac', '1',
        '-b:a', '64k',
        temp_file
    ]
    
    try:
        subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        audio_segment = AudioSegment.from_mp3(temp_file)
        os.remove(temp_file)
        return audio_segment
    except (subprocess.CalledProcessError, Exception) as e:
        console.print(f"[yellow]⚠️ Failed to process audio file {audio_file}: {e}, creating silent segment[/yellow]")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return AudioSegment.silent(duration=100, frame_rate=16000)  # 100ms of silence

def merge_audio_segments(audios, new_sub_times, sample_rate):
    merged_audio = AudioSegment.silent(duration=0, frame_rate=sample_rate)
    
    from core.utils.progress_utils import get_progress, update_st_progress
    progress = get_progress()
    is_internal = not progress.live.is_started
    if is_internal: progress.start()

    task_desc = "🎵 正在合并音频分段..."
    task = progress.add_task(task_desc, total=len(audios))
    
    for i, (audio_file, time_range) in enumerate(zip(audios, new_sub_times)):
        if not os.path.exists(audio_file):
            console.print(f"[bold yellow]⚠️  Warning: File {audio_file} does not exist, skipping...[/bold yellow]")
            progress.advance(task)
            update_st_progress(i + 1, len(audios), task_desc)
            continue
            
        audio_segment = process_audio_segment(audio_file)
        start_time, end_time = time_range
        
        # Add silence segment
        if i > 0:
            prev_end = new_sub_times[i-1][1]
            silence_duration = start_time - prev_end
            if silence_duration > 0:
                silence = AudioSegment.silent(duration=int(silence_duration * 1000), frame_rate=sample_rate)
                merged_audio += silence
        elif start_time > 0:
            silence = AudioSegment.silent(duration=int(start_time * 1000), frame_rate=sample_rate)
            merged_audio += silence
            
        merged_audio += audio_segment
        progress.advance(task)
        update_st_progress(i + 1, len(audios), task_desc)
    
    if is_internal:
        progress.stop()
    else:
        progress.remove_task(task)
    
    return merged_audio

def create_srt_subtitle():
    df, lines, new_sub_times = load_and_flatten_data(_8_1_AUDIO_TASK)
    
    with open(DUB_SUB_FILE, 'w', encoding='utf-8') as f:
        for i, ((start_time, end_time), line) in enumerate(zip(new_sub_times, lines), 1):
            start_str = f"{int(start_time//3600):02d}:{int((start_time%3600)//60):02d}:{int(start_time%60):02d},{int((start_time*1000)%1000):03d}"
            end_str = f"{int(end_time//3600):02d}:{int((end_time%3600)//60):02d}:{int(end_time%60):02d},{int((end_time*1000)%1000):03d}"
            
            f.write(f"{i}\n")
            f.write(f"{start_str} --> {end_str}\n")
            f.write(f"{line}\n\n")
    
    rprint(f"[bold green]✅ Subtitle file created: {DUB_SUB_FILE}[/bold green]")

def verify_and_fix_audios(df, audios):
    """Scan all output audio segments in segs/, validate them, and self-heal any missing or invalid files."""
    console.print("[bold cyan]🔍 [Integrity Check] Verifying all audio segments...[/bold cyan]")
    fixed_count = 0
    checked_count = 0
    
    for index, row in df.iterrows():
        number = row['number']
        lines = eval(row['lines']) if isinstance(row['lines'], str) else row['lines']
        new_sub_times = eval(row['new_sub_times']) if isinstance(row['new_sub_times'], str) else row['new_sub_times']
        
        for line_index, line in enumerate(lines):
            checked_count += 1
            output_file = OUTPUT_FILE_TEMPLATE.format(f"{number}_{line_index}")
            
            # Check if this segment audio is valid
            is_valid = False
            if os.path.exists(output_file):
                is_valid = is_audio_valid(output_file, line)
                
            if not is_valid:
                console.print(f"[yellow]⚠️ [Corrupted/Missing Segment] Audio '{output_file}' is invalid or missing. Triggering self-healing...[/yellow]")
                
                # Step 1: Clean up existing corrupted output file if it exists
                if os.path.exists(output_file):
                    try: os.remove(output_file)
                    except Exception: pass
                    
                # Step 2: Ensure we have a healthy raw temp wav file first
                temp_file = TEMP_FILE_TEMPLATE.format(f"{number}_{line_index}")
                if os.path.exists(temp_file):
                    if not is_audio_valid(temp_file, line):
                        try: os.remove(temp_file)
                        except Exception: pass
                
                # Step 3: Re-generate the raw temp WAV audio via tts_main
                tts_main(line, temp_file, number, df)
                
                # Step 4: Re-apply the speed adjustment dynamically
                if os.path.exists(temp_file):
                    raw_dur = get_audio_duration(temp_file)
                    start_time, end_time = new_sub_times[line_index]
                    target_dur = end_time - start_time
                    
                    if target_dur <= 0:
                        target_dur = 2.0  # Fallback duration to prevent divide-by-zero
                        
                    sf = raw_dur / target_dur
                    sf = max(0.5, min(sf, 4.0)) # Clamp to FFmpeg safe bounds
                    
                    try:
                        adjust_audio_speed(temp_file, output_file, sf)
                        if is_audio_valid(output_file, line):
                            console.print(f"[bold green]✨ [Self-Healed Success] Successfully reconstructed and speed-adjusted <{output_file}> (sf={sf:.3f})[/bold green]")
                            fixed_count += 1
                        else:
                            console.print(f"[red]❌ [Self-Healed Failed] Re-generated audio <{output_file}> still failed validity check![/red]")
                    except Exception as e:
                        console.print(f"[red]❌ [Self-Healed Failed] Failed to adjust speed for re-generated audio: {e}[/red]")
                else:
                    console.print(f"[red]❌ [Self-Healed Failed] Failed to generate raw WAV for <{temp_file}>![/red]")
                    
    if fixed_count > 0:
        console.print(f"[bold green]🎉 [Integrity Checked] Scanned {checked_count} segments, successfully self-healed {fixed_count} corrupted/missing audio files![/bold green]")
    else:
        console.print(f"[bold green]✅ [Integrity Checked] Scanned {checked_count} segments. All audio files are 100% healthy and valid.[/bold green]")

def merge_full_audio():
    """Main function: Process the complete audio merging process"""
    console.print("\n[bold cyan]🎬 Starting audio merging process...[/bold cyan]")
    
    with console.status("[bold cyan]📊 Loading data from Excel...[/bold cyan]"):
        df, lines, new_sub_times = load_and_flatten_data(_8_1_AUDIO_TASK)
    console.print("[bold green]✅ Data loaded successfully[/bold green]")
    
    with console.status("[bold cyan]🔍 Getting audio file list...[/bold cyan]"):
        audios = get_audio_files(df)
    console.print(f"[bold green]✅ Found {len(audios)} audio segments[/bold green]")
    
    with console.status("[bold cyan]📝 Generating subtitle file...[/bold cyan]"):
        create_srt_subtitle()
        
    # 🎯 Check audio segments integrity and auto-repair any invalid files before merging
    verify_and_fix_audios(df, audios)
    
    if not os.path.exists(audios[0]):
        console.print(f"[bold red]❌ Error: First audio file {audios[0]} does not exist![/bold red]")
        return
    
    sample_rate = 16000
    console.print(f"[bold green]✅ Sample rate: {sample_rate}Hz[/bold green]")

    console.print("[bold cyan]🔄 Starting audio merge process...[/bold cyan]")
    merged_audio = merge_audio_segments(audios, new_sub_times, sample_rate)
    
    with console.status("[bold cyan]💾 Exporting final audio file...[/bold cyan]"):
        merged_audio = merged_audio.set_frame_rate(16000).set_channels(1)
        merged_audio.export(DUB_VOCAL_FILE, format="mp3", parameters=["-b:a", "64k"])
    console.print(f"[bold green]✅ Audio file successfully merged![/bold green]")
    console.print(f"[bold green]📁 Output file: {DUB_VOCAL_FILE}[/bold green]")

if __name__ == "__main__":
    merge_full_audio()