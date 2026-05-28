import os,sys
import glob
import re
import subprocess
from core.utils import *

def sanitize_filename(filename):
    # Remove or replace illegal characters
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    # Ensure filename doesn't start or end with a dot or space
    filename = filename.strip('. ')
    # Use default name if filename is empty
    return filename if filename else 'video'

def update_ytdlp():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "yt-dlp"])
        if 'yt_dlp' in sys.modules:
            del sys.modules['yt_dlp']
        rprint("[green]yt-dlp updated[/green]")
    except subprocess.CalledProcessError as e:
        rprint("[yellow]Warning: Failed to update yt-dlp: {e}[/yellow]")
    from yt_dlp import YoutubeDL
    return YoutubeDL

def download_video_ytdlp(url, save_path='output', resolution='1080'):
    os.makedirs(save_path, exist_ok=True)
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best' if resolution == 'best' else f'bestvideo[height<={resolution}]+bestaudio/best[height<={resolution}]',
        'outtmpl': f'{save_path}/%(title)s.%(ext)s',
        'noplaylist': True,
        'writethumbnail': True,
        'postprocessors': [{'key': 'FFmpegThumbnailsConvertor', 'format': 'jpg'}],
    }

    # Read Youtube Cookie File
    cookies_path = load_key("youtube.cookies_path")
    if os.path.exists(cookies_path):
        ydl_opts["cookiefile"] = str(cookies_path)

    # Get YoutubeDL class after updating
    YoutubeDL = update_ytdlp()

    # Fetch and save metadata (title and cover) first
    try:
        ydl_opts_info = {'skip_download': True, 'noplaylist': True}
        if os.path.exists(cookies_path):
            ydl_opts_info["cookiefile"] = str(cookies_path)
        with YoutubeDL(ydl_opts_info) as ydl_info:
            info = ydl_info.extract_info(url, download=False)
            title = info.get('title', 'video')
            thumbnail_url = info.get('thumbnail')
            
            with open(os.path.join(save_path, 'video_title.txt'), 'w', encoding='utf-8') as f:
                f.write(title)
                
            if thumbnail_url:
                import requests
                res = requests.get(thumbnail_url, timeout=10)
                if res.status_code == 200:
                    with open(os.path.join(save_path, 'video_cover.jpg'), 'wb') as f:
                        f.write(res.content)
    except Exception as e:
        print(f"Warning: Failed to fetch metadata in download_video: {e}")

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    # Check and rename files after download
    for file in os.listdir(save_path):
        if os.path.isfile(os.path.join(save_path, file)):
            filename, ext = os.path.splitext(file)
            new_filename = sanitize_filename(filename)
            if new_filename != filename:
                os.rename(os.path.join(save_path, file), os.path.join(save_path, new_filename + ext))

def find_video_files(save_path='output'):
    video_files = [file for file in glob.glob(save_path + "/*") if os.path.splitext(file)[1][1:].lower() in load_key("allowed_video_formats")]
    # change \\ to /, this happen on windows
    if sys.platform.startswith('win'):
        video_files = [file.replace("\\", "/") for file in video_files]
    
    # First try to find original video files (not processed outputs)
    original_videos = [file for file in video_files if not file.startswith("output/output")]
    
    if len(original_videos) == 1:
        return original_videos[0]
    elif len(original_videos) == 0:
        # If no original videos, look for processed video files as fallback
        # Prioritize files with specific patterns: output_sub.mp4 > output_dub.mp4 > others
        processed_videos = [file for file in video_files if file.startswith("output/output")]
        if len(processed_videos) == 1:
            return processed_videos[0]
        elif len(processed_videos) > 1:
            # Try to find the most appropriate processed video file
            sub_videos = [f for f in processed_videos if "_sub" in f]
            if sub_videos:
                return sub_videos[0]
            dub_videos = [f for f in processed_videos if "_dub" in f]
            if dub_videos:
                return dub_videos[0]
            # Return the first one if no specific pattern matches
            return processed_videos[0]
    
    # If we have multiple original videos or no videos at all, raise the original error
    all_videos = original_videos if original_videos else video_files
    raise ValueError(f"Number of videos found {len(all_videos)} is not unique. Please check.")

def download_subtitle_ytdlp(url, save_path='output'):
    """
    Downloads English subtitles for the given YouTube URL.
    Prefers manual/original English subtitles, falls back to auto-generated English subtitles.
    If neither is found, returns None.
    Otherwise, returns a tuple (subtitle_file_path, subtitle_type).
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Get YoutubeDL class
    YoutubeDL = update_ytdlp()
    
    ydl_opts_info = {
        'skip_download': True,
        'noplaylist': True,
    }
    cookies_path = load_key("youtube.cookies_path")
    if os.path.exists(cookies_path):
        ydl_opts_info["cookiefile"] = str(cookies_path)
        
    with YoutubeDL(ydl_opts_info) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
        except Exception as e:
            rprint(f"[red]Failed to extract video info: {e}[/red]")
            return None, "extract_failed"

    # Save video title and cover
    try:
        title = info.get('title', 'video')
        thumbnail_url = info.get('thumbnail')
        
        with open(os.path.join(save_path, 'video_title.txt'), 'w', encoding='utf-8') as f:
            f.write(title)
            
        if thumbnail_url:
            import requests
            res = requests.get(thumbnail_url, timeout=10)
            if res.status_code == 200:
                with open(os.path.join(save_path, 'video_cover.jpg'), 'wb') as f:
                    f.write(res.content)
    except Exception as e:
        rprint(f"[yellow]Warning: Failed to save metadata: {e}[/yellow]")

    subtitles = info.get('subtitles', {}) or {}
    automatic_captions = info.get('automatic_captions', {}) or {}

    selected_lang = None
    is_original = False

    # Prefer 'en' exactly, then anything starting with 'en-' or 'en_'
    def get_best_en_lang(langs_keys):
        langs = list(langs_keys)
        if 'en' in langs:
            return 'en'
        en_langs = [l for l in langs if l.lower().startswith('en-') or l.lower().startswith('en_')]
        if en_langs:
            return en_langs[0]
        return None

    selected_lang = get_best_en_lang(subtitles.keys())
    if selected_lang:
        is_original = True
    else:
        selected_lang = get_best_en_lang(automatic_captions.keys())
        if selected_lang:
            is_original = False

    if not selected_lang:
        rprint("[yellow]No English subtitles (original or auto-generated) found.[/yellow]")
        return None, "no_subtitles"

    sub_type_str = "original" if is_original else "auto-generated"
    rprint(f"[green]Found {sub_type_str} English subtitle with language code: {selected_lang}[/green]")

    # Download options
    ydl_opts_download = {
        'skip_download': True,
        'writesubtitles': is_original,
        'writeautomaticsub': not is_original,
        'subtitleslangs': [selected_lang],
        'outtmpl': f'{save_path}/youtube_subtitle.%(ext)s',
        'noplaylist': True,
    }
    if os.path.exists(cookies_path):
        ydl_opts_download["cookiefile"] = str(cookies_path)

    # Clean existing temp subtitle files to avoid collision
    for f in glob.glob(f"{save_path}/youtube_subtitle.*"):
        try:
            os.remove(f)
        except Exception:
            pass

    with YoutubeDL(ydl_opts_download) as ydl:
        try:
            ydl.download([url])
        except Exception as e:
            rprint(f"[red]Failed to download subtitle: {e}[/red]")
            return None, "download_failed"

    # Find the downloaded file
    downloaded_files = glob.glob(f"{save_path}/youtube_subtitle.*")
    if not downloaded_files:
        rprint("[red]Subtitle downloaded but file not found on disk.[/red]")
        return None, "file_not_found"

    sub_file = downloaded_files[0]
    ext = os.path.splitext(sub_file)[1].lower() # .vtt or .srt
    
    # Rename to a standard name
    target_path = f"{save_path}/youtube_subtitle{ext}"
    if os.path.exists(target_path):
        os.remove(target_path)
    os.rename(sub_file, target_path)

    rprint(f"[green]Subtitle successfully saved to: {target_path}[/green]")
    return target_path, sub_type_str


if __name__ == '__main__':
    # Example usage
    url = input('Please enter the URL of the video you want to download: ')
    resolution = input('Please enter the desired resolution (360/480/720/1080, default 1080): ')
    resolution = int(resolution) if resolution.isdigit() else 1080
    download_video_ytdlp(url, resolution=resolution)
    print(f"🎥 Video has been downloaded to {find_video_files()}")
