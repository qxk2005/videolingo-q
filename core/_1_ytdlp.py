import os,sys
import glob
import re
import subprocess
from core.utils import *
import base64
import hashlib
import time

try:
    from Crypto.Cipher import AES
    from Crypto.Util.Padding import pad
except ImportError:
    try:
        from Cryptodome.Cipher import AES
        from Cryptodome.Util.Padding import pad
    except ImportError:
        AES = None
        pad = None

def srt_time_to_sec(ts: str) -> float:
    ts = ts.strip().replace(',', '.')
    parts = ts.split(':')
    return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])

def sec_to_srt_time(sec: float) -> str:
    hours = int(sec // 3600)
    minutes = int((sec % 3600) // 60)
    seconds = int(sec % 60)
    milliseconds = int(round((sec - int(sec)) * 1000))
    if milliseconds == 1000:
        seconds += 1
        milliseconds = 0
        if seconds == 60:
            minutes += 1
            seconds = 0
            if minutes == 60:
                hours += 1
                minutes = 0
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def resolve_srt_overlaps_file(target_path):
    if not os.path.exists(target_path):
        return
    try:
        rprint("[cyan]Applying timeline de-overlapping to SRT subtitle...[/cyan]")
        with open(target_path, 'r', encoding='utf-8') as sf:
            srt_content = sf.read()
        
        # Split by double newline to get individual blocks
        blocks = re.split(r'\n\s*\n', srt_content.strip())
        items = []
        
        for block in blocks:
            lines = [l.strip() for l in block.strip().splitlines() if l.strip()]
            if not lines:
                continue
            ts_match = ts_idx = None
            for idx, line in enumerate(lines):
                m = re.match(r'([\d:,]+)\s+-->\s+([\d:,]+)', line)
                if m:
                    ts_match, ts_idx = m, idx
                    break
            if not ts_match:
                continue
                
            start_sec = srt_time_to_sec(ts_match.group(1))
            end_sec = srt_time_to_sec(ts_match.group(2))
            text = "\n".join(lines[ts_idx + 1:])
            
            items.append({
                'start': start_sec,
                'end': end_sec,
                'text': text
            })
            
        if not items:
            return
            
        # Resolve overlaps
        for i in range(len(items) - 1):
            curr = items[i]
            nxt = items[i+1]
            if curr['end'] > nxt['start']:
                new_end = nxt['start']
                if new_end < curr['start']:
                    new_end = curr['start']
                curr['end'] = new_end
                
        # Format back to SRT string
        new_blocks = []
        for idx, item in enumerate(items):
            start_str = sec_to_srt_time(item['start'])
            end_str = sec_to_srt_time(item['end'])
            block_str = f"{idx + 1}\n{start_str} --> {end_str}\n{item['text']}"
            new_blocks.append(block_str)
            
        clean_srt = "\n\n".join(new_blocks) + "\n"
        with open(target_path, 'w', encoding='utf-8') as sf:
            sf.write(clean_srt)
        rprint("[green]Timeline de-overlapping applied successfully.[/green]")
    except Exception as e:
        rprint(f"[yellow]Warning: Failed to apply timeline de-overlapping: {e}[/yellow]")

def get_youtube_video_id(url):
    pattern = r'(?:https?://)?(?:www\.)?(?:youtube\.com/(?:watch\?v=|embed/)|youtu\.be/)([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    return None

def download_subtitle_from_thirdparty(url, save_path='output'):
    import urllib.parse
    if AES is None or pad is None:
        rprint("[yellow]Crypto libraries are not installed. Skipping third-party download.[/yellow]")
        return None
        
    video_id = get_youtube_video_id(url)
    if not video_id:
        rprint("[yellow]Invalid YouTube URL for third-party downloader.[/yellow]")
        return None
    
    session = requests = None
    import requests # local import to ensure availability
    
    session = requests.Session()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    }
    session.headers.update(headers)
    
    # 1. Fetch main page
    page_url = f"https://www.downloadyoutubesubtitles.com/zh/?u=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3D{video_id}"
    
    try:
        r = session.get(page_url, timeout=10)
        if r.status_code != 200:
            rprint(f"[yellow]Failed to fetch third-party page: {r.status_code}[/yellow]")
            return None
        html = r.text
    except Exception as e:
        rprint(f"[yellow]Network error when fetching third-party page: {e}[/yellow]")
        return None
        
    # 2. Parse variables
    sid_match = re.search(r"var sid='([^']+)';", html)
    hash_match = re.search(r"var hash='([^']+)';", html)
    hl_match = re.search(r"var hl='([^']+)';", html)
    tutoken_match = re.search(r"var tutoken='([^']+)';", html)
    htoken_match = re.search(r"var htoken='([^']+)';", html)
    
    sid = sid_match.group(1) if sid_match else None
    hash_val = hash_match.group(1) if hash_match else None
    hl = hl_match.group(1) if hl_match else None
    tutoken = tutoken_match.group(1) if tutoken_match else None
    htoken = htoken_match.group(1) if htoken_match else None
    
    if not all([sid, hash_val, hl, tutoken, htoken]):
        rprint("[yellow]Could not find essential parameters in third-party page HTML.[/yellow]")
        return None
        
    # Extract pwx from _$_9361 array
    array_match = re.search(r"var _\$_9361\s*=\s*(\[[^\]]+\]);", html)
    if not array_match:
        rprint("[yellow]Could not find obfuscated array in third-party page HTML.[/yellow]")
        return None
        
    try:
        arr = eval(array_match.group(1))
        pwx = urllib.parse.unquote(arr[11])
    except Exception as e:
        rprint(f"[yellow]Failed to parse pwx array: {e}[/yellow]")
        return None
        
    # 3. Encrypt token
    def encrypt_token(mss, password):
        salt = os.urandom(32)
        key = hashlib.pbkdf2_hmac('sha1', password.encode('utf-8'), salt, 100, dklen=32)
        iv = os.urandom(16)
        cipher = AES.new(key, AES.MODE_CBC, iv)
        ciphertext = cipher.encrypt(pad(mss.encode('utf-8'), AES.block_size))
        combined_hex = salt.hex() + iv.hex() + ciphertext.hex()
        return base64.b64encode(bytes.fromhex(combined_hex)).decode('utf-8')
        
    chrono = int(time.time() * 1000)
    mss = f"https://www.youtube.com/watch?v={video_id};;{chrono}"
    token = encrypt_token(mss, pwx)
    
    # 4. Post to api.php
    api_headers = {
        'Referer': page_url,
        'Origin': 'https://www.downloadyoutubesubtitles.com',
        'X-Requested-With': 'XMLHttpRequest'
    }
    
    api_data = {
        'token': token,
        'sid': sid,
        'hash': hash_val,
        'hl': hl,
        'tutoken': tutoken,
        'htoken': htoken
    }
    
    try:
        api_resp = session.post("https://www.downloadyoutubesubtitles.com/api.php", data=api_data, headers=api_headers, timeout=10)
        if api_resp.status_code != 200:
            rprint(f"[yellow]Third-party API post failed: {api_resp.status_code}[/yellow]")
            return None
        api_html = api_resp.text
    except Exception as e:
        rprint(f"[yellow]Network error when calling third-party API: {e}[/yellow]")
        return None
        
    # 5. Extract SRT link
    srt_links = re.findall(r"data-href='(/get2\.php\?[^']+format=srt[^']*)'", api_html)
    if not srt_links:
        srt_links = re.findall(r'data-href="(/get2\.php\?[^"]+format=srt[^"]*)"', api_html)
        
    if not srt_links:
        rprint("[yellow]No SRT links found in third-party API response.[/yellow]")
        return None
        
    best_link = None
    for link in srt_links:
        if "hl=a.en" in link or "hl=en" in link:
            best_link = link
            break
            
    if not best_link:
        best_link = srt_links[0]
        
    download_url = f"https://www.downloadyoutubesubtitles.com{best_link}"
    rprint(f"[green]Downloading clean SRT from third-party: {download_url}[/green]")
    
    # 6. Download the SRT file
    session.cookies.set('downloadToken', 'butaen', domain='downloadyoutubesubtitles.com', path='/')
    download_headers = {
        'Referer': page_url,
    }
    
    try:
        sub_resp = session.get(download_url, headers=download_headers, timeout=10)
        if sub_resp.status_code != 200:
            rprint(f"[yellow]Failed to download third-party subtitle content: {sub_resp.status_code}[/yellow]")
            return None
        
        os.makedirs(save_path, exist_ok=True)
        target_path = os.path.join(save_path, "youtube_subtitle.srt")
        with open(target_path, "w", encoding="utf-8") as sf:
            sf.write(sub_resp.text)
        rprint(f"[green]Successfully saved third-party subtitle to: {target_path}[/green]")
        return target_path
    except Exception as e:
        rprint(f"[yellow]Error downloading third-party SRT content: {e}[/yellow]")
        return None


def sanitize_filename(filename):
    # Remove or replace illegal characters
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    # Ensure filename doesn't start or end with a dot or space
    filename = filename.strip('. ')
    # Use default name if filename is empty
    return filename if filename else 'video'

def _add_cookies_options(ydl_opts):
    cookies_path = load_key("youtube.cookies_path")
    if cookies_path and os.path.exists(cookies_path):
        ydl_opts["cookiefile"] = str(cookies_path)
    
    try:
        cookies_from_browser = load_key("youtube.cookies_from_browser")
        if cookies_from_browser:
            ydl_opts["cookiesfrombrowser"] = (cookies_from_browser,)
    except Exception:
        pass

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
        'retries': 20,
        'fragment_retries': 20,
        'nocheckcertificate': True,
    }

    _add_cookies_options(ydl_opts)

    # Get YoutubeDL class after updating
    YoutubeDL = update_ytdlp()

    # Fetch and save metadata (title and cover) first
    try:
        ydl_opts_info = {'skip_download': True, 'noplaylist': True, 'nocheckcertificate': True}
        _add_cookies_options(ydl_opts_info)
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
        if "confirm you're not a bot" in str(e) or "Sign in" in str(e):
            rprint("[bold yellow]💡 提示: YouTube 触发了人机验证拦截。您可以在 config.yaml 中配置 'youtube.cookies_from_browser' (如 'chrome') 自动读取浏览器 cookies，或者配置 'cookies_path'。[/bold yellow]")

    with YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([url])
        except Exception as e:
            if "confirm you're not a bot" in str(e) or "Sign in" in str(e):
                rprint("[bold yellow]💡 提示: YouTube 触发了人机验证拦截。您可以在 config.yaml 中配置 'youtube.cookies_from_browser' (如 'chrome') 自动读取浏览器 cookies，或者配置 'cookies_path'。[/bold yellow]")
            raise e
    
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

    # Try downloading clean srt from third party website first
    try:
        rprint("[cyan]尝试从第三方网站获取干净的 SRT 字幕...[/cyan]")
        thirdparty_path = download_subtitle_from_thirdparty(url, save_path)
        if thirdparty_path and os.path.exists(thirdparty_path):
            rprint("[green]成功从第三方获取到干净的 SRT 字幕！[/green]")
            resolve_srt_overlaps_file(thirdparty_path)
            return thirdparty_path, "clean-srt"
        rprint("[yellow]从第三方获取字幕失败或跳过，将降级使用 yt-dlp...[/yellow]")
    except Exception as e:
        rprint(f"[yellow]第三方字幕下载发生异常: {e}，将降级使用 yt-dlp...[/yellow]")
    
    # Get YoutubeDL class
    YoutubeDL = update_ytdlp()
    
    ydl_opts_info = {
        'skip_download': True,
        'noplaylist': True,
        'nocheckcertificate': True,
    }
    _add_cookies_options(ydl_opts_info)
        
    with YoutubeDL(ydl_opts_info) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
        except Exception as e:
            rprint(f"[red]Failed to extract video info: {e}[/red]")
            if "confirm you're not a bot" in str(e) or "Sign in" in str(e):
                rprint("[bold yellow]💡 提示: YouTube 触发了人机验证拦截。您可以在 config.yaml 中配置 'youtube.cookies_from_browser' (如 'chrome') 自动读取浏览器 cookies，或者配置 'cookies_path'。[/bold yellow]")
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
    # Download options
    ydl_opts_download = {
        'skip_download': True,
        'writesubtitles': is_original,
        'writeautomaticsub': not is_original,
        'subtitleslangs': [selected_lang],
        'outtmpl': f'{save_path}/youtube_subtitle.%(ext)s',
        'noplaylist': True,
        'postprocessors': [{
            'key': 'FFmpegSubtitlesConvertor',
            'format': 'srt',
        }],
        'nocheckcertificate': True,
    }
    _add_cookies_options(ydl_opts_download)

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
            if "confirm you're not a bot" in str(e) or "Sign in" in str(e):
                rprint("[bold yellow]💡 提示: YouTube 触发了人机验证拦截。您可以在 config.yaml 中配置 'youtube.cookies_from_browser' (如 'chrome') 自动读取浏览器 cookies，或者配置 'cookies_path'。[/bold yellow]")
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

    if ext == '.srt':
        resolve_srt_overlaps_file(target_path)

    rprint(f"[green]Subtitle successfully saved to: {target_path}[/green]")
    return target_path, sub_type_str


if __name__ == '__main__':
    # Example usage
    url = input('Please enter the URL of the video you want to download: ')
    resolution = input('Please enter the desired resolution (360/480/720/1080, default 1080): ')
    resolution = int(resolution) if resolution.isdigit() else 1080
    download_video_ytdlp(url, resolution=resolution)
    print(f"🎥 Video has been downloaded to {find_video_files()}")
