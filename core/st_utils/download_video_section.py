import os
import re
import shutil
import subprocess
from time import sleep

import streamlit as st
from core._1_ytdlp import download_video_ytdlp, find_video_files, download_subtitle_ytdlp
from core.utils import load_key, get_ffmpeg_video_encoder
from translations.translations import translate as t

OUTPUT_DIR = "output"

def save_youtube_url(url: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "youtube_url.txt"), "w", encoding="utf-8") as f:
        f.write(url.strip())

def load_youtube_url() -> str:
    path = os.path.join(OUTPUT_DIR, "youtube_url.txt")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            pass
    return ""

def delete_existing_videos():
    try:
        video_file = find_video_files()
        if os.path.exists(video_file):
            os.remove(video_file)
    except Exception:
        pass
    # Clean other potential video files in output to prevent dual-video collision
    if os.path.exists(OUTPUT_DIR):
        for f in os.listdir(OUTPUT_DIR):
            if f.endswith(('.mp4', '.mkv', '.avi', '.webm', '.mp3', '.wav')):
                try:
                    os.remove(os.path.join(OUTPUT_DIR, f))
                except Exception:
                    pass

def srt_to_vtt(srt_path: str, vtt_path: str):
    """Convert SRT subtitle file to VTT format for browser HTML5 player compatibility."""
    if not os.path.exists(srt_path):
        return
    try:
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # Replace timestamp commas with dots for VTT format compliance
        content_vtt = re.sub(r'(\d{2}:\d{2}:\d{2}),(\d{3})', r'\1.\2', content)
        # Prepend WEBVTT header
        if not content_vtt.strip().startswith("WEBVTT"):
            content_vtt = "WEBVTT\n\n" + content_vtt.lstrip()
        with open(vtt_path, 'w', encoding='utf-8') as f:
            f.write(content_vtt)
    except Exception as e:
        print(f"Failed to convert srt to vtt: {e}")

def download_video_section():
    st.header(t("a. Download or Upload Video"))
    with st.container(border=True):
        # 1. 尝试探测已有的视频文件
        video_exists = False
        video_file = None
        try:
            video_file = find_video_files()
            video_exists = True
        except:
            video_exists = False
 
        # 2. 如果已有视频，渲染视频播放控件
        if video_exists and video_file:
            srt_path = os.path.join(OUTPUT_DIR, "youtube_subtitle.srt")
            vtt_path = os.path.join(OUTPUT_DIR, "youtube_subtitle.vtt")
            sub_vtt = None
            if os.path.exists(srt_path):
                srt_to_vtt(srt_path, vtt_path)
                if os.path.exists(vtt_path):
                    sub_vtt = vtt_path
            elif os.path.exists(vtt_path):
                sub_vtt = vtt_path

            if sub_vtt:
                st.video(video_file, subtitles=sub_vtt)
            else:
                st.video(video_file)

        # 3. 构造地址框的内容与状态
        if video_exists:
            saved_url = load_youtube_url()
            if saved_url:
                url_value = saved_url
                input_disabled = True
                buttons_disabled = False
            else:
                url_value = t("本地上传视频，无法重新下载 (Local Uploaded Video)")
                input_disabled = True
                buttons_disabled = True
        else:
            saved_url = ""
            url_value = ""
            input_disabled = False
            buttons_disabled = False

        col1, col2 = st.columns([3, 1])
        with col1:
            url = st.text_input(t("Enter YouTube link:"), value=url_value, disabled=input_disabled, key="youtube_url_input")
        with col2:
            res_dict = {
                "360p": "360",
                "1080p": "1080",
                "Best": "best"
            }
            target_res = load_key("ytb_resolution")
            res_options = list(res_dict.keys())
            default_idx = list(res_dict.values()).index(target_res) if target_res in res_dict.values() else 0
            res_display = st.selectbox(t("Resolution"), options=res_options, index=default_idx, disabled=input_disabled, key="resolution_select")
            res = res_dict[res_display]

        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            download_video_clicked = st.button(t("Download Video"), key="download_button", use_container_width=True, disabled=buttons_disabled)
        with col_btn2:
            download_sub_clicked = st.button(t("Download Subtitle Only"), key="download_sub_button", use_container_width=True, disabled=buttons_disabled)

        # 4. 友好报错提示区域
        if "download_error" in st.session_state:
            error_str = st.session_state.download_error
            if "confirm you're not a bot" in error_str or "Sign in" in error_str:
                st.warning("⚠️ **YouTube 触发了人机验证 Bot 拦截！**")
                st.markdown(
                    """
                    <div style="
                        border-left: 5px solid #ff9900;
                        background-color: #fffaf0;
                        padding: 12px 18px;
                        border-radius: 6px;
                        color: #c97d00;
                        font-size: 14px;
                        margin-bottom: 15px;
                        line-height: 1.5;
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    ">
                        💡 <b>如何快速解决此问题 (How to bypass Bot Check)：</b><br/>
                        由于 YouTube 加强了防爬人机请求检测，需要向下载工具提供浏览器登录 Cookie 证明身份：<br/><br/>
                        1. <b>使用浏览器 Cookie（最简便）</b>：在左侧的配置 Tab 面板的「<b>配音与系统设置 -> YouTube 自动读取浏览器 Cookie</b>」中选择您常用的浏览器（如 <code>chrome</code> 或 <code>edge</code>），点击保存。<br/>
                        2. <b>使用导出的 Cookies 文件</b>：在浏览器中安装 Cookie 导出插件（如 <i>Get cookies.txt LOCALLY</i>），导出 <code>youtube.com</code> 的 cookie 文件，并将文件路径填在左侧「<b>YouTube Cookies 文件路径</b>」中。<br/>
                        3. <b>重新点击下载</b>：配置完成后，无需重新启动，直接再次点击下载即可！
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.error(t("Video/Subtitle Download Failed ❌"))
                st.code(error_str, wrap_lines=True)

        # 5. 执行下载响应
        active_url = saved_url if video_exists else url

        if download_video_clicked:
            if active_url:
                if "download_error" in st.session_state:
                    del st.session_state.download_error
                try:
                    if video_exists:
                        with st.spinner("Deleting old video files for replacement..."):
                            delete_existing_videos()
                    
                    with st.spinner("Downloading video..."):
                        download_video_ytdlp(active_url, resolution=res)
                    
                    save_youtube_url(active_url)
                    
                    from core.asr_backend.ytb_subtitle_asr import use_subtitle_file
                    with st.spinner("Downloading subtitles..."):
                        sub_path, sub_type = download_subtitle_ytdlp(active_url)
                    
                    if sub_path:
                        lang = load_key("whisper.language")
                        if lang == 'auto': lang = 'en'
                        with st.spinner("Applying subtitles..."):
                            success = use_subtitle_file(sub_path, lang)
                        if success:
                            st.session_state.download_status_toast = (
                                t("Successfully downloaded and applied official subtitles (type: {type}), WhisperX will be bypassed!").format(type=sub_type), 
                                "🎉"
                            )
                        else:
                            st.session_state.download_status_toast = (
                                t("Downloaded subtitles but failed to parse them."), 
                                "⚠️"
                            )
                    else:
                        st.session_state.download_status_toast = (
                            t("No English subtitles found (WhisperX will be used)."), 
                            "ℹ️"
                        )
                    st.rerun()
                except Exception as e:
                    st.session_state.download_error = str(e)
                    st.rerun()

        if download_sub_clicked:
            if active_url:
                if "download_error" in st.session_state:
                    del st.session_state.download_error
                try:
                    from core.asr_backend.ytb_subtitle_asr import use_subtitle_file
                    with st.spinner("Downloading subtitles..."):
                        sub_path, sub_type = download_subtitle_ytdlp(active_url)
                    
                    if sub_path:
                        save_youtube_url(active_url)
                        lang = load_key("whisper.language")
                        if lang == 'auto': lang = 'en'
                        with st.spinner("Applying subtitles..."):
                            success = use_subtitle_file(sub_path, lang)
                        if success:
                            st.session_state.download_status_toast = (
                                t("Successfully downloaded and applied official subtitles (type: {type}), WhisperX will be bypassed!").format(type=sub_type), 
                                "🎉"
                            )
                        else:
                            st.session_state.download_status_toast = (
                                t("Downloaded subtitles but failed to parse them."), 
                                "⚠️"
                            )
                    else:
                        st.session_state.download_status_toast = (
                            t("No English subtitles found (original or auto-generated) for this link."), 
                            "❌"
                        )
                    st.rerun()
                except Exception as e:
                    st.session_state.download_error = str(e)
                    st.rerun()

        # 5. 如果视频已存在，渲染删除并重置按钮
        if video_exists:
            if st.button(t("Delete and Reselect"), key="delete_video_button", use_container_width=True):
                delete_existing_videos()
                url_file = os.path.join(OUTPUT_DIR, "youtube_url.txt")
                if os.path.exists(url_file):
                    os.remove(url_file)
                if os.path.exists(OUTPUT_DIR):
                    try:
                        shutil.rmtree(OUTPUT_DIR)
                    except Exception:
                        pass
                sleep(1)
                st.rerun()
            return True

        # 6. 如果视频不存在，渲染上传组件
        else:
            uploaded_file = st.file_uploader(t("Or upload video"), type=load_key("allowed_video_formats") + load_key("allowed_audio_formats"))
            if uploaded_file:
                if os.path.exists(OUTPUT_DIR):
                    shutil.rmtree(OUTPUT_DIR)
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                
                raw_name = uploaded_file.name.replace(' ', '_')
                name, ext = os.path.splitext(raw_name)
                clean_name = re.sub(r'[^\w\-_\.]', '', name) + ext.lower()
                    
                with open(os.path.join(OUTPUT_DIR, clean_name), "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # 保存上传视频的文件名作为标题，并清理旧封面和 url
                with open(os.path.join(OUTPUT_DIR, 'video_title.txt'), 'w', encoding='utf-8') as f:
                    f.write(clean_name)
                cover_temp = os.path.join(OUTPUT_DIR, 'video_cover.jpg')
                if os.path.exists(cover_temp):
                    os.remove(cover_temp)
                url_file = os.path.join(OUTPUT_DIR, "youtube_url.txt")
                if os.path.exists(url_file):
                    os.remove(url_file)

                if ext.lower() in load_key("allowed_audio_formats"):
                    convert_audio_to_video(os.path.join(OUTPUT_DIR, clean_name))
                st.rerun()
            else:
                return False

def convert_audio_to_video(audio_file: str) -> str:
    output_video = os.path.join(OUTPUT_DIR, 'black_screen.mp4')
    if not os.path.exists(output_video):
        print(f"Converting audio to video with FFmpeg ...")
        encoder = get_ffmpeg_video_encoder() or 'libx264'
        ffmpeg_cmd = ['ffmpeg', '-y', '-f', 'lavfi', '-i', 'color=c=black:s=640x360', '-i', audio_file, '-shortest', '-c:v', encoder, '-c:a', 'aac', '-pix_fmt', 'yuv420p', output_video]
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True, encoding='utf-8')
        print(f"Converted <{audio_file}> to <{output_video}> using encoder '{encoder}'\n")
        os.remove(audio_file)
    return output_video
