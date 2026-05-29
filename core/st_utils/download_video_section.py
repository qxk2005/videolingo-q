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

        # 4. 执行下载响应
        active_url = saved_url if video_exists else url

        if download_video_clicked:
            if active_url:
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

        if download_sub_clicked:
            if active_url:
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
