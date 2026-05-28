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

def download_video_section():
    st.header(t("a. Download or Upload Video"))
    with st.container(border=True):
        try:
            video_file = find_video_files()
            # For original videos, path-based loading is better as they can be large
            st.video(video_file)
            if st.button(t("Delete and Reselect"), key="delete_video_button"):
                os.remove(video_file)
                if os.path.exists(OUTPUT_DIR):
                    shutil.rmtree(OUTPUT_DIR)
                sleep(1)
                st.rerun()
            return True
        except:
            col1, col2 = st.columns([3, 1])
            with col1:
                url = st.text_input(t("Enter YouTube link:"))
            with col2:
                res_dict = {
                    "360p": "360",
                    "1080p": "1080",
                    "Best": "best"
                }
                target_res = load_key("ytb_resolution")
                res_options = list(res_dict.keys())
                default_idx = list(res_dict.values()).index(target_res) if target_res in res_dict.values() else 0
                res_display = st.selectbox(t("Resolution"), options=res_options, index=default_idx)
                res = res_dict[res_display]
            col_btn1, col_btn2 = st.columns([1, 1])
            with col_btn1:
                download_video_clicked = st.button(t("Download Video"), key="download_button", use_container_width=True)
            with col_btn2:
                download_sub_clicked = st.button(t("Download Subtitle Only"), key="download_sub_button", use_container_width=True)

            if download_video_clicked:
                if url:
                    with st.spinner("Downloading video..."):
                        download_video_ytdlp(url, resolution=res)
                    
                    from core.asr_backend.ytb_subtitle_asr import use_subtitle_file
                    with st.spinner("Downloading subtitles..."):
                        sub_path, sub_type = download_subtitle_ytdlp(url)
                    
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
                if url:
                    from core.asr_backend.ytb_subtitle_asr import use_subtitle_file
                    with st.spinner("Downloading subtitles..."):
                        sub_path, sub_type = download_subtitle_ytdlp(url)
                    
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
                            t("No English subtitles found (original or auto-generated) for this link."), 
                            "❌"
                        )
                    st.rerun()

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

                # 保存上传视频的文件名作为标题，并清理旧封面
                with open(os.path.join(OUTPUT_DIR, 'video_title.txt'), 'w', encoding='utf-8') as f:
                    f.write(clean_name)
                cover_temp = os.path.join(OUTPUT_DIR, 'video_cover.jpg')
                if os.path.exists(cover_temp):
                    os.remove(cover_temp)

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
