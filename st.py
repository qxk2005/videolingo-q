import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os, sys
# SET PATH
current_dir = os.path.dirname(os.path.abspath(__file__))
for path in ["/opt/homebrew/bin", "/usr/local/bin", current_dir]:
    if path not in os.environ.get('PATH', ''):
        os.environ['PATH'] = path + os.pathsep + os.environ.get('PATH', '')
sys.path.append(current_dir)

import importlib
import streamlit as st
import shutil, subprocess
import pandas as pd
from core.st_utils.imports_and_utils import *
import core.st_utils.sidebar_setting as _sidebar_setting_mod
importlib.reload(_sidebar_setting_mod)
from core.st_utils.sidebar_setting import page_setting
from core.utils.models import *
from core import *
for _mod_name in (
    "core.utils.audio_gender",
    "core.tts_backend.tts_main",
    "core._8_1_audio_task",
    "core._8_2_dub_chunks",
    "core._10_gen_audio",
):
    if _mod_name in sys.modules:
        try:
            importlib.reload(sys.modules[_mod_name])
        except Exception:
            pass
from core.asr_backend.audio_preprocess import reapply_vocab_correction_to_cleaned_chunks, detect_suspicious_terms_from_srt, convert_video_to_audio
from core.asr_backend.ytb_subtitle_asr import use_subtitle_file
from core import _1_ytdlp

st.set_page_config(
    page_title="VideoLingo Q", 
    page_icon="docs/logo.svg", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/qxk2005/videolingo-q',
        'Report a bug': "https://github.com/qxk2005/videolingo-q/issues",
        'About': "# VideoLingo Q\n专业视频本地化工作站\n\n版本：1.0.0"
    }
)

SUB_VIDEO = "output/output_sub.mp4"
DUB_VIDEO = "output/output_dub.mp4"
SRC_SRT = "output/src.srt"

_SRT_FILES = [
    "output/src.srt", "output/trans.srt",
    "output/src_trans.srt", "output/trans_src.srt", "output/dub.srt",
]

# Files produced by the text/subtitle pipeline (steps 3–7).
# Deleting these forces a full re-run of translation + subtitle generation
# while keeping the expensive ASR output (step 2) intact.
_SUBTITLE_CACHE_FILES = [
    "output/log/split_by_nlp.txt",
    "output/log/split_by_meaning.txt",
    "output/log/terminology.json",
    "output/log/translation_results.xlsx",
    "output/log/translation_results_for_subtitles.xlsx",
    "output/log/translation_results_remerged.xlsx",
    SUB_VIDEO,
] + _SRT_FILES
_SUBTITLE_CACHE_DIRS = ["output/gpt_log"]

def reset_subtitle_cache():
    """Delete all cached subtitle/translation files so the pipeline runs fresh."""
    print("🗑️ Clearing subtitle cache...")
    for f in _SUBTITLE_CACHE_FILES:
        if os.path.exists(f):
            os.remove(f)
            print(f"  Deleted: {f}")
        else:
            print(f"  Not found (skip): {f}")
    for d in _SUBTITLE_CACHE_DIRS:
        if os.path.exists(d):
            shutil.rmtree(d)
            print(f"  Deleted dir: {d}")
        else:
            print(f"  Not found (skip): {d}")
    print("✅ Subtitle cache cleared.")

def reapply_and_rebuild():
    """Re-apply domain vocab correction to cleaned_chunks.xlsx, then clear all downstream caches."""
    reapply_vocab_correction_to_cleaned_chunks()
    for f in _SUBTITLE_CACHE_FILES:
        if os.path.exists(f):
            os.remove(f)
    for d in _SUBTITLE_CACHE_DIRS:
        if os.path.exists(d):
            shutil.rmtree(d)
    st.toast(t("Vocab correction applied. Click 'Start Processing Subtitles' to rebuild."), icon="✅")


def render_friendly_error(error_str, stage_type="text"):
    import re
    # 尝试匹配 Google OAuth 的授权链接
    auth_url_match = re.search(r'(https://accounts\.google\.com/o/oauth2/auth[^\s\'"<>]+)', error_str)
    
    if auth_url_match:
        auth_url = auth_url_match.group(1)
        st.warning("🔑 **Antigravity CLI (agy) 登录会话过期或尚未授权！**")
        
        # 渲染高档卡片式 Google OAuth 说明
        st.markdown(
            f"""
            <div style="
                border-left: 5px solid #ffa500;
                background-color: #fff9e6;
                padding: 15px 20px;
                border-radius: 8px;
                color: #b77c00;
                font-size: 15px;
                margin-bottom: 20px;
                line-height: 1.6;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            ">
                <b>💡 极速登录并恢复任务指南 (Quick Auth Guide)：</b><br/>
                1. <b>第一步：获取授权码</b>。点击下方的金色登录按钮前往 Google 完成账号授权，并复制返回的 Authorization Code。<br/>
                2. <b>第二步：保存配置</b>。将 Code 粘贴到左侧 LLM 配置的「<b>Antigravity Token Code</b>」输入框中。后续系统将自动使用它保持登录，免去重复输入！<br/>
                3. <b>第三步：恢复运行</b>。点击最下方的<b>「清除错误并重试」</b>，后台会自动进行静默激活登录，任务即可瞬间继续执行！
            </div>
            """,
            unsafe_allow_html=True
        )
        # 用 Streamlit 原生的 Link Button 极其高端漂亮地展示，100% 绝对安全
        st.link_button("🔗 点击在此浏览器页面中登录 Google 并获取 Code", auth_url, use_container_width=True)
        st.divider()
    else:
        st.error(t("Task Processing Failed ❌"))
        
        # 用 st.code 安全、规范、不破坏 HTML 地输出具体的错误回溯详情
        st.markdown("**错误详情 (Error Detail)：**")
        st.code(error_str, wrap_lines=True)
        
        # 提供美观大方的排查建议
        if stage_type == "text":
            st.markdown(
                """
                <div style="
                    border-left: 5px solid #ff4b4b;
                    background-color: #ffeef0;
                    padding: 12px 18px;
                    border-radius: 6px;
                    color: #d11a2a;
                    font-size: 14px;
                    margin-bottom: 15px;
                    font-weight: 500;
                    line-height: 1.5;
                ">
                    💡 <b>建议排查步骤 (Troubleshooting Tips)：</b><br/>
                    1. <b>API Key 检查</b>：请检查左侧 LLM 配置里的 API Key 是否正确（以及是否有额度、是否欠费）。<br/>
                    2. <b>网络与代理</b>：如果您使用的是国外模型 API，请确保您的本地代理或 Base URL 可以稳定连通。<br/>
                    3. <b>如果是火山引擎/豆包</b>：请确保 AppID 和 Access Token 配置完整有效，且未发生接口限制报错。<br/>
                    4. <b>本地命令行 (Antigravity CLI)</b>：如果您开启了使用 CLI 选项，请确保 `agy` 在本地命令行中可以正常调用。
                </div>
                """,
                unsafe_allow_html=True
            )
        else: # dubbing
            st.markdown(
                """
                <div style="
                    border-left: 5px solid #ff4b4b;
                    background-color: #ffeef0;
                    padding: 12px 18px;
                    border-radius: 6px;
                    color: #d11a2a;
                    font-size: 14px;
                    margin-bottom: 15px;
                    font-weight: 500;
                    line-height: 1.5;
                ">
                    💡 <b>建议排查步骤 (Troubleshooting Tips)：</b><br/>
                    1. <b>字幕文件缺失</b>：如果报错中提到 <code>FileNotFoundError: Subtitle files missing</code>，这是因为您可能没有成功执行完 <b>“b. 翻译并生成字幕”</b> 步骤。请确保第二阶段的步骤全部打勾完成后再进行配音。<br/>
                    2. <b>TTS 接口故障</b>：如果您选择的是 <b>Edge TTS</b> 或 <b>豆包 TTS</b>，请确保对应的 AppID、Access Token 或本地网络连通，且没有在配音生成时发生连接或鉴权报错。
                </div>
                """,
                unsafe_allow_html=True
            )

    # 核心：无论何种错误，最下方绝对能安全渲染出来原生的“清除错误并重试”按钮！
    btn_key = "clear_error_and_retry_btn" if stage_type == "text" else "clear_dubbing_error_btn"
    err_state_key = "processing_error" if stage_type == "text" else "dubbing_error"
    if st.button(t("Clear Error & Retry"), key=btn_key, use_container_width=True):
        del st.session_state[err_state_key]
        st.rerun()
    st.divider()


def _show_suspicious_terms_section():
    """Display detected unusual/high-freq terms from src.srt for user review."""
    terms = detect_suspicious_terms_from_srt(SRC_SRT)
    with st.expander(t("🔍 Review Suspicious Terms (optional)"), expanded=True):
        st.markdown(t("The following unusual or high-frequency terms were found in the subtitles. Check for misrecognitions, add corrections to **Domain Vocabulary** in the sidebar, then click **Regenerate Subtitles** to re-process before burning."))
        if not terms:
            st.info(t("No suspicious terms detected."))
        else:
            df_terms = pd.DataFrame(terms)[['term', 'category', 'count', 'example']]
            st.dataframe(df_terms, use_container_width=True, hide_index=True)


def _render_steps(steps, current_step_key, completion_check_func):
    """Render a list of steps with dynamic coloring and checkmarks."""
    html_content = ""
    for i, step_name in enumerate(steps, 1):
        status = "pending"
        if completion_check_func(i):
            status = "completed"
        if st.session_state.get(current_step_key) == i:
            status = "running"
        
        color = "#808080" # pending: gray
        icon = ""
        if status == "completed":
            color = "#28a745" # completed: green
            icon = " ✅"
        elif status == "running":
            color = "#ffc107" # running: yellow
            icon = " ⏳"
        
        html_content += f"<p style='font-size: 20px; color: {color}; margin: 5px 0;'>{i}. {step_name}{icon}</p>"
    
    st.markdown(html_content, unsafe_allow_html=True)

def text_processing_section():
    st.header(t("b. Translate and Generate Subtitles"))
    with st.container(border=True):
        # ── 友好报错提示区域 ──
        if "processing_error" in st.session_state:
            render_friendly_error(st.session_state.processing_error, "text")

        st.markdown(f"<p style='font-size: 20px;'>{t('This stage includes the following steps:')}</p>", unsafe_allow_html=True)
        
        # ── Progress Bar Placeholder for Text Stage ──
        from core.utils.progress_utils import set_st_progress_placeholder
        text_progress_container = st.empty()
        set_st_progress_placeholder(text_progress_container)

        steps = [
            t("WhisperX word-level transcription"),
            t("Sentence segmentation using NLP and LLM"),
            t("Summarization and multi-step translation"),
            t("Cutting and aligning long subtitles"),
            t("Generating timeline and subtitles"),
            t("Merging subtitles into the video")
        ]
        
        def check_text_step_completed(step_idx):
            if step_idx == 1: return os.path.exists(_2_CLEANED_CHUNKS)
            if step_idx == 2: return os.path.exists(_3_2_SPLIT_BY_MEANING)
            if step_idx == 3: return os.path.exists(_4_2_TRANSLATION)
            if step_idx == 4: return os.path.exists(_5_SPLIT_SUB)
            if step_idx == 5: return os.path.exists(SRC_SRT)
            if step_idx == 6: return os.path.exists(SUB_VIDEO)
            return False

        _render_steps(steps, "text_processing_step", check_text_step_completed)

        # Handle Sequential Execution
        if st.session_state.get("text_processing_step"):
            step = st.session_state.text_processing_step
            try:
                if step == 1:
                    with st.spinner(t("Using Whisper for transcription...")): _2_asr.transcribe()
                    with st.spinner(t("Running speaker diarization...")): _2_5_diarize.diarize_main()
                    st.session_state.text_processing_step = 2
                    st.rerun()
                elif step == 2:
                    with st.spinner(t("Splitting long sentences...")):
                        _3_1_split_nlp.split_by_spacy()
                        _3_2_split_meaning.split_sentences_by_meaning()
                    st.session_state.text_processing_step = 3
                    st.rerun()
                elif step == 3:
                    with st.spinner(t("Summarizing and translating...")):
                        _4_1_summarize.get_summary()
                        _4_2_translate.translate_all()
                    st.session_state.text_processing_step = 4
                    st.rerun()
                elif step == 4:
                    with st.spinner(t("Processing and aligning subtitles...")):
                        _5_split_sub.split_for_sub_main()
                    st.session_state.text_processing_step = 5
                    st.rerun()
                elif step == 5:
                    with st.spinner(t("Generating timeline and subtitles...")):
                        _6_gen_sub.align_timestamp_main()
                    st.session_state.text_processing_step = 6
                    st.rerun()
                elif step == 6:
                    with st.spinner(t("Merging subtitles to video...")):
                        _7_sub_into_vid.merge_subtitles_to_video()
                    st.session_state.text_processing_step = None
                    st.success(t("Subtitle processing complete! 🎉"))
                    st.balloons()
                    st.rerun()
            except Exception as e:
                # Catch exception, save to state, and gracefully halt progress
                st.session_state.processing_error = str(e)
                st.session_state.text_processing_step = None
                st.rerun()

        if not os.path.exists(SRC_SRT):
            # If error is present, disable the button until cleared to maintain safety
            btn_disabled = "processing_error" in st.session_state
            if st.button(t("Start Processing Subtitles"), key="text_processing_button", disabled=btn_disabled):
                # 动态探测断点：从第一个未完成的步骤开始执行
                start_step = 1
                for step_idx in range(1, 7):
                    if not check_text_step_completed(step_idx):
                        start_step = step_idx
                        break
                st.session_state.text_processing_step = start_step
                st.rerun()

            # Subtitle upload bypass
            with st.expander(t("📄 Use Official Subtitles (skip WhisperX)"), expanded=False):
                st.markdown(t("Upload an SRT or VTT subtitle file (e.g. downloaded from YouTube). The pipeline will use it directly instead of running WhisperX, saving significant time."))
                uploaded_sub = st.file_uploader(t("Upload subtitle file (.srt / .vtt)"), type=['srt', 'vtt'], key="subtitle_file_upload")
                if uploaded_sub:
                    lang = load_key("whisper.language")
                    if lang == 'auto': lang = 'en'
                    if st.button(t("Use This Subtitle File"), key="use_subtitle_button"):
                        sub_save_path = f"output/uploaded_subtitle.{uploaded_sub.name.rsplit('.', 1)[-1].lower()}"
                        os.makedirs("output", exist_ok=True)
                        with open(sub_save_path, 'wb') as f: f.write(uploaded_sub.getbuffer())
                        with st.spinner(t("Parsing subtitle file...")): success = use_subtitle_file(sub_save_path, lang)
                        if success:
                            st.toast(t("Subtitle parsed! Click 'Start Processing Subtitles' to continue."), icon="✅")
                            st.rerun()
                        else: st.error(t("Failed to parse subtitle file. Please check the format."))

            if st.button(t("Regenerate Subtitles"), key="reset_subtitle_button"):
                reset_subtitle_cache()
                st.rerun()

        elif not os.path.exists(SUB_VIDEO):
            if load_key("auto_burn_subtitles"):
                st.session_state.text_processing_step = 6
                st.rerun()
            else:
                _show_suspicious_terms_section()
                if st.button(t("Burn Subtitles to Video"), key="burn_to_video_button"):
                    st.session_state.text_processing_step = 6
                    st.rerun()
                if st.button(t("Regenerate Subtitles"), key="reset_subtitle_button_pre_burn"):
                    reset_subtitle_cache()
                    st.rerun()
        else:
            if load_key("burn_subtitles"): 
                safe_video_display(SUB_VIDEO)
            download_subtitle_zip_button(text=t("Download All Srt Files"))
            if st.button(t("Re-apply Vocab & Rebuild"), key="reapply_vocab_button"):
                reapply_and_rebuild()
                st.rerun()
            if st.button(t("Archive to 'history'"), key="cleanup_in_text_processing"):
                cleanup()
                st.rerun()
            if st.button(t("Regenerate Subtitles"), key="reset_subtitle_button_done"):
                reset_subtitle_cache()
                st.rerun()

def _render_speaker_review_section():
    """
    Render the interactive Speaker & Voice Review / Override panel in the Dubbing section.
    Features:
      1. Role Cards: Per-speaker original audio clip preview (st.audio), line count, sample line,
         one-click gender toggle (Male/Female), and optional per-speaker TTS voice override.
      2. Expandable Sentence-level Table (st.data_editor): Fine-grained per-line speaker/gender/voice override.
      3. Smart Incremental Rebuild: Only invalidates and regenerates modified sentences' TTS cache.
    """
    if not os.path.exists(_8_1_AUDIO_TASK):
        return

    try:
        df_tasks = pd.read_excel(_8_1_AUDIO_TASK)
    except Exception as e:
        st.warning(f"无法读取配音任务表 ({e})")
        return

    if df_tasks.empty:
        return

    # Ensure required columns exist
    if 'gender' not in df_tasks.columns:
        df_tasks['gender'] = 'female'
    if 'speaker_id' not in df_tasks.columns:
        df_tasks['speaker_id'] = ['SPEAKER_00' if g == 'male' else 'SPEAKER_01' for g in df_tasks['gender']]
    if 'voice' not in df_tasks.columns:
        df_tasks['voice'] = ''
    df_tasks['voice'] = df_tasks['voice'].fillna('').astype(str)
    df_tasks['speaker_id'] = df_tasks['speaker_id'].fillna('SPEAKER_00').astype(str)
    df_tasks['gender'] = df_tasks['gender'].fillna('female').astype(str)

    from core.utils.audio_gender import extract_speaker_sample_clips
    from core.st_utils.sidebar_setting import get_available_tts_voices
    from core._10_gen_audio import invalidate_audio_cache_for_numbers
    from core._8_2_dub_chunks import process_cutoffs

    sample_clips = extract_speaker_sample_clips(df_tasks)
    voice_catalog = get_available_tts_voices()  # {display_label: voice_id}
    voice_id_to_label = {v: k for k, v in voice_catalog.items()}
    voice_labels = list(voice_catalog.keys())

    is_paused_for_review = bool(st.session_state.get("awaiting_speaker_review", False))

    with st.expander(f"🎙️ {t('Speaker & Voice Review / Override')}", expanded=is_paused_for_review or not os.path.exists(DUB_VIDEO)):
        if is_paused_for_review:
            st.info(f"⏸️ {t('Paused after speaker identification. Please review or adjust speaker genders/voices below, then click Confirm & Continue Dubbing.')}")
        else:
            st.caption(t("Preview each detected speaker's original voice, switch gender or assign dedicated voices per speaker, and incrementally re-synthesize only modified lines."))

        unique_speakers = sorted(df_tasks['speaker_id'].unique())
        speaker_updates = {}

        tab_cards, tab_table = st.tabs([
            "🎭 按说话人角色卡片调整",
            f"📝 {t('Sentence-Level Fine-Tuning Table (Optional)')}"
        ])

        with tab_cards:
            # Render Speaker Role Cards (up to 2 columns per row; no inner st.columns to respect Streamlit nesting limit)
            cols_per_row = 2 if len(unique_speakers) > 1 else 1
            for i in range(0, len(unique_speakers), cols_per_row):
                row_spks = unique_speakers[i:i + cols_per_row]
                cols = st.columns(cols_per_row)
                for col_idx, spk in enumerate(row_spks):
                    spk_rows = df_tasks[df_tasks['speaker_id'] == spk]
                    line_count = len(spk_rows)
                    # Majority gender & voice for this speaker
                    maj_gender = spk_rows['gender'].mode().iloc[0] if not spk_rows['gender'].mode().empty else 'female'
                    if maj_gender not in ('male', 'female'):
                        maj_gender = 'female'
                    non_empty_voices = [v for v in spk_rows['voice'].tolist() if v]
                    cur_spk_voice = non_empty_voices[0] if non_empty_voices else ''
                    cur_voice_label = voice_id_to_label.get(cur_spk_voice, voice_labels[0])

                    sample_text = str(spk_rows.iloc[0].get('text', '') or spk_rows.iloc[0].get('origin', ''))[:48]
                    with cols[col_idx]:
                        with st.container(border=True):
                            badge_icon = "👨" if maj_gender == "male" else "👩"
                            st.markdown(f"**{badge_icon} `{spk}`** · 共 **{line_count}** 句")
                            st.caption(f"💬 示例：*{sample_text}*")

                            if spk in sample_clips and os.path.exists(sample_clips[spk]):
                                st.audio(sample_clips[spk], format="audio/wav")

                            gender_opts = ["male", "female"]
                            gender_fmt = {"male": f"👨 {t('Male')}", "female": f"👩 {t('Female')}"}
                            new_gen = st.selectbox(
                                t("Gender"),
                                options=gender_opts,
                                index=gender_opts.index(maj_gender),
                                format_func=lambda x: gender_fmt.get(x, x),
                                key=f"spk_card_gender_{spk}"
                            )
                            new_v_label = st.selectbox(
                                t("Dedicated Voice"),
                                options=voice_labels,
                                index=voice_labels.index(cur_voice_label) if cur_voice_label in voice_labels else 0,
                                key=f"spk_card_voice_{spk}"
                            )
                            new_v_id = voice_catalog.get(new_v_label, "")

                            speaker_updates[spk] = {
                                "old_gender": maj_gender,
                                "new_gender": new_gen,
                                "old_voice": cur_spk_voice,
                                "new_voice": new_v_id,
                            }

        with tab_table:
            edit_cols = ['number', 'start_time', 'end_time', 'speaker_id', 'gender', 'voice', 'text']
            avail_cols = [c for c in edit_cols if c in df_tasks.columns]
            edited_df = st.data_editor(
                df_tasks[avail_cols],
                column_config={
                    "number": st.column_config.NumberColumn("序号", disabled=True, width="small"),
                    "start_time": st.column_config.TextColumn("开始时间", disabled=True, width="small"),
                    "end_time": st.column_config.TextColumn("结束时间", disabled=True, width="small"),
                    "speaker_id": st.column_config.TextColumn(t("Speaker"), width="small"),
                    "gender": st.column_config.SelectboxColumn(t("Gender"), options=["male", "female"], required=True, width="small"),
                    "voice": st.column_config.SelectboxColumn(t("Dedicated Voice"), options=[""] + [v for v in voice_catalog.values() if v], width="medium"),
                    "text": st.column_config.TextColumn("配音台词", width="large"),
                },
                hide_index=True,
                use_container_width=True,
                key="sentence_level_dub_editor"
            )

        def _apply_speaker_and_table_edits() -> list:
            """Apply speaker-card and table edits to df_tasks, persist to Excel, and return modified task numbers."""
            orig_df = df_tasks.copy()
            updated_df = df_tasks.copy()

            # 1. Apply per-row edits from st.data_editor first
            for col in ['speaker_id', 'gender', 'voice', 'text']:
                if col in edited_df.columns:
                    updated_df[col] = edited_df[col].fillna('').astype(str)

            # 2. Apply speaker-card level bulk updates (if user changed a speaker card's gender or voice)
            for spk, upd in speaker_updates.items():
                mask = updated_df['speaker_id'] == spk
                if upd['new_gender'] != upd['old_gender']:
                    updated_df.loc[mask, 'gender'] = upd['new_gender']
                if upd['new_voice'] != upd['old_voice']:
                    updated_df.loc[mask, 'voice'] = upd['new_voice']

            # 3. Identify which task numbers actually changed
            changed_nums = []
            for idx_row in range(len(updated_df)):
                num = int(updated_df.iloc[idx_row]['number'])
                for col in ['speaker_id', 'gender', 'voice', 'text']:
                    if str(orig_df.iloc[idx_row].get(col, '') or '') != str(updated_df.iloc[idx_row].get(col, '') or ''):
                        changed_nums.append(num)
                        break

            if changed_nums:
                # Recompute chunk cutoffs so speaker/gender/voice transitions always have cut_off = 1
                try:
                    updated_df = process_cutoffs(updated_df)
                except Exception:
                    pass
                updated_df.to_excel(_8_1_AUDIO_TASK, index=False)
                invalidate_audio_cache_for_numbers(changed_nums)

            return changed_nums

        col_act1, col_act2 = st.columns(2)
        with col_act1:
            if st.button(f"💾 {t('Save Adjustments & Incremental Rebuild')}", key="save_speaker_adjustments_btn", use_container_width=True):
                changed_nums = _apply_speaker_and_table_edits()
                st.session_state.awaiting_speaker_review = False
                if changed_nums:
                    st.toast(f"✅ 已更新 {len(changed_nums)} 句配音配置并清理对应缓存，正在增量重新合成...", icon="🎙️")
                    st.session_state.audio_processing_step = 2
                else:
                    st.toast("ℹ️ 未检测到配置变更", icon="ℹ️")
                st.rerun()

        with col_act2:
            if is_paused_for_review or not os.path.exists(DUB_VIDEO):
                if st.button(f"▶️ {t('Confirm Speakers & Continue Dubbing')}", key="confirm_speakers_continue_btn", type="primary", use_container_width=True):
                    _apply_speaker_and_table_edits()
                    st.session_state.awaiting_speaker_review = False
                    st.session_state.audio_processing_step = 2
                    st.rerun()


def audio_processing_section():
    st.header(t("c. Dubbing"))
    with st.container(border=True):
        # ── 友好报错提示区域 ──
        if "dubbing_error" in st.session_state:
            render_friendly_error(st.session_state.dubbing_error, "dubbing")

        st.markdown(f"<p style='font-size: 20px;'>{t('This stage includes the following steps:')}</p>", unsafe_allow_html=True)
        
        # ── Progress Bar Placeholder for Audio Stage ──
        from core.utils.progress_utils import set_st_progress_placeholder
        audio_progress_container = st.empty()
        set_st_progress_placeholder(audio_progress_container)

        steps = [
            t("Generate audio tasks and chunks"),
            t("Extract reference audio"),
            t("Generate and merge audio files"),
            t("Merge final audio into video")
        ]
        
        def check_audio_step_completed(step_idx):
            if step_idx == 1: return os.path.exists(_8_1_AUDIO_TASK)
            if step_idx == 2:
                if load_key("skip_refer") and (os.path.exists("output/dub.mp3") or os.path.exists(DUB_VIDEO)):
                    return True
                return os.path.exists(_AUDIO_REFERS_DIR) and any(os.listdir(_AUDIO_REFERS_DIR))
            if step_idx == 3: return os.path.exists("output/dub.mp3") or os.path.exists("output/dub.wav")
            if step_idx == 4: return os.path.exists(DUB_VIDEO)
            return False

        _render_steps(steps, "audio_processing_step", check_audio_step_completed)

        if st.session_state.get("audio_processing_step"):
            step = st.session_state.audio_processing_step
            try:
                if step == 1:
                    with st.spinner(t("Generate audio tasks")):
                        video_file = _1_ytdlp.find_video_files()
                        convert_video_to_audio(video_file)
                        _8_1_audio_task.gen_audio_task_main()
                        _8_2_dub_chunks.gen_dub_chunks()
                    try:
                        pause_dub = bool(load_key("pause_before_dub"))
                    except Exception:
                        pause_dub = False
                    if pause_dub:
                        st.session_state.audio_processing_step = None
                        st.session_state.awaiting_speaker_review = True
                    else:
                        st.session_state.audio_processing_step = 2
                    st.rerun()
                elif step == 2:
                    if not load_key("skip_refer"):
                        with st.spinner(t("Extract refer audio")): _9_refer_audio.extract_refer_audio_main()
                    st.session_state.audio_processing_step = 3
                    st.rerun()
                elif step == 3:
                    with st.spinner(t("Generate and merge audio files")):
                        _10_gen_audio.gen_audio()
                        _11_merge_audio.merge_full_audio()
                    st.session_state.audio_processing_step = 4
                    st.rerun()
                elif step == 4:
                    with st.spinner(t("Merge dubbing to the video")): _12_dub_to_vid.merge_video_audio()
                    st.session_state.audio_processing_step = None
                    st.session_state.awaiting_speaker_review = False
                    st.success(t("Audio processing complete! 🎇"))
                    st.balloons()
                    st.rerun()
            except Exception as e:
                # Catch exception, save to state, and gracefully halt progress
                st.session_state.dubbing_error = str(e)
                st.session_state.audio_processing_step = None
                st.rerun()

        # ── 常驻展示「🎙️ 说话人识别与配音微调」面板（只要已完成第 1 步任务生成） ──
        _render_speaker_review_section()

        video_file_found = False
        try:
            _1_ytdlp.find_video_files()
            video_file_found = True
        except: video_file_found = False

        if not os.path.exists(DUB_VIDEO):
            if not video_file_found: st.warning(t("Please download or upload a video first in section 'a.' before starting the audio processing."))
            if st.button(t("Start Audio Processing"), key="audio_processing_button", disabled=not video_file_found):
                st.session_state.awaiting_speaker_review = False
                st.session_state.audio_processing_step = 1
                st.rerun()
            if os.path.exists(_8_1_AUDIO_TASK):
                if st.button(t("Delete dubbing files"), key="delete_dubbing_files_retry"):
                    st.session_state.awaiting_speaker_review = False
                    delete_dubbing_files()
                    st.rerun()
        else:
            st.success(t("Audio processing is complete! You can check the audio files in the `output` folder."))
            if load_key("burn_subtitles"): 
                safe_video_display(DUB_VIDEO)
            if st.button(t("Delete dubbing files"), key="delete_dubbing_files"):
                st.session_state.awaiting_speaker_review = False
                delete_dubbing_files()
                st.rerun()
            if st.button(t("Archive to 'history'"), key="cleanup_in_audio_processing"):
                st.session_state.awaiting_speaker_review = False
                cleanup()
                st.rerun()

def process_audio():
    video_file = _1_ytdlp.find_video_files()
    convert_video_to_audio(video_file)
    with st.spinner(t("Generate audio tasks")): 
        _8_1_audio_task.gen_audio_task_main()
        _8_2_dub_chunks.gen_dub_chunks()
    with st.spinner(t("Extract refer audio")):
        _9_refer_audio.extract_refer_audio_main()
    with st.spinner(t("Generate all audio")):
        _10_gen_audio.gen_audio()
    with st.spinner(t("Merge full audio")):
        _11_merge_audio.merge_full_audio()
    with st.spinner(t("Merge dubbing to the video")):
        _12_dub_to_vid.merge_video_audio()
    
    st.success(t("Audio processing complete! 🎇"))
    st.balloons()

def get_video_info(file_path):
    """Get video encoding info using ffprobe with robust error handling."""
    if not file_path or not os.path.exists(file_path):
        return None
    try:
        import json
        # Ensure we use absolute path to avoid confusion
        abs_path = os.path.abspath(file_path)
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_streams', '-show_format', abs_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        if result.returncode != 0:
            return None
            
        data = json.loads(result.stdout)
        if 'streams' not in data:
            return None
            
        v_stream = next((s for s in data['streams'] if s.get('codec_type') == 'video'), {})
        a_stream = next((s for s in data['streams'] if s.get('codec_type') == 'audio'), {})
        
        info = {
            "File": os.path.basename(file_path),
            "Video Codec": v_stream.get('codec_name', 'N/A'),
            "Audio Codec": a_stream.get('codec_name', 'N/A'),
            "Resolution": f"{v_stream.get('width', 'N/A')}x{v_stream.get('height', 'N/A')}",
            "Pixel Format": v_stream.get('pix_fmt', 'N/A'),
            "Size": f"{os.path.getsize(file_path) / (1024*1024):.2f} MB"
        }
        return info
    except Exception as e:
        # Silently fail but log to console for debugging
        print(f"ffprobe error for {file_path}: {str(e)}")
        return None

def video_comparison_section():
    """Display comparison of video encoding info."""
    st.markdown(f"### 🔍 {t('Encoding Comparison')}")
    
    # Identify files - use absolute or properly relative paths
    orig_video = None
    try:
        orig_video = _1_ytdlp.find_video_files()
    except:
        pass
        
    videos = {
        "Original": orig_video,
        "Subtitled": SUB_VIDEO,
        "Dubbed": DUB_VIDEO
    }
    
    comparison_data = []
    for label, path in videos.items():
        if path and os.path.exists(path):
            info = get_video_info(path)
            if info:
                info["Stage"] = label
                comparison_data.append(info)
    
    if comparison_data:
        df_comp = pd.DataFrame(comparison_data)
        cols = ["Stage", "Video Codec", "Audio Codec", "Pixel Format", "Resolution", "Size"]
        st.dataframe(df_comp[cols], hide_index=True, use_container_width=True)
    else:
        # Debug info for the user if nothing is found
        if not any([os.path.exists(v) for v in videos.values() if v]):
            st.info("No video files found in output directory yet.")
        else:
            st.warning("Video files detected but failed to extract encoding info.")

def safe_video_display(file_path):
    """Safely display large videos using Streamlit's optimized media server."""
    if not os.path.exists(file_path):
        return
    
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    
    # ── Method 1: Path-based streaming (Best for 100MB+) ──
    # We use a trick: Streamlit handles absolute paths by serving them through its own media server
    abs_path = os.path.abspath(file_path)
    
    try:
        if file_size_mb > 150:
            st.video(abs_path)
            st.info(f"💡 {t('Large video detected')} ({file_size_mb:.1f} MB). {t('Using stream mode for better performance.')}")
        else:
            # For smaller files, reading as bytes is more responsive
            with open(file_path, 'rb') as f:
                st.video(f.read())
    except Exception as e:
        st.error(f"播放器加载失败: {str(e)}")
        st.warning("请尝试重新生成视频（当前文件体积过大，可能超出了浏览器或系统的单文件处理限制）。")

def file_browser():
    """Interactive Windows-style file browser with encoding comparison."""
    # ── 1. 视频封面和标题展示区域 (放置于最上方) ──
    st.markdown(f"### 🖼️ {t('Video Cover & Title')}")
    
    cover_path = "output/video_cover.jpg"
    title_path = "output/video_title.txt"
    
    video_title = t("暂未获取到视频标题")
    if os.path.exists(title_path):
        try:
            with open(title_path, 'r', encoding='utf-8') as f:
                video_title = f.read().strip()
        except Exception:
            pass

    # 显示封面
    if os.path.exists(cover_path):
        st.image(cover_path, use_column_width=True)
    else:
        # 显示一个带空白虚线边框的高颜值占位符
        st.markdown(
            f"""
            <div style="
                border: 2px dashed #999;
                border-radius: 12px;
                height: 180px;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                color: #999;
                font-size: 15px;
                background: linear-gradient(135deg, rgba(238,238,238,0.4) 0%, rgba(220,220,220,0.4) 100%);
                box-shadow: inset 0 0 10px rgba(0,0,0,0.05);
                margin-bottom: 10px;
            ">
                <span style="font-size: 28px; margin-bottom: 8px;">🖼️</span>
                <span>{t('未下载封面')}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # 显示标题
    st.markdown(f"<p style='font-size: 15px; font-weight: bold; text-align: center; margin-top: 8px; margin-bottom: 15px; color: #333; line-height: 1.4;'>{video_title}</p>", unsafe_allow_html=True)
    # ──────────────────────────────
    
    # ── 2. 文件浏览区域 (Output Files) ──
    st.divider()
    st.markdown(f"### 📂 {t('Output Files')}")
    output_dir = "output"
    
    if not os.path.exists(output_dir):
        st.info("No output directory yet.")
    else:
        # Initialize expanded paths in session state
        if "expanded_dirs" not in st.session_state:
            st.session_state.expanded_dirs = set()

        # Toolbar
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🔄 " + t("Refresh"), key="refresh_files", use_container_width=True):
                st.rerun()
        with col_btn2:
            if st.button("📁 " + "全部收起", key="collapse_all", use_container_width=True):
                st.session_state.expanded_dirs = set()
                st.rerun()

        def render_tree_node(current_path, level=0):
            try:
                items = sorted(os.listdir(current_path))
            except Exception:
                return

            dirs = [d for d in items if os.path.isdir(os.path.join(current_path, d)) and not d.startswith('.') and d != '__pycache__']
            files = [f for f in items if os.path.isfile(os.path.join(current_path, f)) and not f.startswith('.')]

            for d in dirs:
                full_path = os.path.join(current_path, d)
                is_expanded = full_path in st.session_state.expanded_dirs
                
                # Windows style: Arrow + Folder Icon + Name
                arrow = "▼" if is_expanded else "▶"
                folder_icon = "📂" if is_expanded else "📁"
                indent = "　" * level # Ideographic Space for visual structure
                
                if st.button(f"{indent}{arrow} {folder_icon} {d}", key=f"btn_{full_path}"):
                    if is_expanded:
                        st.session_state.expanded_dirs.remove(full_path)
                    else:
                        st.session_state.expanded_dirs.add(full_path)
                    st.rerun()
                
                if is_expanded:
                    render_tree_node(full_path, level + 1)

            for f in files:
                indent = "　" * level + "　　" # Align with folder text
                full_path = os.path.join(current_path, f)
                safe_key = f"dl_{full_path.replace('/', '_').replace('.', '_').replace('-', '_').replace(' ', '_')}"
                
                # Determine MIME type for download compatibility
                mime_type = "application/octet-stream"
                ext = os.path.splitext(f)[1].lower()
                if ext in ['.srt', '.vtt', '.txt']:
                    mime_type = "text/plain"
                elif ext == '.json':
                    mime_type = "application/json"
                elif ext == '.mp4':
                    mime_type = "video/mp4"
                elif ext == '.mp3':
                    mime_type = "audio/mpeg"
                
                # 🚀 优先使用惰性加载数据，避免在大文件加载时占用内存并极大地提升首屏响应性能
                def get_file_data(path=full_path):
                    with open(path, "rb") as file_data:
                        return file_data.read()

                try:
                    st.download_button(
                        label=f"{indent}📄 {f}",
                        data=get_file_data,
                        file_name=f,
                        mime=mime_type,
                        key=safe_key
                    )
                except Exception:
                    # 🛡️ 降级：如果遇到未知渲染错误，则退化回只读文本显示，这能彻底避免重复同一个 key 实例化组件导致的 React 越界崩溃
                    st.markdown(f"<p style='margin: 0; padding: 2px 5px; font-size: 0.9em; color: #555;'>{indent}📄 {f}</p>", unsafe_allow_html=True)

        # Scrollable container for the tree with custom class for styling
        st.markdown('<div class="windows-file-tree">', unsafe_allow_html=True)
        with st.container(height=600, border=True):
            render_tree_node(output_dir)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── 3. 视频编码对比表格 (整体放到最下方) ──
    st.divider()
    video_comparison_section()

def main():
    if "download_status_toast" in st.session_state:
        toast_msg, toast_icon = st.session_state.download_status_toast
        st.toast(toast_msg, icon=toast_icon)
        del st.session_state.download_status_toast

    st.markdown(button_style, unsafe_allow_html=True)
    
    # ── Split Main Body into Center and Right ──
    col_center, col_right = st.columns([7, 3])

    # Important: Render Right Sidebar FIRST to avoid it disappearing during blocking operations in Center
    with col_right:
        file_browser()

    with col_center:
        welcome_text = t("Welcome to VideoLingo Q")
        st.markdown(f"""
            <div class="top-header">
                <p class="top-header-text">{welcome_text}</p>
            </div>
        """, unsafe_allow_html=True)

        # add settings (Left part is st.sidebar)
        with st.sidebar:
            page_setting()
        
        download_video_section()
        text_processing_section()
        audio_processing_section()

if __name__ == "__main__":
    main()
