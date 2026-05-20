import json
import os
import streamlit as st
from translations.translations import translate as t
from translations.translations import DISPLAY_LANGUAGES
from core.utils import *

# ── LLM Profile helpers ──────────────────────────────────────────────────────
LLM_PROFILES_PATH = "llm_profiles.json"

def _load_profiles():
    if not os.path.exists(LLM_PROFILES_PATH):
        return {}
    try:
        with open(LLM_PROFILES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}

def _save_profiles(profiles: dict):
    with open(LLM_PROFILES_PATH, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)

def _current_api_values():
    return {
        "key":              load_key("api.key"),
        "base_url":         load_key("api.base_url"),
        "model":            load_key("api.model"),
        "llm_support_json": load_key("api.llm_support_json"),
        "use_gemini_cli":   load_key("api.use_gemini_cli"),
    }

def config_input(label, key, help=None):
    """Generic config input handler"""
    val = st.text_input(label, value=load_key(key), help=help)
    if val != load_key(key):
        update_key(key, val)
    return val

def page_setting():
    try:
        from streamlit import fragment as st_fragment
    except ImportError:
        try:
            from streamlit import experimental_fragment as st_fragment
        except ImportError:
            def st_fragment(func): return func

    @st_fragment
    def _render_sidebar():
        display_language = st.selectbox("Display Language 🌐", 
                                    options=list(DISPLAY_LANGUAGES.keys()),
                                    index=list(DISPLAY_LANGUAGES.values()).index(load_key("display_language")))
        if DISPLAY_LANGUAGES[display_language] != load_key("display_language"):
            update_key("display_language", DISPLAY_LANGUAGES[display_language])
            st.rerun()

        # with st.expander(t("Youtube Settings"), expanded=True):
        #     config_input(t("Cookies Path"), "youtube.cookies_path")

        with st.expander(t("LLM Configuration"), expanded=True):
            # ── Gemini CLI Toggle (Now at the top) ────────────────────────
            use_gemini_cli = st.toggle(t("Use Gemini CLI"), value=load_key("api.use_gemini_cli"), help=t("If enabled, use gemini-cli to call LLM, ignoring above settings"))
            if use_gemini_cli != load_key("api.use_gemini_cli"):
                update_key("api.use_gemini_cli", use_gemini_cli)
                # No st.rerun() here to allow fragment-only update
            
            if not use_gemini_cli:
                # ── Profile selector (only when profiles exist) ──────────────────
                profiles = _load_profiles()
                if profiles:
                    profile_names = list(profiles.keys())
                    st.markdown(f"**{t('LLM Profiles')}**")
                    selected_profile = st.selectbox(
                        t("Select a profile to load or delete"),
                        options=profile_names,
                        label_visibility="collapsed",
                        key="llm_profile_select",
                    )
                    col_load, col_del = st.columns(2)
                    with col_load:
                        if st.button(f"📥 {t('Load Profile')}", key="llm_profile_load", use_container_width=True):
                            p = profiles[selected_profile]
                            update_key("api.key",              p["key"])
                            update_key("api.base_url",         p["base_url"])
                            update_key("api.model",            p["model"])
                            update_key("api.llm_support_json", p["llm_support_json"])
                            if "use_gemini_cli" in p:
                                update_key("api.use_gemini_cli", p["use_gemini_cli"])
                            st.toast(t("Profile loaded"), icon="✅")
                            st.rerun()
                    with col_del:
                        if st.button(f"🗑️ {t('Delete Profile')}", key="llm_profile_delete", use_container_width=True):
                            del profiles[selected_profile]
                            _save_profiles(profiles)
                            st.toast(t("Profile deleted"), icon="🗑️")
                            st.rerun()

                # ── Save current config as named profile ─────────────────────────
                st.markdown(f"**{t('Save Current Config')}**")
                profile_name_input = st.text_input(
                    t("Profile Name"),
                    value="",
                    placeholder=t("Enter profile name"),
                    key="llm_profile_name",
                    label_visibility="collapsed",
                )
                if st.button(f"💾 {t('Save Current Configuration')}", key="llm_profile_save", use_container_width=True):
                    name = profile_name_input.strip()
                    if name:
                        profiles = _load_profiles()
                        profiles[name] = _current_api_values()
                        _save_profiles(profiles)
                        st.toast(t("Profile saved"), icon="💾")
                        st.rerun()
                    else:
                        st.toast(t("Please enter a profile name"), icon="⚠️")

                st.divider()

                # ── Existing API inputs (Conditional) ──────────────────────────
                config_input(t("API_KEY"), "api.key")
                config_input(t("BASE_URL"), "api.base_url", help=t("Openai format, will add /v1/chat/completions automatically"))

                c1, c2 = st.columns([4, 1])
                with c1:
                    config_input(t("MODEL"), "api.model", help=t("click to check API validity") + " 👉")
                with c2:
                    if st.button("📡", key="api"):
                        st.toast(t("API Key is valid") if check_api() else t("API Key is invalid"),
                                icon="✅" if check_api() else "❌")
                llm_support_json = st.toggle(t("LLM JSON Format Support"), value=load_key("api.llm_support_json"), help=t("Enable if your LLM supports JSON mode output"))
                if llm_support_json != load_key("api.llm_support_json"):
                    update_key("api.llm_support_json", llm_support_json)

        with st.expander(t("Subtitles Settings"), expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                langs = {
                    "🇺🇸 English": "en",
                    "🇨🇳 简体中文": "zh",
                    "🇪🇸 Español": "es",
                    "🇷🇺 Русский": "ru",
                    "🇫🇷 Français": "fr",
                    "🇩🇪 Deutsch": "de",
                    "🇮🇹 Italiano": "it",
                    "🇯🇵 日本語": "ja"
                }
                lang = st.selectbox(
                    t("Recog Lang"),
                    options=list(langs.keys()),
                    index=list(langs.values()).index(load_key("whisper.language"))
                )
                if langs[lang] != load_key("whisper.language"):
                    update_key("whisper.language", langs[lang])

            import platform as _platform
            _runtime_options = ["local", "cloud", "elevenlabs"]
            if _platform.system() == "Darwin":
                _runtime_options.insert(1, "mlx")
            _current_runtime = load_key("whisper.runtime")
            if _current_runtime not in _runtime_options:
                _current_runtime = "local"
            runtime = st.selectbox(
                t("WhisperX Runtime"),
                options=_runtime_options,
                index=_runtime_options.index(_current_runtime),
                help=t("Local runtime requires >8GB GPU, cloud runtime requires 302ai API key, elevenlabs runtime requires ElevenLabs API key") + (" | mlx: Apple Silicon accelerated (pip install mlx-whisper)" if _platform.system() == "Darwin" else "")
            )
            if runtime != load_key("whisper.runtime"):
                update_key("whisper.runtime", runtime)
                st.rerun()
            if runtime == "cloud":
                config_input(t("WhisperX 302ai API"), "whisper.whisperX_302_api_key")
            if runtime == "elevenlabs":
                config_input(("ElevenLabs API"), "whisper.elevenlabs_api_key")

            with c2:
                target_language = st.text_input(t("Target Lang"), value=load_key("target_language"), help=t("Input any language in natural language, as long as llm can understand"))
                if target_language != load_key("target_language"):
                    update_key("target_language", target_language)

            demucs = st.toggle(t("Vocal separation enhance"), value=load_key("demucs"), help=t("Recommended for videos with loud background noise, but will increase processing time"))
            if demucs != load_key("demucs"):
                update_key("demucs", demucs)
            
            burn_subtitles = st.toggle(t("Burn-in Subtitles"), value=load_key("burn_subtitles"), help=t("Whether to burn subtitles into the video, will increase processing time"))
            if burn_subtitles != load_key("burn_subtitles"):
                update_key("burn_subtitles", burn_subtitles)

            auto_burn_subtitles = st.toggle(t("Auto Burn Subtitles"), value=load_key("auto_burn_subtitles"), help=t("Automatically burn subtitles to video after processing, skipping the manual review step"))
            if auto_burn_subtitles != load_key("auto_burn_subtitles"):
                update_key("auto_burn_subtitles", auto_burn_subtitles)

            domain_vocab = st.text_area(
                t("Domain Vocabulary"),
                value=load_key("domain_vocab"),
                help=t("Enter domain-specific terms (comma or newline separated) to improve ASR accuracy. Use 'wrong -> correct' for explicit replacement. E.g.: Claude Cowork, called cowork -> Claude Cowork"),
                placeholder="Claude Cowork\ncalled cowork -> Claude Cowork",
                height=80,
            )
            if domain_vocab != load_key("domain_vocab"):
                update_key("domain_vocab", domain_vocab)

            efficiency_mode = st.toggle(t("Efficiency Mode"), value=load_key("efficiency_mode"), help=t("Send all subtitle data to LLM in large batches to reduce API calls. Automatically falls back to standard mode on failure."))
            if efficiency_mode != load_key("efficiency_mode"):
                update_key("efficiency_mode", efficiency_mode)
        with st.expander(t("Dubbing Settings"), expanded=True):
            tts_methods = ["azure_tts", "openai_tts", "fish_tts", "sf_fish_tts", "edge_tts", "gpt_sovits", "custom_tts", "sf_cosyvoice2", "f5tts"]
            select_tts = st.selectbox(t("TTS Method"), options=tts_methods, index=tts_methods.index(load_key("tts_method")))
            if select_tts != load_key("tts_method"):
                update_key("tts_method", select_tts)

            # sub settings for each tts method
            if select_tts == "sf_fish_tts":
                config_input(t("SiliconFlow API Key"), "sf_fish_tts.api_key")
                
                # Add mode selection dropdown
                mode_options = {
                    "preset": t("Preset"),
                    "custom": t("Refer_stable"),
                    "dynamic": t("Refer_dynamic")
                }
                selected_mode = st.selectbox(
                    t("Mode Selection"),
                    options=list(mode_options.keys()),
                    format_func=lambda x: mode_options[x],
                    index=list(mode_options.keys()).index(load_key("sf_fish_tts.mode")) if load_key("sf_fish_tts.mode") in mode_options.keys() else 0
                )
                if selected_mode != load_key("sf_fish_tts.mode"):
                    update_key("sf_fish_tts.mode", selected_mode)
                if selected_mode == "preset":
                    config_input("Voice", "sf_fish_tts.voice")

            elif select_tts == "openai_tts":
                config_input("302ai API", "openai_tts.api_key")
                config_input(t("OpenAI Voice"), "openai_tts.voice")

            elif select_tts == "fish_tts":
                config_input("302ai API", "fish_tts.api_key")
                fish_tts_character = st.selectbox(t("Fish TTS Character"), options=list(load_key("fish_tts.character_id_dict").keys()), index=list(load_key("fish_tts.character_id_dict").keys()).index(load_key("fish_tts.character")))
                if fish_tts_character != load_key("fish_tts.character"):
                    update_key("fish_tts.character", fish_tts_character)

            elif select_tts == "azure_tts":
                config_input("302ai API", "azure_tts.api_key")
                config_input(t("Azure Voice"), "azure_tts.voice")
            
            elif select_tts == "gpt_sovits":
                st.info(t("Please refer to Github homepage for GPT_SoVITS configuration"))
                config_input(t("SoVITS Character"), "gpt_sovits.character")
                
                refer_mode_options = {1: t("Mode 1: Use provided reference audio only"), 2: t("Mode 2: Use first audio from video as reference"), 3: t("Mode 3: Use each audio from video as reference")}
                selected_refer_mode = st.selectbox(
                    t("Refer Mode"),
                    options=list(refer_mode_options.keys()),
                    format_func=lambda x: refer_mode_options[x],
                    index=list(refer_mode_options.keys()).index(load_key("gpt_sovits.refer_mode")),
                    help=t("Configure reference audio mode for GPT-SoVITS")
                )
                if selected_refer_mode != load_key("gpt_sovits.refer_mode"):
                    update_key("gpt_sovits.refer_mode", selected_refer_mode)
                    
            elif select_tts == "edge_tts":
                edge_tts_voices = {
                    "zh-CN-XiaoxiaoNeural (晓晓·女声·新闻)": "zh-CN-XiaoxiaoNeural",
                    "zh-CN-YunxiNeural (云希·男声·通用)": "zh-CN-YunxiNeural",
                    "zh-CN-YunjianNeural (云健·男声·解说)": "zh-CN-YunjianNeural",
                    "zh-CN-XiaoyiNeural (晓伊·女声·故事)": "zh-CN-XiaoyiNeural",
                    "zh-CN-YunxiaNeural (云夏·男声·卡通)": "zh-CN-YunxiaNeural",
                    "zh-CN-YunyangNeural (云扬·男声·新闻)": "zh-CN-YunyangNeural",
                    "zh-CN-liaoning-XiaobeiNeural (晓北·辽宁方言)": "zh-CN-liaoning-XiaobeiNeural",
                    "zh-CN-shaanxi-XiaoniNeural (晓妮·陕西方言)": "zh-CN-shaanxi-XiaoniNeural",
                    "zh-HK-HiuMaanNeural (晓曼·港漫女声)": "zh-HK-HiuMaanNeural",
                    "zh-HK-WanLungNeural (云龙·香港男声)": "zh-HK-WanLungNeural",
                    "zh-TW-HsiaoChenNeural (晓臻·台湾女声)": "zh-TW-HsiaoChenNeural",
                    "zh-TW-YunJheNeural (云哲·台湾男声)": "zh-TW-YunJheNeural",
                    "en-US-JennyNeural (Jenny·US Female)": "en-US-JennyNeural",
                    "en-US-GuyNeural (Guy·US Male)": "en-US-GuyNeural",
                    "en-US-AriaNeural (Aria·US Female)": "en-US-AriaNeural",
                    "en-US-ChristopherNeural (Chris·US Male)": "en-US-ChristopherNeural",
                    "en-GB-SoniaNeural (Sonia·UK Female)": "en-GB-SoniaNeural",
                    "en-GB-RyanNeural (Ryan·UK Male)": "en-GB-RyanNeural",
                }
                current_voice = load_key("edge_tts.voice")
                # find display name for current value, fallback to first option
                current_display = next((k for k, v in edge_tts_voices.items() if v == current_voice), list(edge_tts_voices.keys())[0])
                selected_display = st.selectbox(
                    t("Edge TTS Voice"),
                    options=list(edge_tts_voices.keys()),
                    index=list(edge_tts_voices.keys()).index(current_display)
                )
                selected_voice = edge_tts_voices[selected_display]
                if selected_voice != current_voice:
                    update_key("edge_tts.voice", selected_voice)
                
                # 🔊 Preview Button for Edge TTS
                if st.button(f"🔊 {t('Listen Preview')}", key="preview_edge_tts", use_container_width=True):
                    preview_text = "这是您的配音预览，听起来还不错吧？" if selected_voice.startswith("zh-") else "This is your voice preview, sounds good, right?"
                    import tempfile
                    import subprocess
                    with st.spinner(t("Generating preview...")):
                        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
                            tmp_path = tmp_file.name
                        try:
                            cmd = ["edge-tts", "--voice", selected_voice, "--text", preview_text, "--write-media", tmp_path]
                            subprocess.run(cmd, check=True, capture_output=True)
                            with open(tmp_path, "rb") as f:
                                audio_bytes = f.read()
                            st.audio(audio_bytes, format="audio/mp3")
                        except Exception as e:
                            st.error(f"Preview failed: {str(e)}")
                        finally:
                            if os.path.exists(tmp_path):
                                try: os.remove(tmp_path)
                                except: pass

            elif select_tts == "sf_cosyvoice2":
                config_input(t("SiliconFlow API Key"), "sf_cosyvoice2.api_key")
            
            elif select_tts == "f5tts":
                config_input("302ai API", "f5tts.302_api")

    _render_sidebar()
        
def check_api():
    try:
        resp = ask_gpt("This is a test, response 'message':'success' in json format.", 
                      resp_type="json", log_title='None')
        return resp.get('message') == 'success'
    except Exception:
        return False
    
if __name__ == "__main__":
    check_api()
