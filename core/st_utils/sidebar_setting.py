import json
import os
import streamlit as st
from time import sleep
from translations.translations import translate as t
from translations.translations import DISPLAY_LANGUAGES
from core.utils import *

# ── LLM Profile helpers ──────────────────────────────────────────────────────
LLM_PROFILES_PATH = "llm_profiles.json"
LLM_PROFILES_EXAMPLE_PATH = "llm_profiles.example.json"

def _load_profiles():
    if not os.path.exists(LLM_PROFILES_PATH):
        if os.path.exists(LLM_PROFILES_EXAMPLE_PATH):
            try:
                import shutil
                shutil.copy(LLM_PROFILES_EXAMPLE_PATH, LLM_PROFILES_PATH)
            except Exception:
                return {}
        else:
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
        "use_antigravity_cli": load_key("api.use_antigravity_cli"),
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

        config_input(t("Global Proxy"), "proxy", help=t("Global network proxy settings (e.g. http://127.0.0.1:7890). Leave empty to use system default."))

        st.divider()

        # ── Tab-based Settings ──────────────────────────────────────────
        tab_llm, tab_sub, tab_dub = st.tabs([
            "LLM", 
            "字幕", 
            "配音"
        ])

        with tab_llm:
            # ── Antigravity CLI Toggle (Now at the top) ────────────────────────
            use_antigravity_cli = st.toggle(t("Use Antigravity CLI"), value=load_key("api.use_antigravity_cli"), help=t("If enabled, use antigravity-cli to call LLM, ignoring above settings"))
            if use_antigravity_cli != load_key("api.use_antigravity_cli"):
                update_key("api.use_antigravity_cli", use_antigravity_cli)

            if use_antigravity_cli:
                antigravity_token_code = st.text_input(
                    t("Antigravity Token Code"),
                    value=load_key("api.antigravity_token_code"),
                    placeholder=t("Enter Authorization Code from browser"),
                    help=t("When your agy login expires during run, paste the google authorization code here and click retry to automatically authenticate and resume.")
                )
                if antigravity_token_code != load_key("api.antigravity_token_code"):
                    update_key("api.antigravity_token_code", antigravity_token_code)
            
            if not use_antigravity_cli:
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
                            if "use_antigravity_cli" in p:
                                update_key("api.use_antigravity_cli", p["use_antigravity_cli"])
                            elif "use_gemini_cli" in p:
                                update_key("api.use_antigravity_cli", p["use_gemini_cli"])
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

        with tab_sub:
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

            if efficiency_mode:
                batch_split_size = st.number_input(
                    t("Batch Split Size"),
                    min_value=0,
                    max_value=200,
                    value=load_key("batch_split_size") if load_key("batch_split_size") is not None else 100,
                    help=t("Maximum number of sentences sent to LLM in one batch. Set to 0 to send all sentences at once. Reduce this value if you experience model context or timeout errors.")
                )
                if batch_split_size != load_key("batch_split_size"):
                    update_key("batch_split_size", batch_split_size)

                batch_translate_size = st.number_input(
                    t("Batch Translate Size"),
                    min_value=0,
                    max_value=200,
                    value=load_key("batch_translate_size") if load_key("batch_translate_size") is not None else 15,
                    help=t("Maximum number of subtitle lines translated by LLM in one batch. Set to 0 to send all. Reduce this value if translation experiences timeouts.")
                )
                if batch_translate_size != load_key("batch_translate_size"):
                    update_key("batch_translate_size", batch_translate_size)

            st.divider()
            st.markdown(f"**YouTube {t('Settings')}**")
            
            cookies_from_browser_options = ["", "chrome", "firefox", "edge", "safari", "opera", "brave"]
            current_cfb = load_key("youtube.cookies_from_browser")
            if current_cfb not in cookies_from_browser_options:
                cookies_from_browser_options.append(current_cfb)
            
            selected_cfb = st.selectbox(
                t("Extract Cookies From Browser"),
                options=cookies_from_browser_options,
                index=cookies_from_browser_options.index(current_cfb) if current_cfb else 0,
                help=t("Select browser to automatically extract YouTube cookies to bypass bot checks. Leave empty if not needed.")
            )
            if selected_cfb != current_cfb:
                update_key("youtube.cookies_from_browser", selected_cfb)

            # Cookies 文件上传与管理组件
            current_cookies_path = load_key("youtube.cookies_path")
            has_cookies_file = False
            if current_cookies_path and os.path.exists(current_cookies_path):
                has_cookies_file = True

            if has_cookies_file:
                st.info(f"📁 {t('Saved cookies file')}: `{current_cookies_path}`")
                if st.button(t("Delete Saved Cookies"), key="delete_cookies_button", use_container_width=True):
                    try:
                        if os.path.exists(current_cookies_path):
                            os.remove(current_cookies_path)
                        update_key("youtube.cookies_path", "")
                        st.success(t("Saved cookies file deleted successfully!"))
                        sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"删除失败: {e}")
            else:
                uploaded_cookies = st.file_uploader(
                    t("Upload YouTube Cookies (.txt)"), 
                    type=["txt"], 
                    key="youtube_cookies_uploader",
                    help=t("Upload Netscape format cookies.txt file to bypass bot checks.")
                )
                if uploaded_cookies is not None:
                    try:
                        cookies_save_dir = "output"
                        os.makedirs(cookies_save_dir, exist_ok=True)
                        target_cookies_path = os.path.join(cookies_save_dir, "youtube_cookies.txt")
                        with open(target_cookies_path, "wb") as f:
                            f.write(uploaded_cookies.getbuffer())
                        update_key("youtube.cookies_path", target_cookies_path)
                        st.success(t("Cookies uploaded and saved successfully!"))
                        sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"保存失败: {e}")

        with tab_dub:
            tts_methods = ["edge_tts", "doubao_tts"]
            current_tts = load_key("tts_method")
            if current_tts not in tts_methods:
                current_tts = "edge_tts"
                update_key("tts_method", "edge_tts")
            select_tts = st.selectbox(t("TTS Method"), options=tts_methods, index=tts_methods.index(current_tts))
            if select_tts != load_key("tts_method"):
                update_key("tts_method", select_tts)

            # TTS Concurrency (Batch Size)
            tts_max_workers = st.slider(t("TTS Batch Size"), min_value=1, max_value=10, value=load_key("tts_max_workers"), help=t("Number of audio files to generate in parallel. Increase for faster processing, but be aware of API limits."))
            if tts_max_workers != load_key("tts_max_workers"):
                update_key("tts_max_workers", tts_max_workers)

            # Skip Reference Audio Toggle
            skip_refer = st.toggle(t("Skip Reference Audio"), value=load_key("skip_refer"), help=t("Skip extracting reference audio from the original video. Useful when using fixed TTS voices."))
            if skip_refer != load_key("skip_refer"):
                update_key("skip_refer", skip_refer)

            # sub settings for each tts method
            if select_tts == "doubao_tts":
                config_input("Volcengine AppID", "doubao_tts.appid")
                config_input("Volcengine Access Token", "doubao_tts.access_token")
                doubao_voices = {
                    "vivi 2.0 (中女·旗舰)": "zh_female_vv_uranus_bigtts",
                    "流畅女声 2.0 (视频配音)": "zh_female_liuchangnv_uranus_bigtts",
                    "儒雅逸辰 2.0 (视频配音)": "zh_male_ruyayichen_uranus_bigtts",
                    "温柔妈妈 2.0 (通用)": "zh_female_wenroumama_uranus_bigtts",
                    "解说小明 2.0 (通用)": "zh_male_jieshuoxiaoming_uranus_bigtts",
                    "TVB女声 2.0 (通用)": "zh_female_tvbnv_uranus_bigtts",
                    "译制片男 2.0 (通用)": "zh_male_yizhipiannan_uranus_bigtts",
                    "俏皮女声 2.0 (通用)": "zh_female_qiaopinv_uranus_bigtts",
                    "直率英子 2.0 (通用)": "zh_female_zhishuaiyingzi_uranus_bigtts",
                    "邻家男孩 2.0 (通用)": "zh_male_linjiananhai_uranus_bigtts",
                    "四郎 2.0 (通用)": "zh_male_silang_uranus_bigtts",
                    "小何 (中女·旗舰)": "zh_female_xiaohe_uranus_bigtts",
                    "云舟 (中男·旗舰)": "zh_male_m191_uranus_bigtts",
                    "小天 (中男·旗舰)": "zh_male_taocheng_uranus_bigtts",
                    "Tim (英男·旗舰)": "en_male_tim_uranus_bigtts",
                    "知性灿灿 (中女·Saturn)": "saturn_zh_female_cancan_tob",
                    "可爱女生 (中女·Saturn)": "saturn_zh_female_keainvsheng_tob",
                    "调皮公主 (中女·Saturn)": "saturn_zh_female_tiaopigongzhu_tob",
                    "爽朗少年 (中男·Saturn)": "saturn_zh_male_shuanglangshaonian_tob",
                    "天才同桌 (中男·Saturn)": "saturn_zh_male_tiancaitongzhu_tob",
                }
                current_doubao_voice = load_key("doubao_tts.voice")
                current_display = next((k for k, v in doubao_voices.items() if v == current_doubao_voice), list(doubao_voices.keys())[0])
                selected_display = st.selectbox(
                    t("Doubao Voice"),
                    options=list(doubao_voices.keys()),
                    index=list(doubao_voices.keys()).index(current_display)
                )
                selected_voice = doubao_voices[selected_display]
                if selected_voice != current_doubao_voice:
                    update_key("doubao_tts.voice", selected_voice)
                
                # 🔊 Preview Button for Doubao TTS
                if st.button(f"🔊 {t('Listen Preview')}", key="preview_doubao_tts", use_container_width=True):
                    preview_text = "这是您的豆包2.0配音预览，听起来不错吧？" if load_key("display_language") == "zh-CN" else "This is your Doubao 2.0 voice preview, sounds good, right?"
                    import tempfile
                    from core.tts_backend.doubao_tts import doubao_tts as gen_doubao_tts
                    with st.spinner(t("Generating preview...")):
                        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
                            tmp_path = tmp_file.name
                        try:
                            gen_doubao_tts(preview_text, tmp_path)
                            with open(tmp_path, "rb") as f:
                                audio_bytes = f.read()
                            st.audio(audio_bytes, format="audio/mp3")
                        except Exception as e:
                            st.error(f"Preview failed: {str(e)}")
                        finally:
                            if os.path.exists(tmp_path):
                                try: os.remove(tmp_path)
                                except: pass
            
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
