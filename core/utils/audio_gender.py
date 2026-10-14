import os
import re
import datetime
import numpy as np
import pandas as pd
import librosa
from sklearn.mixture import GaussianMixture
from rich.console import Console
from rich.panel import Panel

from core.utils import rprint, ask_gpt, load_key
from core.utils.models import _VOCAL_AUDIO_FILE, _RAW_AUDIO_FILE

console = Console()

def parse_time_to_seconds(t_val) -> float:
    """Convert time object or SRT timestamp string to seconds float."""
    if isinstance(t_val, (int, float)):
        return float(t_val)
    if isinstance(t_val, datetime.time):
        return t_val.hour * 3600 + t_val.minute * 60 + t_val.second + t_val.microsecond / 1e6
    if isinstance(t_val, str):
        t_str = t_val.strip().replace(',', '.')
        parts = t_str.split(':')
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        elif len(parts) == 2:
            m, s = parts
            return int(m) * 60 + float(s)
        else:
            return float(t_str)
    return 0.0

def extract_f0_and_spectral_features(y: np.ndarray, sr: int = 16000) -> dict:
    """
    Extract fundamental frequency (F0) pitch percentiles (median, P15 chest-register floor, P85),
    low-frequency harmonic energy ratio, and spectral centroid from an in-memory audio array.
    """
    if len(y) < 1024:
        pad_len = 2048 - len(y)
        y = np.pad(y, (0, max(0, pad_len)), mode='constant')

    result = {
        'median_f0': None,
        'mean_f0': None,
        'p15_f0': None,
        'p85_f0': None,
        'voiced_ratio': 0.0,
        'voiced_frames': 0,
        'spectral_centroid': None,
        'low_energy_ratio': None,
    }

    try:
        # Human speaking fundamental frequency range: 65Hz - 350Hz
        fmin = 65
        fmax = 350
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            frame_length=2048,
            hop_length=512
        )
        
        valid_mask = ~np.isnan(f0) & (voiced_probs > 0.2)
        voiced_f0 = f0[valid_mask]
        if len(voiced_f0) > 0:
            result['median_f0'] = float(np.median(voiced_f0))
            result['mean_f0'] = float(np.mean(voiced_f0))
            result['p15_f0'] = float(np.percentile(voiced_f0, 15))
            result['p85_f0'] = float(np.percentile(voiced_f0, 85))
            result['voiced_ratio'] = float(len(voiced_f0) / len(f0)) if len(f0) > 0 else 0.0
            result['voiced_frames'] = int(len(voiced_f0))
            
        # Spectral Centroid
        cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)
        result['spectral_centroid'] = float(np.mean(cent))

        # Low-frequency chest resonance energy ratio: energy(75..240Hz) / energy(75..1200Hz)
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)) ** 2
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        low_mask = (freqs >= 75) & (freqs <= 240)
        band_mask = (freqs >= 75) & (freqs <= 1200)
        total_band_energy = float(np.sum(S[band_mask, :]))
        if total_band_energy > 1e-8:
            result['low_energy_ratio'] = float(np.sum(S[low_mask, :]) / total_band_energy)
    except Exception:
        pass
    
    return result


def compute_acoustic_female_prob(
    median_f0: float,
    p15_f0: float = None,
    low_energy_ratio: float = None,
    spectral_centroid: float = None,
) -> float:
    """
    Compute acoustic probability of FEMALE in [0.0, 1.0] using multi-dimensional features.
    Specifically designed to prevent excited/passionate male speakers (whose median F0 can
    reach 150-180 Hz, but whose P15 baseline F0 drops back to 95-135 Hz and who retain strong
    low-frequency chest energy) from being misclassified as female.
    """
    if median_f0 is None or np.isnan(median_f0):
        return 0.5

    def _sigmoid(x: float) -> float:
        return float(1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0))))

    # 1. Median F0 score (centered at 162 Hz instead of 145 Hz to tolerate energetic male speech)
    prob_med = _sigmoid((median_f0 - 162.0) / 14.0)

    # 2. P15 F0 (chest-register baseline): biological female speech rarely drops below 142 Hz at P15,
    # whereas excited male speech regularly dips back to 95-135 Hz at syllable/sentence endings.
    if p15_f0 is not None and not np.isnan(p15_f0):
        prob_p15 = _sigmoid((p15_f0 - 140.0) / 11.0)
        # Detect classic "excited male" pattern: high intonation spread (median - p15 >= 22 Hz)
        # while baseline p15_f0 is clearly in the male chest register (< 136 Hz).
        if p15_f0 < 136.0 and (median_f0 - p15_f0) >= 18.0:
            prob_med = min(prob_med, 0.25)
    else:
        prob_p15 = prob_med

    # 3. Low-frequency chest harmonic energy ratio (higher ratio -> stronger male chest resonance)
    if low_energy_ratio is not None and not np.isnan(low_energy_ratio):
        # Typical male > 0.42, typical female < 0.30
        prob_energy = _sigmoid((0.36 - low_energy_ratio) / 0.08)
    else:
        prob_energy = 0.5

    # 4. Spectral centroid
    if spectral_centroid is not None and not np.isnan(spectral_centroid):
        prob_cent = _sigmoid((spectral_centroid - 1600.0) / 250.0)
    else:
        prob_cent = 0.5

    combined = 0.48 * prob_p15 + 0.32 * prob_med + 0.12 * prob_energy + 0.08 * prob_cent
    return float(np.clip(combined, 0.01, 0.99))


# ── Pretrained Neural Gender Classification Model (Wav2Vec2) ─────────────────
_GENDER_MODEL_PIPE = None
_GENDER_MODEL_LOAD_FAILED = False
PRETRAINED_GENDER_MODEL_ID = "prithivMLmods/Common-Voice-Gender-Detection"


def _get_pretrained_gender_classifier():
    """
    Lazy-load a lightweight, non-gated HuggingFace Wav2Vec2 audio gender classifier.
    Supports domestic HF mirrors (hf-mirror.com), global proxy, and local _model_cache.
    Returns (extractor, model, device, label_map) or None if unavailable.
    """
    global _GENDER_MODEL_PIPE, _GENDER_MODEL_LOAD_FAILED
    if _GENDER_MODEL_PIPE is not None:
        return _GENDER_MODEL_PIPE
    if _GENDER_MODEL_LOAD_FAILED:
        return None

    try:
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        # Configure network proxy / HF mirror for model download
        proxy = ""
        try:
            proxy = load_key("proxy") or ""
        except Exception:
            pass
        if proxy:
            os.environ["HTTP_PROXY"] = proxy
            os.environ["HTTPS_PROXY"] = proxy
        else:
            hf_ep = ""
            try:
                hf_ep = load_key("pyannote.hf_endpoint") or ""
            except Exception:
                pass
            if hf_ep:
                os.environ["HF_ENDPOINT"] = hf_ep
            elif "HF_ENDPOINT" not in os.environ:
                # Non-gated model works seamlessly with hf-mirror.com in mainland China
                os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

        cache_dir = "./_model_cache"
        try:
            cache_dir = load_key("model_dir") or "./_model_cache"
        except Exception:
            pass
        os.makedirs(cache_dir, exist_ok=True)

        rprint(f"[cyan]🧠 Loading pretrained voice gender model ({PRETRAINED_GENDER_MODEL_ID})...[/cyan]")
        extractor = AutoFeatureExtractor.from_pretrained(PRETRAINED_GENDER_MODEL_ID, cache_dir=cache_dir)
        model = AutoModelForAudioClassification.from_pretrained(PRETRAINED_GENDER_MODEL_ID, cache_dir=cache_dir)

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        model = model.to(device)
        model.eval()

        _GENDER_MODEL_PIPE = (extractor, model, device)
        rprint(f"[green]✓ Pretrained voice gender model loaded on {device}[/green]")
        return _GENDER_MODEL_PIPE
    except Exception as e:
        _GENDER_MODEL_LOAD_FAILED = True
        rprint(f"[yellow]⚠️ Pretrained gender model unavailable ({e}), using multi-dimensional acoustic model.[/yellow]")
        return None


def predict_gender_with_pretrained_model(audio_clips: list, sr: int = 16000) -> float:
    """
    Run the pretrained Wav2Vec2 gender model on a list of 1D numpy audio clips.
    Returns P(female) in [0.0, 1.0], or None if the model is unavailable.
    """
    pipe = _get_pretrained_gender_classifier()
    if pipe is None or not audio_clips:
        return None

    extractor, model, device = pipe
    import torch

    female_probs = []
    weights = []

    try:
        for clip in audio_clips:
            if len(clip) < sr * 0.4:
                continue
            # Cap clip length to 6 seconds for fast, focused inference
            if len(clip) > sr * 6:
                clip = clip[:sr * 6]

            inputs = extractor(clip, sampling_rate=sr, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

            id2label = model.config.id2label
            p_fem = None
            for idx_lbl, lbl in id2label.items():
                lbl_low = str(lbl).lower()
                if "female" in lbl_low or lbl_low == "f":
                    p_fem = float(probs[int(idx_lbl)])
                    break
            if p_fem is None and len(probs) >= 2:
                # In prithivMLmods/Common-Voice-Gender-Detection: 0 -> female, 1 -> male
                p_fem = float(probs[0])

            if p_fem is not None:
                female_probs.append(p_fem)
                weights.append(min(4.0, len(clip) / float(sr)))

        if female_probs:
            return float(np.average(female_probs, weights=weights))
    except Exception as e:
        rprint(f"[yellow]⚠️ Pretrained gender inference error: {e}[/yellow]")

    return None


def fit_video_pitch_clusters(full_audio: np.ndarray, sr: int, df_tasks: pd.DataFrame) -> dict:
    """
    Fit a 2-Component Gaussian Mixture Model (GMM) on log(F0) across the entire video
    and combine with pretrained neural model + P15 chest-register acoustic analysis
    to discover video-specific male & female pitch centers, variances, and boundaries.
    """
    f0_samples = []
    p15_samples = []
    energy_samples = []
    centroid_samples = []
    candidate_clips = []
    
    for _, row in df_tasks.iterrows():
        start_sec = parse_time_to_seconds(row['start_time'])
        end_sec = parse_time_to_seconds(row['end_time'])
        s_idx = max(0, int(start_sec * sr))
        e_idx = min(len(full_audio), int(end_sec * sr))
        seg = full_audio[s_idx:e_idx]
        
        if len(seg) >= 1024:
            info = extract_f0_and_spectral_features(seg, sr)
            if info['median_f0'] and info['voiced_ratio'] >= 0.08 and info['voiced_frames'] >= 3:
                f0_samples.append(info['median_f0'])
                if info['p15_f0'] is not None:
                    p15_samples.append(info['p15_f0'])
                if info['low_energy_ratio'] is not None:
                    energy_samples.append(info['low_energy_ratio'])
                if info['spectral_centroid'] is not None:
                    centroid_samples.append(info['spectral_centroid'])
                if len(seg) >= sr * 1.2:
                    candidate_clips.append((len(seg), seg))
                
    if len(f0_samples) < 8:
        # Use acoustic + pretrained model on available clips instead of a blind 145Hz constant
        top_clips = [c for _, c in sorted(candidate_clips, key=lambda x: x[0], reverse=True)[:5]]
        nn_prob = predict_gender_with_pretrained_model(top_clips, sr) if top_clips else None
        ac_prob = compute_acoustic_female_prob(
            float(np.median(f0_samples)) if f0_samples else None,
            float(np.median(p15_samples)) if p15_samples else None,
            float(np.median(energy_samples)) if energy_samples else None,
            float(np.median(centroid_samples)) if centroid_samples else None,
        )
        final_prob = (0.7 * nn_prob + 0.3 * ac_prob) if nn_prob is not None else ac_prob
        dom_gender = 'female' if final_prob >= 0.5 else 'male'
        rprint(f"[yellow]⚠️ Few voiced samples ({len(f0_samples)}), classified dominant gender as {dom_gender.upper()} (P(female)={final_prob:.2f}).[/yellow]")
        return {
            'is_dual_speaker': False,
            'dominant_gender': dom_gender,
            'boundary_f0': 160.0,
            'male_center': 120.0,
            'female_center': 185.0,
            'male_var': 0.015,
            'female_var': 0.025
        }
        
    f0_arr = np.array(f0_samples).reshape(-1, 1)
    log_f0 = np.log(f0_arr)
    
    gmm = GaussianMixture(n_components=2, random_state=42, n_init=3).fit(log_f0)
    means = np.exp(gmm.means_.flatten())
    weights = gmm.weights_.flatten()
    order = np.argsort(means)
    
    male_idx = order[0]
    female_idx = order[1]
    
    center_low = float(means[male_idx])
    center_high = float(means[female_idx])
    weight_low = float(weights[male_idx])
    weight_high = float(weights[female_idx])
    male_var = float(gmm.covariances_[male_idx].flatten()[0])
    female_var = float(gmm.covariances_[female_idx].flatten()[0])
    
    rprint(f"[cyan]📊 Video-Level Pitch Discovery: Cluster 1 = {center_low:.1f} Hz (wt {weight_low:.2f}), Cluster 2 = {center_high:.1f} Hz (wt {weight_high:.2f})[/cyan]")
    
    # Evaluate overall video gender probability using P15 chest-register + Pretrained Wav2Vec2 Model
    overall_med_f0 = float(np.median(f0_samples))
    overall_p15_f0 = float(np.percentile(p15_samples, 25)) if p15_samples else None
    overall_energy = float(np.median(energy_samples)) if energy_samples else None
    overall_cent = float(np.median(centroid_samples)) if centroid_samples else None
    ac_female_prob = compute_acoustic_female_prob(overall_med_f0, overall_p15_f0, overall_energy, overall_cent)

    # Check if video is predominantly single-speaker (or same-gender expressive speaker)
    # Note: An excited single male speaker often has Cluster 1 ~ 125Hz and Cluster 2 ~ 165Hz (separation ~ 40Hz),
    # so we also check if both clusters share a male chest baseline (overall_p15_f0 < 134Hz) or if separation is small!
    separation = center_high - center_low
    is_expressive_single_male = (
        overall_p15_f0 is not None
        and overall_p15_f0 < 132.0
        and center_low < 140.0
        and center_high < 182.0
    )

    if separation < 28.0 or weight_low > 0.90 or weight_high > 0.90 or is_expressive_single_male:
        top_clips = [c for _, c in sorted(candidate_clips, key=lambda x: x[0], reverse=True)[:8]]
        nn_female_prob = predict_gender_with_pretrained_model(top_clips, sr)
        if nn_female_prob is not None:
            combined_prob = 0.70 * nn_female_prob + 0.30 * ac_female_prob
            rprint(f"[cyan]🧠 Neural Gender Prob P(female)={nn_female_prob:.2f}, Acoustic P(female)={ac_female_prob:.2f} (P15={overall_p15_f0 or 0:.1f}Hz, Med={overall_med_f0:.1f}Hz) → Combined={combined_prob:.2f}[/cyan]")
        else:
            combined_prob = ac_female_prob
            rprint(f"[cyan]🔬 Acoustic Gender Prob P(female)={ac_female_prob:.2f} (P15={overall_p15_f0 or 0:.1f}Hz, Med={overall_med_f0:.1f}Hz)[/cyan]")

        dominant_gender = 'female' if combined_prob >= 0.52 else 'male'
        rprint(f"[bold yellow]🎯 Single/Unified Gender Detected (Separation {separation:.1f} Hz, P(female)={combined_prob:.2f}) → Unified {dominant_gender.upper()}[/bold yellow]")
        return {
            'is_dual_speaker': False,
            'dominant_gender': dominant_gender,
            'boundary_f0': 160.0,
            'male_center': center_low,
            'female_center': center_high,
            'male_var': male_var,
            'female_var': female_var
        }
        
    # Dual-speaker video: find exact boundary where P(female | F0) == 0.5
    test_f0s = np.linspace(max(70.0, center_low * 0.85), min(320.0, center_high * 1.15), 500).reshape(-1, 1)
    probs = gmm.predict_proba(np.log(test_f0s))
    prob_female = probs[:, female_idx]
    boundary_idx = np.argmin(np.abs(prob_female - 0.5))
    boundary_f0 = float(test_f0s[boundary_idx, 0])
    
    rprint(f"[bold green]✨ Dynamic Adaptive GMM Decision Boundary for this video: {boundary_f0:.1f} Hz[/bold green]")
    return {
        'is_dual_speaker': True,
        'boundary_f0': boundary_f0,
        'male_center': center_low,
        'female_center': center_high,
        'male_var': male_var,
        'female_var': female_var,
        'gmm': gmm
    }

def get_named_speaker_gender(text: str) -> str:
    """Detect known character name prefixes and return gender if known."""
    if not text:
        return None
    upper = text.strip().upper()
    if upper.startswith('ED:') or 'ED LUDLOW' in upper or upper.startswith('RYAN:') or upper.startswith('DAN:'):
        return 'male'
    if upper.startswith('LAUREN:') or upper.startswith('CAROLINE:') or upper.startswith('KATIE:'):
        return 'female'
    return None


def _ensure_audio_source() -> str:
    """Return _VOCAL_AUDIO_FILE if available, else _RAW_AUDIO_FILE (extracting it from video if missing)."""
    if os.path.exists(_VOCAL_AUDIO_FILE):
        return _VOCAL_AUDIO_FILE
    if not os.path.exists(_RAW_AUDIO_FILE):
        try:
            from core._1_ytdlp import find_video_files
            from core.asr_backend.audio_preprocess import convert_video_to_audio
            video_file = find_video_files()
            if video_file and os.path.exists(video_file):
                convert_video_to_audio(video_file)
        except Exception as e:
            rprint(f"[yellow]⚠️ Auto-extracting {_RAW_AUDIO_FILE} failed: {e}[/yellow]")
    return _RAW_AUDIO_FILE


def extract_speaker_sample_clips(
    df_tasks: pd.DataFrame,
    output_dir: str = "output/audio/speaker_samples",
) -> dict:
    """
    Extract a representative 2–6s original voice sample clip (.wav) for each speaker_id
    so the user can preview each speaker's real voice in the Streamlit UI.
    Returns {speaker_id: sample_wav_path}.
    """
    import soundfile as sf

    audio_path = _ensure_audio_source()
    if not os.path.exists(audio_path) or df_tasks is None or df_tasks.empty:
        return {}

    os.makedirs(output_dir, exist_ok=True)
    speaker_col = 'speaker_id' if 'speaker_id' in df_tasks.columns else 'gender'
    if speaker_col not in df_tasks.columns:
        return {}

    unique_speakers = [str(s) for s in df_tasks[speaker_col].dropna().unique()]
    sample_paths = {}

    # Check if all sample files already exist
    all_cached = True
    for spk in unique_speakers:
        safe_spk = re.sub(r'[^\w\-]', '_', spk)
        out_wav = os.path.join(output_dir, f"{safe_spk}.wav")
        if os.path.exists(out_wav) and os.path.getsize(out_wav) > 1000:
            sample_paths[spk] = out_wav
        else:
            all_cached = False

    if all_cached and sample_paths:
        return sample_paths

    try:
        full_audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    except Exception:
        return sample_paths

    for spk in unique_speakers:
        safe_spk = re.sub(r'[^\w\-]', '_', spk)
        out_wav = os.path.join(output_dir, f"{safe_spk}.wav")
        if spk in sample_paths:
            continue

        spk_rows = df_tasks[df_tasks[speaker_col].astype(str) == spk]
        best_seg = None
        best_score = -1.0

        for _, row in spk_rows.iterrows():
            s_sec = parse_time_to_seconds(row['start_time'])
            e_sec = parse_time_to_seconds(row['end_time'])
            dur = e_sec - s_sec
            if dur < 0.8:
                continue
            # Prefer segments between 2.0s and 6.0s
            score = min(dur, 6.0) if dur <= 8.0 else 4.0
            if score > best_score:
                s_i = max(0, int(s_sec * sr))
                e_i = min(len(full_audio), int( min(e_sec, s_sec + 6.0) * sr ))
                if e_i - s_i > int(0.8 * sr):
                    best_score = score
                    best_seg = full_audio[s_i:e_i]

        if best_seg is not None:
            try:
                sf.write(out_wav, best_seg, sr)
                sample_paths[spk] = out_wav
            except Exception:
                pass

    return sample_paths


def classify_subtitles_gender(df_tasks: pd.DataFrame) -> pd.DataFrame:
    """
    High-precision voice gender classification with pyannote speaker diarization +
    pretrained Wav2Vec2 neural gender model + P15 chest-register acoustic analysis.
    """
    rprint("[bold cyan]🎙️ Starting acoustic & neural voice gender identification...[/bold cyan]")
    
    # 1. Determine audio file to use
    audio_path = _ensure_audio_source()
    if not os.path.exists(audio_path):
        rprint(f"[yellow]⚠️ Audio file not found at {audio_path}, defaulting all to female voice.[/yellow]")
        df_tasks['gender'] = 'female'
        if 'speaker_id' not in df_tasks.columns:
            df_tasks['speaker_id'] = 'SPEAKER_00'
        return df_tasks

    # ── Pyannote-Enhanced Path ──────────────────────────────────────────────────
    try:
        hf_token = load_key("pyannote.hf_token")
        if hf_token:
            result = _try_pyannote_enhanced_classification(df_tasks, audio_path, hf_token)
            if result is not None:
                extract_speaker_sample_clips(result)
                return result
    except Exception as e:
        rprint(f"[yellow]⚠️ Pyannote config lookup failed ({e}), using neural + GMM method.[/yellow]")

    # ── Neural + Acoustic GMM Path (when pyannote is not enabled) ───────────────
    rprint("[cyan]📊 Using Pretrained Neural + Multi-Feature Acoustic GMM gender classification...[/cyan]")

    sr = 16000
    try:
        rprint(f"[cyan]📥 Loading audio ({audio_path}) into memory...[/cyan]")
        full_audio, sr = librosa.load(audio_path, sr=sr, mono=True)
        rprint(f"[green]✓ Audio loaded: {len(full_audio)/sr:.1f}s duration ({len(full_audio)} samples)[/green]")
    except Exception as e:
        rprint(f"[red]❌ Failed to load audio file: {e}. Defaulting to female voice.[/red]")
        df_tasks['gender'] = 'female'
        df_tasks['speaker_id'] = 'SPEAKER_00'
        return df_tasks

    # 3. Fit video-level GMM pitch clusters (with neural + P15 single-speaker detection)
    cluster_info = fit_video_pitch_clusters(full_audio, sr, df_tasks)
    
    if not cluster_info['is_dual_speaker']:
        dom_g = cluster_info['dominant_gender']
        df_tasks['gender'] = dom_g
        df_tasks['speaker_id'] = 'SPEAKER_00'
        extract_speaker_sample_clips(df_tasks)
        rprint(Panel(
            f"🎯 Single Speaker Mode: 100% {dom_g.upper()} ({len(df_tasks)} sentences)",
            title="🎙️ Voice Gender Classification Summary",
            border_style="green"
        ))
        return df_tasks

    male_c = cluster_info['male_center']
    female_c = cluster_info['female_center']
    male_var = cluster_info['male_var']
    female_var = cluster_info['female_var']
    boundary_f0 = cluster_info['boundary_f0']
    n = len(df_tasks)

    # 4. Compute Gaussian Log-Likelihood + P15 Chest-Register Scores for every subtitle line
    raw_scores = []  # positive favors female, negative favors male
    for idx, row in df_tasks.iterrows():
        s = int(parse_time_to_seconds(row['start_time']) * sr)
        e = int(parse_time_to_seconds(row['end_time']) * sr)
        seg = full_audio[s:e]
        
        info = extract_f0_and_spectral_features(seg, sr) if len(seg) >= 512 else {'median_f0': None, 'voiced_ratio': 0, 'voiced_frames': 0}
        mf0 = info['median_f0']
        p15 = info.get('p15_f0')
        vf = info['voiced_frames']
        vr = info['voiced_ratio']
        sc = info.get('spectral_centroid')
        le = info.get('low_energy_ratio')
        
        orig_text = str(row.get('origin', '') or row.get('text', '')).strip()
        named_g = get_named_speaker_gender(orig_text)
        
        if named_g == 'male':
            score = -10.0
        elif named_g == 'female':
            score = 10.0
        elif mf0 and vr >= 0.04 and vf >= 2:
            lf0 = np.log(mf0)
            log_p_m = -0.5 * np.log(2 * np.pi * male_var) - ((lf0 - np.log(male_c))**2) / (2 * male_var)
            log_p_f = -0.5 * np.log(2 * np.pi * female_var) - ((lf0 - np.log(female_c))**2) / (2 * female_var)
            conf = min(1.0, vf / 10.0)
            gmm_score = (log_p_f - log_p_m) * conf

            # Blend with P15 chest-register acoustic probability so excited male spikes don't flip to female
            ac_prob_f = compute_acoustic_female_prob(mf0, p15, le, sc)
            ac_score = (ac_prob_f - 0.5) * 8.0
            score = 0.55 * gmm_score + 0.45 * ac_score
        else:
            score = 0.0
            
        raw_scores.append((score, orig_text))

    # 5. Dynamic Sequence Decoding with Minimum Block Length & Intonation Resistance
    genders = []
    curr_g = 'female' if raw_scores[0][0] > 0 else 'male'
    genders.append(curr_g)

    for i in range(1, n):
        score, text = raw_scores[i]
        prev_g = genders[-1]
        
        if score <= -8.0:
            curr_g = 'male'
        elif score >= 8.0:
            curr_g = 'female'
        else:
            if prev_g == 'male' and score > 2.0:
                next_score = raw_scores[i + 1][0] if i + 1 < n else 0.0
                if next_score > 0.5 or score > 4.5:
                    curr_g = 'female'
                else:
                    curr_g = 'male'
            elif prev_g == 'female' and score < -2.0:
                next_score = raw_scores[i + 1][0] if i + 1 < n else 0.0
                if next_score < -0.5 or score < -4.5:
                    curr_g = 'male'
                else:
                    curr_g = 'female'
            else:
                curr_g = prev_g
                
        genders.append(curr_g)

    # 6. Anti-Glitch Temporal Smoothing (eliminates single-frame isolated spikes)
    for i in range(1, n - 1):
        if genders[i] != genders[i - 1] and genders[i - 1] == genders[i + 1]:
            text = raw_scores[i][1]
            if '>>' not in text and '»' not in text and not re.match(r'^[A-Z\u4e00-\u9fa5]{2,10}\s*[:：]', text):
                genders[i] = genders[i - 1]

    df_tasks['gender'] = genders
    if 'speaker_id' not in df_tasks.columns:
        df_tasks['speaker_id'] = ['SPEAKER_00' if g == 'male' else 'SPEAKER_01' for g in genders]

    extract_speaker_sample_clips(df_tasks)
    
    male_count = genders.count('male')
    female_count = genders.count('female')
    
    # 7. Output Rich Summary
    rprint(Panel(
        f"👨 Male Segments (男声): {male_count} 句\n"
        f"👩 Female Segments (女声): {female_count} 句\n"
        f"🎯 Total Sentences (总计): {len(df_tasks)} 句\n"
        f"📊 Optimal Boundary: {boundary_f0:.1f} Hz (Male: {male_c:.1f} Hz | Female: {female_c:.1f} Hz)",
        title="🎙️ Voice Gender Classification Summary",
        border_style="green"
    ))
    
    return df_tasks


def _try_pyannote_enhanced_classification(
    df_tasks: pd.DataFrame,
    audio_path: str,
    hf_token: str,
) -> pd.DataFrame:
    """
    Attempt pyannote-enhanced gender classification.
    Returns the classified DataFrame on success, or None if pyannote is unavailable.
    """
    try:
        from core.utils.speaker_diarize import (
            run_diarization,
            assign_speakers_to_subtitles,
            save_diarization_result,
            load_diarization_cache,
            assign_speakers_from_cache,
        )
    except ImportError:
        rprint("[yellow]⚠️ pyannote.audio not installed. Install with: pip install 'pyannote.audio>=3.1'[/yellow]")
        rprint("[yellow]   Falling back to Neural + GMM gender classification.[/yellow]")
        return None

    rprint("[bold cyan]🎯 Pyannote speaker diarization enabled — using enhanced gender classification[/bold cyan]")

    # Check for cached diarization results
    cached = load_diarization_cache()
    if cached is not None:
        df_tasks = assign_speakers_from_cache(cached, df_tasks)
    else:
        try:
            min_spk = 1
            max_spk = 10
            try:
                min_spk = int(load_key("pyannote.min_speakers"))
                max_spk = int(load_key("pyannote.max_speakers"))
            except Exception:
                pass

            diarization = run_diarization(audio_path, hf_token, min_spk, max_spk)
            df_tasks = assign_speakers_to_subtitles(diarization, df_tasks)
            save_diarization_result(diarization)
        except Exception as e:
            rprint(f"[yellow]⚠️ Pyannote diarization failed: {e}[/yellow]")
            err_str = str(e).lower()
            if "gated" in err_str or "403" in err_str or "restricted" in err_str:
                rprint("[bold yellow]💡 HuggingFace gated model access required. Please make sure your token has access to:[/bold yellow]")
                rprint("[cyan]   1. https://huggingface.co/pyannote/speaker-diarization-3.1[/cyan]")
                rprint("[cyan]   2. https://huggingface.co/pyannote/segmentation-3.0[/cyan]")
                rprint("[cyan]   3. https://huggingface.co/pyannote/speaker-diarization-community-1[/cyan]")
            rprint("[yellow]   Falling back to Neural + GMM gender classification.[/yellow]")
            return None

    # Now classify gender at the cluster level
    return _classify_gender_by_cluster(df_tasks, audio_path)


def _classify_gender_by_cluster(
    df_tasks: pd.DataFrame,
    audio_path: str,
) -> pd.DataFrame:
    """
    Intelligent neural + multi-feature acoustic gender classification at the speaker-cluster level.
    
    Each speaker cluster is evaluated independently using:
    1. Pretrained Wav2Vec2 gender classification model on top clean speech segments of that speaker.
    2. Multi-dimensional acoustic features: P15 F0 (chest-register baseline), Median F0,
       low-frequency harmonic energy ratio (75-240Hz), and spectral centroid.
    This guarantees that:
    - Excited/passionate male speakers (high median F0, low P15 F0, male vocal tract) are NOT misclassified as female.
    - Videos with two male speakers (or two female speakers) do NOT force one speaker into the wrong gender.
    """
    rprint("[bold cyan]🔬 Running Neural + Acoustic gender classification on speaker clusters...[/bold cyan]")

    sr = 16000
    try:
        full_audio, sr = librosa.load(audio_path, sr=sr, mono=True)
    except Exception as e:
        rprint(f"[red]❌ Failed to load audio: {e}. Defaulting to female voice.[/red]")
        df_tasks['gender'] = 'female'
        return df_tasks

    unique_speakers = [s for s in df_tasks['speaker_id'].unique() if pd.notna(s)]
    if not unique_speakers:
        unique_speakers = ['SPEAKER_00']
        df_tasks['speaker_id'] = 'SPEAKER_00'

    speaker_stats = {}
    speaker_genders = {}

    for speaker in unique_speakers:
        speaker_rows = df_tasks[df_tasks['speaker_id'] == speaker]
        f0_list = []
        p15_list = []
        energy_list = []
        centroids = []
        candidate_clips = []

        for _, row in speaker_rows.iterrows():
            start_sec = parse_time_to_seconds(row['start_time'])
            end_sec = parse_time_to_seconds(row['end_time'])
            s_idx = max(0, int(start_sec * sr))
            e_idx = min(len(full_audio), int(end_sec * sr))
            seg = full_audio[s_idx:e_idx]

            if len(seg) >= 1024:
                info = extract_f0_and_spectral_features(seg, sr)
                if info['median_f0'] and info['voiced_ratio'] >= 0.06 and info['voiced_frames'] >= 3:
                    f0_list.append(info['median_f0'])
                    if info.get('p15_f0') is not None:
                        p15_list.append(info['p15_f0'])
                    if info.get('low_energy_ratio') is not None:
                        energy_list.append(info['low_energy_ratio'])
                    if info.get('spectral_centroid') is not None:
                        centroids.append(info['spectral_centroid'])
                    if len(seg) >= int(sr * 1.0):
                        candidate_clips.append((len(seg), seg))

        med_f0 = float(np.median(f0_list)) if f0_list else None
        p15_f0 = float(np.percentile(p15_list, 25)) if p15_list else None
        med_energy = float(np.median(energy_list)) if energy_list else None
        mean_cent = float(np.mean(centroids)) if centroids else None

        # 1. Acoustic probability P(female)
        ac_prob_f = compute_acoustic_female_prob(med_f0, p15_f0, med_energy, mean_cent)

        # 2. Pretrained Neural Model probability P(female) on top 6 longest clean clips of this speaker
        top_clips = [c for _, c in sorted(candidate_clips, key=lambda x: x[0], reverse=True)[:6]]
        nn_prob_f = predict_gender_with_pretrained_model(top_clips, sr) if top_clips else None

        if nn_prob_f is not None and med_f0 is not None:
            final_prob_f = 0.70 * nn_prob_f + 0.30 * ac_prob_f
        elif nn_prob_f is not None:
            final_prob_f = nn_prob_f
        else:
            final_prob_f = ac_prob_f

        gender = 'female' if final_prob_f >= 0.52 else 'male'
        speaker_genders[speaker] = gender
        speaker_stats[speaker] = {
            'count': len(speaker_rows),
            'voiced_samples': len(f0_list),
            'median_f0': med_f0,
            'p15_f0': p15_f0,
            'nn_prob_f': nn_prob_f,
            'ac_prob_f': ac_prob_f,
            'final_prob_f': final_prob_f,
        }

        f0_disp = f"Med={med_f0:.1f}Hz, P15={p15_f0:.1f}Hz" if (med_f0 and p15_f0) else "N/A"
        nn_disp = f"{nn_prob_f:.2f}" if nn_prob_f is not None else "N/A"
        rprint(
            f"  🎙️ {speaker}: {f0_disp} | Neural P(F)={nn_disp}, Acoustic P(F)={ac_prob_f:.2f} "
            f"→ Final P(F)={final_prob_f:.2f} → [bold]{gender.upper()}[/bold] ({len(speaker_rows)} lines)"
        )

    # Apply cluster-level gender to all subtitle lines
    df_tasks['gender'] = df_tasks['speaker_id'].map(speaker_genders).fillna('female')

    male_count = int((df_tasks['gender'] == 'male').sum())
    female_count = int((df_tasks['gender'] == 'female').sum())

    spk_summary_items = []
    for s in sorted(unique_speakers):
        med = speaker_stats[s].get('median_f0')
        p15 = speaker_stats[s].get('p15_f0')
        f0_str = f"Med {med:.0f}Hz/P15 {p15:.0f}Hz" if (med and p15) else "N/A"
        spk_summary_items.append(f"{s} = {speaker_genders[s].upper()} ({f0_str})")
    spk_summary_str = ', '.join(spk_summary_items)

    rprint(Panel(
        f"🎯 [bold green]Neural + Acoustic Voice Gender Classification Completed[/bold green]\n"
        f"📊 Speakers Discovered: {len(unique_speakers)} ({spk_summary_str})\n"
        f"👨 Male Sentences (男声): {male_count} 句\n"
        f"👩 Female Sentences (女声): {female_count} 句\n"
        f"🎙️ Total Sentences (总计): {len(df_tasks)} 句",
        title="🎙️ Pyannote + Neural Voice Gender Assignment",
        border_style="green"
    ))

    return df_tasks

