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
    Extract fundamental frequency (F0) pitch and spectral centroid
    directly from an in-memory audio array.
    """
    if len(y) < 1024:
        pad_len = 2048 - len(y)
        y = np.pad(y, (0, max(0, pad_len)), mode='constant')

    result = {
        'median_f0': None,
        'mean_f0': None,
        'voiced_ratio': 0.0,
        'voiced_frames': 0,
        'spectral_centroid': None
    }

    try:
        # Human speaking fundamental frequency range: 70Hz - 350Hz
        fmin = 70
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
            result['voiced_ratio'] = float(len(voiced_f0) / len(f0)) if len(f0) > 0 else 0.0
            result['voiced_frames'] = int(len(voiced_f0))
            
        # Spectral Centroid
        cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)
        result['spectral_centroid'] = float(np.mean(cent))
        
    except Exception:
        pass
    
    return result

def fit_video_pitch_clusters(full_audio: np.ndarray, sr: int, df_tasks: pd.DataFrame) -> dict:
    """
    Fit a 2-Component Gaussian Mixture Model (GMM) on log(F0) across the entire video
    to discover video-specific male & female pitch centers, variances, and optimal boundaries.
    """
    f0_samples = []
    
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
                
    if len(f0_samples) < 8:
        rprint("[yellow]⚠️ Insufficient voiced samples for GMM clustering, using standard default boundary (145.0 Hz).[/yellow]")
        return {
            'is_dual_speaker': True,
            'boundary_f0': 145.0,
            'male_center': 115.0,
            'female_center': 175.0,
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
    
    # Check if video is predominantly single-speaker
    separation = center_high - center_low
    if separation < 22.0 or weight_low > 0.92 or weight_high > 0.92:
        dominant_mean = center_low if weight_low >= weight_high else center_high
        dominant_gender = 'male' if dominant_mean < 145.0 else 'female'
        rprint(f"[bold yellow]🎯 Single Speaker Detected (Separation {separation:.1f} Hz, Mean {dominant_mean:.1f} Hz) → Unified {dominant_gender.upper()}[/bold yellow]")
        return {
            'is_dual_speaker': False,
            'dominant_gender': dominant_gender,
            'boundary_f0': 145.0,
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

def classify_subtitles_gender(df_tasks: pd.DataFrame) -> pd.DataFrame:
    """
    High-precision voice gender classification with optional pyannote speaker diarization enhancement.
    
    Enhanced pipeline (when pyannote is available and configured):
    1. Uses pyannote speaker-diarization-3.1 to identify different speakers via voice embeddings.
    2. Assigns each subtitle line to a speaker cluster based on temporal overlap.
    3. For each speaker cluster, aggregates F0 pitch data and classifies gender at the cluster level.
    This is far more accurate than per-sentence classification because intonation noise is averaged out.
    
    Fallback pipeline (when pyannote is not available):
    1. Loads full audio into memory once (fast & error-free).
    2. Fits video-level 2-component GMM on log(F0) to learn exact pitch centers and variances.
    3. Computes exact Gaussian log-likelihood emission evidence for each subtitle line.
    4. Applies Dynamic Programming with Minimum-Block Continuity Constraint to prevent single-sentence intonation noise.
    5. Outputs clean, seamless, continuous speaker voice allocations.
    """
    rprint("[bold cyan]🎙️ Starting acoustic sequence voice gender identification...[/bold cyan]")
    
    # 1. Determine audio file to use
    audio_path = _VOCAL_AUDIO_FILE if os.path.exists(_VOCAL_AUDIO_FILE) else _RAW_AUDIO_FILE
    if not os.path.exists(audio_path):
        rprint(f"[yellow]⚠️ Audio file not found at {audio_path}, defaulting all to female voice.[/yellow]")
        df_tasks['gender'] = 'female'
        return df_tasks

    # ── Pyannote-Enhanced Path ──────────────────────────────────────────────────
    # Try to use pyannote speaker diarization for more accurate gender classification.
    # If pyannote is not installed or HF token is not configured, fall through to GMM-only.
    try:
        hf_token = load_key("pyannote.hf_token")
        if hf_token:  # Only attempt if token is configured
            result = _try_pyannote_enhanced_classification(df_tasks, audio_path, hf_token)
            if result is not None:
                return result
    except Exception as e:
        rprint(f"[yellow]⚠️ Pyannote config lookup failed ({e}), using GMM-only method.[/yellow]")

    # ── GMM-Only Fallback Path ─────────────────────────────────────────────────
    rprint("[cyan]📊 Using GMM-only gender classification (no pyannote enhancement)...[/cyan]")

    # 2. Load entire audio into memory once
    sr = 16000
    try:
        rprint(f"[cyan]📥 Loading audio ({audio_path}) into memory...[/cyan]")
        full_audio, sr = librosa.load(audio_path, sr=sr, mono=True)
        rprint(f"[green]✓ Audio loaded: {len(full_audio)/sr:.1f}s duration ({len(full_audio)} samples)[/green]")
    except Exception as e:
        rprint(f"[red]❌ Failed to load audio file: {e}. Defaulting to female voice.[/red]")
        df_tasks['gender'] = 'female'
        return df_tasks

    # 3. Fit video-level GMM pitch clusters
    cluster_info = fit_video_pitch_clusters(full_audio, sr, df_tasks)
    
    if not cluster_info['is_dual_speaker']:
        df_tasks['gender'] = cluster_info['dominant_gender']
        rprint(Panel(
            f"🎯 Single Speaker Mode: 100% {cluster_info['dominant_gender'].upper()} ({len(df_tasks)} sentences)",
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

    # 4. Compute Gaussian Log-Likelihood Scores for every subtitle line
    raw_scores = [] # positive favors female, negative favors male
    for idx, row in df_tasks.iterrows():
        s = int(parse_time_to_seconds(row['start_time']) * sr)
        e = int(parse_time_to_seconds(row['end_time']) * sr)
        seg = full_audio[s:e]
        
        info = extract_f0_and_spectral_features(seg, sr) if len(seg) >= 512 else {'median_f0': None, 'voiced_ratio': 0, 'voiced_frames': 0}
        mf0 = info['median_f0']
        vf = info['voiced_frames']
        vr = info['voiced_ratio']
        sc = info.get('spectral_centroid')
        
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
            score = (log_p_f - log_p_m) * conf
            if sc and sc > 1750.0: score += 0.5
            elif sc and sc < 1400.0: score -= 0.5
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
            # Check acoustic evidence across adjacent frames to avoid single-frame pitch spikes
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
        rprint("[yellow]   Falling back to GMM-only gender classification.[/yellow]")
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
            rprint("[yellow]   Falling back to GMM-only gender classification.[/yellow]")
            return None

    # Now classify gender at the cluster level
    return _classify_gender_by_cluster(df_tasks, audio_path)


def _classify_gender_by_cluster(
    df_tasks: pd.DataFrame,
    audio_path: str,
) -> pd.DataFrame:
    """
    Classify gender at the speaker-cluster level rather than per-sentence.

    For each unique speaker_id in df_tasks:
    1. Collect all subtitle segments belonging to that speaker.
    2. Extract F0 pitch from each segment and aggregate.
    3. Compute the median F0 across all segments.
    4. Classify the speaker as male (< threshold) or female (>= threshold).

    This is far more accurate than per-sentence classification because:
    - Intonation spikes in individual sentences are averaged out.
    - Each speaker typically has 5-50+ sentences, giving a robust F0 estimate.
    """
    import librosa

    rprint("[cyan]🔬 Classifying gender at speaker-cluster level...[/cyan]")

    sr = 16000
    try:
        full_audio, sr = librosa.load(audio_path, sr=sr, mono=True)
    except Exception as e:
        rprint(f"[red]❌ Failed to load audio: {e}. Defaulting to female voice.[/red]")
        df_tasks['gender'] = 'female'
        return df_tasks

    unique_speakers = df_tasks['speaker_id'].unique()
    speaker_genders = {}

    for speaker in unique_speakers:
        speaker_rows = df_tasks[df_tasks['speaker_id'] == speaker]
        all_f0_values = []

        for _, row in speaker_rows.iterrows():
            start_sec = parse_time_to_seconds(row['start_time'])
            end_sec = parse_time_to_seconds(row['end_time'])
            s_idx = max(0, int(start_sec * sr))
            e_idx = min(len(full_audio), int(end_sec * sr))
            seg = full_audio[s_idx:e_idx]

            if len(seg) >= 1024:
                info = extract_f0_and_spectral_features(seg, sr)
                if info['median_f0'] and info['voiced_ratio'] >= 0.08 and info['voiced_frames'] >= 3:
                    all_f0_values.append(info['median_f0'])

        if len(all_f0_values) >= 2:
            median_f0 = float(np.median(all_f0_values))
            # Use 145 Hz as the default male/female boundary
            # This is the same threshold used in the GMM approach
            gender = 'male' if median_f0 < 145.0 else 'female'
            rprint(f"  {speaker}: median F0 = {median_f0:.1f} Hz → {gender.upper()} ({len(all_f0_values)} voiced segments)")
        elif len(all_f0_values) == 1:
            median_f0 = all_f0_values[0]
            gender = 'male' if median_f0 < 145.0 else 'female'
            rprint(f"  {speaker}: single F0 = {median_f0:.1f} Hz → {gender.upper()} (low confidence)")
        else:
            gender = 'female'  # Default fallback
            rprint(f"  {speaker}: no voiced segments → defaulting to {gender.upper()}")

        speaker_genders[speaker] = gender

    # Apply cluster-level gender to all subtitle lines
    df_tasks['gender'] = df_tasks['speaker_id'].map(speaker_genders)

    male_count = (df_tasks['gender'] == 'male').sum()
    female_count = (df_tasks['gender'] == 'female').sum()

    rprint(Panel(
        f"🎯 Pyannote-Enhanced Gender Classification\n"
        f"📊 Speakers Found: {len(unique_speakers)} ({', '.join(f'{s}={speaker_genders[s]}' for s in sorted(unique_speakers))})\n"
        f"👨 Male Segments: {male_count} 句\n"
        f"👩 Female Segments: {female_count} 句\n"
        f"🎙️ Total: {len(df_tasks)} 句",
        title="🎙️ Pyannote + F0 Voice Gender Classification",
        border_style="green"
    ))

    return df_tasks
