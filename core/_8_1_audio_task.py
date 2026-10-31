import datetime
import os
import re
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from core.prompts import get_subtitle_trim_prompt, get_batch_trim_prompt
from core.tts_backend.estimate_duration import init_estimator, estimate_duration
from core.utils.audio_gender import classify_subtitles_gender
from core.utils import *
from core.utils.models import *

console = Console()
speed_factor = load_key("speed_factor")

TRANS_SUBS_FOR_AUDIO_FILE = 'output/audio/trans_subs_for_audio.srt'
SRC_SUBS_FOR_AUDIO_FILE = 'output/audio/src_subs_for_audio.srt'
ESTIMATOR = None

def check_len_then_trim(text, duration):
    global ESTIMATOR
    if ESTIMATOR is None:
        ESTIMATOR = init_estimator()
    estimated_duration = estimate_duration(text, ESTIMATOR) / speed_factor['max']
    
    console.print(f"Subtitle text: {text}, "
                  f"[bold green]Estimated reading duration: {estimated_duration:.2f} seconds[/bold green]")

    if estimated_duration > duration:
        rprint(Panel(f"Estimated reading duration {estimated_duration:.2f} seconds exceeds given duration {duration:.2f} seconds, shortening...", title="Processing", border_style="yellow"))
        original_text = text
        prompt = get_subtitle_trim_prompt(text, duration)
        def valid_trim(response):
            if 'result' not in response:
                return {'status': 'error', 'message': 'No result in response'}
            return {'status': 'success', 'message': ''}
        try:    
            response = ask_gpt(prompt, resp_type='json', log_title='sub_trim', valid_def=valid_trim)
            shortened_text = response['result']
        except Exception:
            rprint("[bold red]🚫 AI refused to answer due to sensitivity, so manually remove punctuation[/bold red]")
            shortened_text = re.sub(r'[,.!?;:，。！？；：]', ' ', text).strip()
        rprint(Panel(f"Subtitle before shortening: {original_text}\nSubtitle after shortening: {shortened_text}", title="Subtitle Shortening Result", border_style="green"))
        return shortened_text
    else:
        return text

def batch_check_len_then_trim(texts, durations):
    """Batch version of check_len_then_trim. Returns list of (possibly shortened) texts."""
    global ESTIMATOR
    if ESTIMATOR is None:
        ESTIMATOR = init_estimator()

    needs_trim = []  # list of (original_index, text, duration)
    results = list(texts)  # start with originals

    for idx, (text, duration) in enumerate(zip(texts, durations)):
        estimated = estimate_duration(text, ESTIMATOR) / speed_factor['max']
        console.print(f"Subtitle text: {text}, "
                      f"[bold green]Estimated reading duration: {estimated:.2f} seconds[/bold green]")
        if estimated > duration:
            rprint(Panel(f"Estimated reading duration {estimated:.2f}s exceeds {duration:.2f}s, queued for shortening...",
                         title="Processing", border_style="yellow"))
            needs_trim.append((idx, text, duration))

    if not needs_trim:
        return results

    items = [(text, dur) for _, text, dur in needs_trim]
    prompt = get_batch_trim_prompt(items)

    def valid_batch_trim(response):
        for i in range(1, len(items) + 1):
            k = str(i)
            if k not in response or 'result' not in response.get(k, {}):
                return {'status': 'error', 'message': f'Missing key {k} or result field'}
        return {'status': 'success', 'message': ''}

    try:
        response = ask_gpt(prompt, resp_type='json', log_title='batch_sub_trim', valid_def=valid_batch_trim)
        for local_i, (orig_idx, original_text, _) in enumerate(needs_trim, 1):
            shortened = response[str(local_i)]['result']
            rprint(Panel(f"Before: {original_text}\nAfter:  {shortened}", title="Subtitle Shortening Result", border_style="green"))
            results[orig_idx] = shortened
    except Exception as e:
        rprint(f"[bold yellow]⚠️ Batch trim failed ({e}), falling back to punctuation removal[/bold yellow]")
        for orig_idx, text, _ in needs_trim:
            results[orig_idx] = re.sub(r'[,.!?;:，。！？；：]', ' ', text).strip()

    return results


def time_diff_seconds(t1, t2, base_date):
    """Calculate the difference in seconds between two time objects"""
    dt1 = datetime.datetime.combine(base_date, t1)
    dt2 = datetime.datetime.combine(base_date, t2)
    return (dt2 - dt1).total_seconds()

def process_srt():
    """Process srt file, generate audio tasks"""
    
    # Defensive check: if subtitle files are missing, try to regenerate them
    if not os.path.exists(TRANS_SUBS_FOR_AUDIO_FILE) or not os.path.exists(SRC_SUBS_FOR_AUDIO_FILE):
        print(f"Subtitle files missing, attempting to regenerate...")
        try:
            from core._6_gen_sub import align_timestamp_main
            align_timestamp_main()
        except Exception as e:
            raise FileNotFoundError(f"Subtitle files missing and regeneration failed: {e}")

    with open(TRANS_SUBS_FOR_AUDIO_FILE, 'r', encoding='utf-8') as file:
        content = file.read()
    
    with open(SRC_SUBS_FOR_AUDIO_FILE, 'r', encoding='utf-8') as src_file:
        src_content = src_file.read()
    
    subtitles = []
    src_subtitles = {}
    
    for block in src_content.strip().split('\n\n'):
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        if len(lines) < 3:
            continue
        
        number = int(lines[0])
        src_text = ' '.join(lines[2:])
        src_subtitles[number] = src_text
    
    for block in content.strip().split('\n\n'):
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        if len(lines) < 3:
            continue
        
        try:
            number = int(lines[0])
            start_time, end_time = lines[1].split(' --> ')
            start_time = datetime.datetime.strptime(start_time, '%H:%M:%S,%f').time()
            end_time = datetime.datetime.strptime(end_time, '%H:%M:%S,%f').time()
            duration = time_diff_seconds(start_time, end_time, datetime.date.today())
            text = ' '.join(lines[2:])
            # Remove content within parentheses (including English and Chinese parentheses) and brackets
            text = re.sub(r'\([^)]*\)', '', text).strip()
            text = re.sub(r'（[^）]*）', '', text).strip()
            text = re.sub(r'\[[^\]]*\]', '', text).strip()
            # Remove '-' character, can continue to add illegal characters that cause errors
            text = text.replace('-', '')

            # Add the original text from src_subs_for_audio.srt
            origin = src_subtitles.get(number, '')

        except ValueError as e:
            rprint(Panel(f"Unable to parse subtitle block '{block}', error: {str(e)}, skipping this subtitle block.", title="Error", border_style="red"))
            continue
        
        subtitles.append({'number': number, 'start_time': start_time, 'end_time': end_time, 'duration': duration, 'text': text, 'origin': origin})
    
    df = pd.DataFrame(subtitles)

    # 1. Run Speaker & Voice Gender Classification FIRST on raw unmerged, unscaled timestamps
    #    so short Q&A turns are never merged across different speakers before classification,
    #    and timestamps match the original audio file (before video_slow_factor scaling).
    df = classify_subtitles_gender(df)
    if 'voice' not in df.columns:
        df['voice'] = ''

    # 2. Merge or extend short subtitles with strict speaker-boundary isolation
    i = 0
    MIN_SUB_DUR = load_key("min_subtitle_duration")
    while i < len(df):
        today = datetime.date.today()
        if df.loc[i, 'duration'] < MIN_SUB_DUR:
            if i < len(df) - 1:
                gap_to_nextStart = time_diff_seconds(df.loc[i, 'start_time'], df.loc[i + 1, 'start_time'], today)
                same_speaker = (
                    str(df.loc[i, 'speaker_id']) == str(df.loc[i + 1, 'speaker_id'])
                    and str(df.loc[i, 'gender']) == str(df.loc[i + 1, 'gender'])
                    and str(df.loc[i, 'voice'] or '') == str(df.loc[i + 1, 'voice'] or '')
                )
                if same_speaker and gap_to_nextStart < MIN_SUB_DUR:
                    rprint(f"[bold yellow]Merging subtitles {i+1} and {i+2} (same speaker: {df.loc[i, 'speaker_id']})[/bold yellow]")
                    df.loc[i, 'text'] += ' ' + df.loc[i + 1, 'text']
                    df.loc[i, 'origin'] += ' ' + df.loc[i + 1, 'origin']
                    df.loc[i, 'end_time'] = df.loc[i + 1, 'end_time']
                    df.loc[i, 'duration'] = time_diff_seconds(df.loc[i, 'start_time'], df.loc[i, 'end_time'], today)
                    df = df.drop(i + 1).reset_index(drop=True)
                    continue

                # If different speaker, extend only up to next speaker's start_time to prevent cross-speaker overlap
                target_dur = MIN_SUB_DUR if same_speaker else min(MIN_SUB_DUR, max(df.loc[i, 'duration'], gap_to_nextStart))
                if target_dur > df.loc[i, 'duration']:
                    rprint(f"[bold blue]Extending subtitle {i+1} duration to {target_dur:.2f} seconds[/bold blue]")
                    df.loc[i, 'end_time'] = (
                        datetime.datetime.combine(today, df.loc[i, 'start_time']) +
                        datetime.timedelta(seconds=target_dur)
                    ).time()
                    df.loc[i, 'duration'] = target_dur
                i += 1
            else:
                rprint(f"[bold red]The last subtitle {i+1} duration is less than {MIN_SUB_DUR} seconds, but not extending[/bold red]")
                i += 1
        else:
            i += 1
    
    df['start_time'] = df['start_time'].apply(lambda x: x.strftime('%H:%M:%S.%f')[:-3])
    df['end_time'] = df['end_time'].apply(lambda x: x.strftime('%H:%M:%S.%f')[:-3])

    # Save original text before trimming so gen_dub_chunks can match against trans.srt.
    df['orig_text'] = df['text'].copy()
    # AI-assisted text compression before TTS generation.
    efficiency_mode = load_key("efficiency_mode")
    from core.utils.progress_utils import get_progress, update_st_progress
    progress = get_progress()
    is_internal = not progress.live.is_started
    if is_internal: progress.start()

    task_desc = "✂️ 正在根据预估语速压缩字幕..."
    if efficiency_mode:
        df['text'] = batch_check_len_then_trim(df['text'].tolist(), df['duration'].tolist())
    else:
        task = progress.add_task(task_desc, total=len(df))
        def process_with_progress(row):
            res = check_len_then_trim(row['text'], row['duration'])
            progress.advance(task)
            update_st_progress(int(progress.tasks[task].completed), len(df), task_desc)
            return res
        df['text'] = df.apply(process_with_progress, axis=1)
        if not is_internal: progress.remove_task(task)

    if is_internal: progress.stop()

    # Apply video slow-down factor: scale all timestamps so that each subtitle window is
    # proportionally wider. Enables the audio to fit with less speed-up.
    # The output video must be slowed by the same factor in _12_dub_to_vid.py.
    try:
        video_slow_factor = load_key("video_slow_factor")
    except KeyError:
        video_slow_factor = 1.0
    if video_slow_factor != 1.0:
        def _scale_time_str(time_str: str) -> str:
            t = datetime.datetime.strptime(time_str, '%H:%M:%S.%f').time()
            total_secs = t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6
            scaled = total_secs / video_slow_factor
            td = datetime.timedelta(seconds=scaled)
            total_us = int(td.total_seconds() * 1_000_000)
            h = total_us // 3_600_000_000
            m = (total_us % 3_600_000_000) // 60_000_000
            s = (total_us % 60_000_000) // 1_000_000
            ms = (total_us % 1_000_000) // 1_000
            return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

        df['start_time'] = df['start_time'].apply(_scale_time_str)
        df['end_time'] = df['end_time'].apply(_scale_time_str)
        df['duration'] = df['duration'] / video_slow_factor
        rprint(f"[bold cyan]🎬 Timestamps scaled by 1/{video_slow_factor:.2f} (video_slow_factor={video_slow_factor})[/bold cyan]")

    return df

def _tc_str_to_sec(tc_val) -> float:
    s = str(tc_val or "").strip().replace(",", ".")
    parts = s.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    try:
        return float(s)
    except Exception:
        return 0.0


def _clean_srt_line_for_audio(text: str) -> str:
    t = re.sub(r"\([^)]*\)|（[^）]*）|\[[^\]]*\]", "", str(text or "")).strip()
    return t.replace("-", "")


def _clean_for_exact_match(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    return re.sub(r"[^\w\s]|[\s]", "", text)


def _parse_srt_blocks_for_sync(path: str):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read().strip()
    if not raw:
        return []
    blocks = []
    for b in re.split(r"\r?\n\r?\n+", raw):
        lines = [ln.strip("\r").strip() for ln in b.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        try:
            idx = int(lines[0])
            tc = lines[1]
            txt = " ".join(lines[2:])
        except ValueError:
            idx = len(blocks) + 1
            tc = lines[0]
            txt = " ".join(lines[1:])
        if "-->" not in tc:
            continue
        s_tc, e_tc = [x.strip() for x in tc.split("-->", 1)]
        blocks.append(
            {
                "index": idx,
                "timecode": tc,
                "start_sec": _tc_str_to_sec(s_tc),
                "end_sec": _tc_str_to_sec(e_tc),
                "text": txt,
            }
        )
    return blocks


def _write_srt_blocks_for_sync(path: str, blocks: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = [f"{b['index']}\n{b['timecode']}\n{b['text']}" for b in blocks]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(out) + "\n")


def _sync_single_audio_srt(display_srt_path: str, audio_srt_path: str) -> int:
    """Propagate text edits from display SRT (e.g. output/trans.srt) into remerged audio SRT."""
    disp = _parse_srt_blocks_for_sync(display_srt_path)
    aud = _parse_srt_blocks_for_sync(audio_srt_path)
    if not disp or not aud:
        return 0

    changed = 0
    if len(disp) == len(aud):
        for d, a in zip(disp, aud):
            if d["text"].strip() != a["text"].strip():
                a["text"] = d["text"].strip()
                changed += 1
    else:
        last_i = 0
        new_texts = []
        for a in aud:
            grp = []
            for i in range(last_i, len(disp)):
                grp.append(disp[i])
                if disp[i]["end_sec"] >= a["end_sec"] - 0.05:
                    last_i = i + 1
                    break
            else:
                last_i = len(disp)
            if not grp:
                return 0
            joined = " ".join(x["text"].strip() for x in grp if x["text"].strip())
            new_texts.append(joined)
        if last_i != len(disp):
            return 0
        for a, nt in zip(aud, new_texts):
            if nt != a["text"].strip():
                a["text"] = nt
                changed += 1

    if changed > 0:
        _write_srt_blocks_for_sync(audio_srt_path, aud)
    return changed


def _apply_orig_diff_to_trimmed_text(old_orig: str, new_orig: str, old_text: str) -> str:
    import difflib

    old_o = str(old_orig or "").strip()
    new_o = str(new_orig or "").strip()
    old_t = str(old_text or "").strip()
    if not old_t or old_t == old_o or not old_o:
        return new_o

    tok_re = re.compile(r"[A-Za-z0-9_]+(?:[.:/-][A-Za-z0-9_]+)*|[\u4e00-\u9fff]|\s+|[^\w\s]", re.UNICODE)
    t_old = tok_re.findall(old_o)
    t_new = tok_re.findall(new_o)
    opcodes = difflib.SequenceMatcher(None, t_old, t_new, autojunk=False).get_opcodes()

    updated = old_t
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            continue
        left_ctx = "".join(t_old[max(0, i1 - 2) : i1])
        right_ctx = "".join(t_old[i2 : min(len(t_old), i2 + 2)])
        old_seg = "".join(t_old[i1:i2])
        new_seg = "".join(t_new[j1:j2])

        if not re.sub(r"[\s,，.。!！?？:：;；、]", "", old_seg) and not re.sub(
            r"[\s,，.。!！?？:：;；、]", "", new_seg
        ):
            continue

        candidates = []
        if left_ctx or right_ctx:
            candidates.append((left_ctx + old_seg + right_ctx, left_ctx + new_seg + right_ctx))
        if left_ctx:
            candidates.append((left_ctx + old_seg, left_ctx + new_seg))
        if right_ctx:
            candidates.append((old_seg + right_ctx, new_seg + right_ctx))
        if old_seg.strip():
            candidates.append((old_seg.strip(), new_seg.strip()))

        for src_pat, dst_pat in candidates:
            if src_pat and src_pat in updated:
                updated = updated.replace(src_pat, dst_pat, 1)
                break
        else:
            return new_o

    return updated


def sync_subtitles_to_audio_files(
    display_trans_srt: str = "output/trans.srt",
    display_src_srt: str = "output/src.srt",
) -> dict:
    """Keep `trans_subs_for_audio.srt`, `src_subs_for_audio.srt`, and existing `tts_tasks.xlsx`
    synchronized with any post-translation edits made to `output/trans.srt` or `output/src.srt`."""
    trans_srt_changed = _sync_single_audio_srt(display_trans_srt, TRANS_SUBS_FOR_AUDIO_FILE)
    src_srt_changed = _sync_single_audio_srt(display_src_srt, SRC_SUBS_FOR_AUDIO_FILE)
    updated_task_rows = 0
    invalidated_files = 0

    if os.path.exists(_8_1_AUDIO_TASK) and os.path.exists(display_trans_srt):
        disp_trans = _parse_srt_blocks_for_sync(display_trans_srt)
        disp_src = _parse_srt_blocks_for_sync(display_src_srt)
        if disp_trans:
            df = pd.read_excel(_8_1_AUDIO_TASK)
            if len(df) > 0:
                try:
                    slow_factor = float(load_key("video_slow_factor") or 1.0)
                except Exception:
                    slow_factor = 1.0

                last_t = 0
                trans_groups = []
                src_groups = []
                has_same_len_src = len(disp_src) == len(disp_trans)

                for idx in range(len(df)):
                    if idx + 1 < len(df):
                        next_unscaled_start = _tc_str_to_sec(df.iloc[idx + 1]["start_time"]) * slow_factor
                        grp_idx = []
                        while last_t < len(disp_trans) and disp_trans[last_t]["start_sec"] < next_unscaled_start - 0.02:
                            grp_idx.append(last_t)
                            last_t += 1
                    else:
                        grp_idx = list(range(last_t, len(disp_trans)))
                        last_t = len(disp_trans)
                    trans_groups.append(grp_idx)
                    src_groups.append(grp_idx if has_same_len_src else [])

                if all(len(g) > 0 for g in trans_groups) and last_t == len(disp_trans):
                    if "orig_text" not in df.columns:
                        df["orig_text"] = df["text"].copy()

                    has_lines_col = "lines" in df.columns
                    affected_numbers = []

                    for idx, grp_idx in enumerate(trans_groups):
                        clean_lines = [
                            _clean_srt_line_for_audio(disp_trans[i]["text"])
                            for i in grp_idx
                            if _clean_srt_line_for_audio(disp_trans[i]["text"])
                        ]
                        new_orig = " ".join(clean_lines)
                        old_orig = "" if pd.isna(df.at[idx, "orig_text"]) else str(df.at[idx, "orig_text"])
                        old_text = "" if pd.isna(df.at[idx, "text"]) else str(df.at[idx, "text"])

                        if _clean_for_exact_match(new_orig) != _clean_for_exact_match(old_orig):
                            new_text = _apply_orig_diff_to_trimmed_text(old_orig, new_orig, old_text)
                            df.at[idx, "orig_text"] = new_orig
                            df.at[idx, "text"] = new_text
                            if has_lines_col and pd.notna(df.at[idx, "lines"]):
                                from core._8_2_dub_chunks import sync_lines_with_text

                                if _clean_for_exact_match(new_text) == _clean_for_exact_match(new_orig):
                                    was_str = isinstance(df.at[idx, "lines"], str)
                                    df.at[idx, "lines"] = repr(clean_lines) if was_str else clean_lines
                                else:
                                    df.at[idx, "lines"] = sync_lines_with_text(clean_lines, new_text)
                                    if isinstance(df.iloc[idx]["lines"], list):
                                        df.at[idx, "lines"] = repr(df.at[idx, "lines"])
                            affected_numbers.append(int(df.at[idx, "number"]))
                            updated_task_rows += 1

                        if has_same_len_src and "origin" in df.columns and src_groups[idx]:
                            new_src_origin = " ".join(
                                disp_src[i]["text"].strip() for i in src_groups[idx] if disp_src[i]["text"].strip()
                            )
                            old_src_origin = "" if pd.isna(df.at[idx, "origin"]) else str(df.at[idx, "origin"])
                            if new_src_origin and new_src_origin != old_src_origin:
                                df.at[idx, "origin"] = new_src_origin
                                if "src_lines" in df.columns and pd.notna(df.at[idx, "src_lines"]):
                                    src_l = [
                                        _clean_srt_line_for_audio(disp_src[i]["text"]) for i in src_groups[idx]
                                    ]
                                    df.at[idx, "src_lines"] = (
                                        repr(src_l) if isinstance(df.at[idx, "src_lines"], str) else src_l
                                    )

                    if updated_task_rows > 0 or trans_srt_changed > 0 or src_srt_changed > 0:
                        df.to_excel(_8_1_AUDIO_TASK, index=False)
                        if affected_numbers:
                            try:
                                from core._10_gen_audio import invalidate_audio_cache_for_numbers

                                invalidated_files = invalidate_audio_cache_for_numbers(affected_numbers)
                            except Exception:
                                pass
                        rprint(
                            f"[green]🔄 Synced {updated_task_rows} edited subtitle task row(s) into {_8_1_AUDIO_TASK}[/green]"
                        )

    return {
        "trans_audio_srt_changed": trans_srt_changed,
        "src_audio_srt_changed": src_srt_changed,
        "updated_task_rows": updated_task_rows,
        "invalidated_cache_files": invalidated_files,
    }


def gen_audio_task_main():
    sync_subtitles_to_audio_files()
    # Auto-invalidate stale task file if subtitle SRT was regenerated with new structure after it
    if os.path.exists(_8_1_AUDIO_TASK) and os.path.exists(TRANS_SUBS_FOR_AUDIO_FILE):
        if os.path.getmtime(TRANS_SUBS_FOR_AUDIO_FILE) > os.path.getmtime(_8_1_AUDIO_TASK) + 0.5:
            os.remove(_8_1_AUDIO_TASK)
            rprint("[yellow]🔄 Subtitle SRT is newer than audio task file, regenerating...[/yellow]")
    if os.path.exists(_8_1_AUDIO_TASK):
        try:
            existing_df = pd.read_excel(_8_1_AUDIO_TASK)
            updated = False
            if 'gender' not in existing_df.columns or 'speaker_id' not in existing_df.columns:
                rprint("[yellow]🔄 Existing audio task file missing speaker/gender columns, classifying and updating...[/yellow]")
                existing_df = classify_subtitles_gender(existing_df)
                updated = True
            if 'voice' not in existing_df.columns:
                existing_df['voice'] = ''
                updated = True
            if updated:
                existing_df.to_excel(_8_1_AUDIO_TASK, index=False)
                return
        except Exception:
            pass
        rprint(f"[yellow]⚠️ File <{_8_1_AUDIO_TASK}> already exists, skip <gen_audio_task_main> step.[/yellow]")
        return
    df = process_srt()
    console.print(df)
    df.to_excel(_8_1_AUDIO_TASK, index=False)
    rprint(Panel(f"Successfully generated {_8_1_AUDIO_TASK}", title="Success", border_style="green"))

if __name__ == '__main__':
    gen_audio_task_main()