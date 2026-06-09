import pandas as pd
import json
import concurrent.futures
from core.translate_lines import translate_lines
from core._4_1_summarize import search_things_to_note_in_prompt
from core._8_1_audio_task import check_len_then_trim, batch_check_len_then_trim
from core._6_gen_sub import align_timestamp
from core.utils import *
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from difflib import SequenceMatcher
from core.utils.models import *
console = Console()

# Function to split text into chunks
def split_chunks_by_chars(chunk_size, max_i): 
    """Split text into chunks based on character count, return a list of multi-line text chunks"""
    with open(_3_2_SPLIT_BY_MEANING, "r", encoding="utf-8") as file:
        sentences = file.read().strip().split('\n')

    chunks = []
    chunk = ''
    sentence_count = 0
    for sentence in sentences:
        if len(chunk) + len(sentence + '\n') > chunk_size or sentence_count == max_i:
            chunks.append(chunk.strip())
            chunk = sentence + '\n'
            sentence_count = 1
        else:
            chunk += sentence + '\n'
            sentence_count += 1
    chunks.append(chunk.strip())
    return chunks

# Get context from surrounding chunks
def get_previous_content(chunks, chunk_index):
    return None if chunk_index == 0 else chunks[chunk_index - 1].split('\n')[-3:] # Get last 3 lines
def get_after_content(chunks, chunk_index):
    return None if chunk_index == len(chunks) - 1 else chunks[chunk_index + 1].split('\n')[:2] # Get first 2 lines

# 🔍 Translate a single chunk
def translate_chunk(chunk, chunks, theme_prompt, i):
    things_to_note_prompt = search_things_to_note_in_prompt(chunk)
    previous_content_prompt = get_previous_content(chunks, i)
    after_content_prompt = get_after_content(chunks, i)
    translation, english_result, corrected_sources = translate_lines(chunk, previous_content_prompt, after_content_prompt, things_to_note_prompt, theme_prompt, i)
    return i, english_result, translation, corrected_sources

# Add similarity calculation function
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def _split_chunk_into_sub(chunk, chunk_size=600, max_i=10):
    """Split a large chunk string into smaller sub-chunk strings."""
    lines = [l for l in chunk.split('\n') if l]
    sub_chunks, current, char_count = [], [], 0
    for line in lines:
        if char_count + len(line) > chunk_size or len(current) >= max_i:
            if current:
                sub_chunks.append('\n'.join(current))
            current, char_count = [line], len(line)
        else:
            current.append(line)
            char_count += len(line)
    if current:
        sub_chunks.append('\n'.join(current))
    return sub_chunks

def _translate_with_fallback(chunk, all_chunks, theme_prompt, i):
    """Translate chunk; on failure re-split into standard sub-chunks and translate each."""
    try:
        return translate_chunk(chunk, all_chunks, theme_prompt, i)
    except Exception as e:
        err_str = str(e)
        # 🛡️ 对根本无法通过重试/切分解决的登录和认证硬伤错误，直接拦截并向上抛出，杜绝卡死
        if "尚未登录" in err_str or "agy login" in err_str or "API key is not set" in err_str or "Authentication required" in err_str or "Antigravity CLI call failed" in err_str or "after re-auth" in err_str:
            raise e

        console.print(f"[yellow]⚠️ Chunk {i} failed ({e}). Re-splitting into sub-chunks...[/yellow]")
        sub_chunks = _split_chunk_into_sub(chunk, chunk_size=600, max_i=10)
        src_all, trans_all, corrected_all = [], [], []
        for sc in sub_chunks:
            _, sc_src, sc_trans, sc_corrected = translate_chunk(sc, [sc], theme_prompt, i)
            src_all.extend([l for l in sc_src.split('\n') if l])
            trans_all.extend([l for l in sc_trans.split('\n') if l])
            corrected_all.extend([l for l in sc_corrected.split('\n') if l])
        return i, '\n'.join(src_all), '\n'.join(trans_all), '\n'.join(corrected_all)

# 🚀 Main function to translate all chunks
@check_file_exists(_4_2_TRANSLATION)
def translate_all():
    console.print("[bold green]Start Translating All...[/bold green]")
    with open(_4_1_TERMINOLOGY, 'r', encoding='utf-8') as file:
        theme_prompt = json.load(file).get('theme')

    efficiency_mode = load_key("efficiency_mode")
    if efficiency_mode:
        # 💡 读取用户在效率模式下设置的翻译分批最大行数，默认 15
        batch_size = load_key("batch_translate_size")
        if batch_size is None:
            batch_size = 15
        elif batch_size == 0:
            batch_size = 40 # 若用户设置为 0（无限），出于安全与超时防护我们将其限制在 40 句
            
        # 成比例地缩放单批字符上限，最大程度预防超时或截断
        chunk_char_limit = batch_size * 100
        
        chunks = split_chunks_by_chars(chunk_size=chunk_char_limit, max_i=batch_size)
        console.print(f"[cyan]⚡ Efficiency mode: {len(chunks)} large chunk(s) (chunk size: {batch_size}, char limit: {chunk_char_limit})[/cyan]")
        translate_fn = lambda chunk, i: _translate_with_fallback(chunk, chunks, theme_prompt, i)
    else:
        chunks = split_chunks_by_chars(chunk_size=600, max_i=10)
        translate_fn = lambda chunk, i: translate_chunk(chunk, chunks, theme_prompt, i)

    from core.utils.progress_utils import get_progress, update_st_progress
    progress = get_progress()
    is_internal = not progress.live.is_started
    
    if is_internal:
        progress.start()
        
    task_desc = "📝 正在翻译字幕块..."
    task = progress.add_task(task_desc, total=len(chunks))
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=load_key("max_workers"))
    try:
        futures = [executor.submit(translate_fn, chunk, i) for i, chunk in enumerate(chunks)]
        results = []
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                results.append(future.result())
            except Exception as e:
                for f in futures:
                    f.cancel()
                executor.shutdown(wait=False)
                raise e
            progress.update(task, advance=1)
            update_st_progress(i + 1, len(chunks), task_desc)
    finally:
        executor.shutdown(wait=False)
        if is_internal:
            try:
                progress.stop()
            except Exception:
                pass
        else:
            try:
                progress.remove_task(task)
            except Exception:
                pass

    results.sort(key=lambda x: x[0])  # Sort results based on original order

    # 💾 Save results to lists and Excel file
    src_text, trans_text, corrected_text = [], [], []
    for i, chunk in enumerate(chunks):
        chunk_lines = chunk.split('\n')
        src_text.extend(chunk_lines)

        # Calculate similarity between current chunk and translation results
        chunk_text = ''.join(chunk_lines).lower()
        matching_results = [(r, similar(''.join(r[1].split('\n')).lower(), chunk_text))
                          for r in results]
        best_match = max(matching_results, key=lambda x: x[1])

        # Check similarity and handle exceptions
        if best_match[1] < 0.9:
            console.print(f"[yellow]Warning: No matching translation found for chunk {i}[/yellow]")
            raise ValueError(f"Translation matching failed (chunk {i})")
        elif best_match[1] < 1.0:
            console.print(f"[yellow]Warning: Similar match found (chunk {i}, similarity: {best_match[1]:.3f})[/yellow]")

        trans_text.extend(best_match[0][2].split('\n'))
        # Collect corrected sources (index 3); fallback to original chunk lines if unavailable or mismatched
        corrected_lines = best_match[0][3].split('\n') if len(best_match[0]) > 3 else chunk_lines
        corrected_text.extend(corrected_lines if len(corrected_lines) == len(chunk_lines) else chunk_lines)

    # Trim long translation text
    df_text = pd.read_excel(_2_CLEANED_CHUNKS)
    df_text['text'] = df_text['text'].str.strip('"').str.strip()
    df_translate = pd.DataFrame({'Source': src_text, 'Translation': trans_text, 'Corrected_Source': corrected_text})
    subtitle_output_configs = [('trans_subs_for_audio.srt', ['Translation'])]
    df_time = align_timestamp(df_text, df_translate, subtitle_output_configs, output_dir=None, for_display=False)
    console.print(df_time)
    # apply check_len_then_trim to df_time['Translation'], only when duration > MIN_TRIM_DURATION.
    min_trim = load_key("min_trim_duration")
    if efficiency_mode:
        trim_idxs = [i for i, d in enumerate(df_time['duration']) if d > min_trim]
        if trim_idxs:
            sub_texts = [df_time['Translation'].iloc[i] for i in trim_idxs]
            sub_durs = [df_time['duration'].iloc[i] for i in trim_idxs]
            trimmed = batch_check_len_then_trim(sub_texts, sub_durs)
            for local_i, orig_i in enumerate(trim_idxs):
                df_time.at[df_time.index[orig_i], 'Translation'] = trimmed[local_i]
    else:
        df_time['Translation'] = df_time.apply(
            lambda x: check_len_then_trim(x['Translation'], x['duration']) if x['duration'] > min_trim else x['Translation'], axis=1
        )
    console.print(df_time)
    
    df_time.to_excel(_4_2_TRANSLATION, index=False)
    console.print("[bold green]✅ Translation completed and results saved.[/bold green]")

if __name__ == '__main__':
    translate_all()