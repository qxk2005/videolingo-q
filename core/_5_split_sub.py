import pandas as pd
from typing import List, Tuple
import concurrent.futures

from core._3_2_split_meaning import split_sentence, _apply_split
from core.prompts import get_align_prompt, get_batch_align_prompt, get_batch_split_prompt
from rich.panel import Panel
from rich.console import Console
from rich.table import Table
from core.utils import *
from core.utils.models import *
console = Console()

# ! You can modify your own weights here
# Chinese and Japanese 2.5 characters, Korean 2 characters, Thai 1.5 characters, full-width symbols 2 characters, other English-based and half-width symbols 1 character
def calc_len(text: str) -> float:
    text = str(text) # force convert
    def char_weight(char):
        code = ord(char)
        if 0x4E00 <= code <= 0x9FFF or 0x3040 <= code <= 0x30FF:  # Chinese and Japanese
            return 1.75
        elif 0xAC00 <= code <= 0xD7A3 or 0x1100 <= code <= 0x11FF:  # Korean
            return 1.5
        elif 0x0E00 <= code <= 0x0E7F:  # Thai
            return 1
        elif 0xFF01 <= code <= 0xFF5E:  # full-width symbols
            return 1.75
        else:  # other characters (e.g. English and half-width symbols)
            return 1

    return sum(char_weight(char) for char in text)

def align_subs(src_sub: str, tr_sub: str, src_part: str) -> Tuple[List[str], List[str], str]:
    align_prompt = get_align_prompt(src_sub, tr_sub, src_part)
    
    def valid_align(response_data):
        if not isinstance(response_data, dict):
            return {"status": "error", "message": f"Response is not a dict: {type(response_data).__name__}"}
        if 'align' not in response_data:
            return {"status": "error", "message": "Missing required key: `align`"}
        if len(response_data['align']) < 2:
            return {"status": "error", "message": "Align does not contain more than 1 part as expected!"}
        return {"status": "success", "message": "Align completed"}
    parsed = ask_gpt(align_prompt, resp_type='json', valid_def=valid_align, log_title='align_subs')
    align_data = parsed['align']
    src_parts = src_part.split('\n')
    tr_parts = [item[f'target_part_{i+1}'].strip() for i, item in enumerate(align_data)]
    
    whisper_language = load_key("whisper.language")
    language = load_key("whisper.detected_language") if whisper_language == 'auto' else whisper_language
    joiner = get_joiner(language)
    tr_remerged = joiner.join(tr_parts)
    
    table = Table(title="🔗 Aligned parts")
    table.add_column("Language", style="cyan")
    table.add_column("Parts", style="magenta")
    table.add_row("SRC_LANG", "\n".join(src_parts))
    table.add_row("TARGET_LANG", "\n".join(tr_parts))
    console.print(table)
    
    return src_parts, tr_parts, tr_remerged

def _batch_align_subs(items):
    """Batch align N subtitle pairs in one LLM call.
    items: list of (src_sub, tr_sub, src_part). Returns parsed response dict or None on failure.
    """
    prompt = get_batch_align_prompt(items)

    def valid_batch_align(rd):
        if not isinstance(rd, dict):
            return {"status": "error", "message": "Not a dict"}
        for li in range(1, len(items) + 1):
            k = str(li)
            if k not in rd:
                return {"status": "error", "message": f"Missing key '{k}'"}
            if "align" not in rd[k] or not isinstance(rd[k]["align"], list) or len(rd[k]["align"]) < 2:
                return {"status": "error", "message": f"Invalid 'align' in item '{k}'"}
        return {"status": "success", "message": "Batch align completed"}

    try:
        return ask_gpt(prompt, resp_type='json', valid_def=valid_batch_align, log_title='batch_align_subs')
    except Exception as e:
        console.print(f"[yellow]⚠️ Batch align failed: {e}[/yellow]")
        return None


def split_align_subs(src_lines: List[str], tr_lines: List[str]):
    subtitle_set = load_key("subtitle")
    MAX_SUB_LENGTH = subtitle_set["max_length"]
    TARGET_SUB_MULTIPLIER = subtitle_set["target_multiplier"]
    remerged_tr_lines = tr_lines.copy()

    to_split = []
    for i, (src, tr) in enumerate(zip(src_lines, tr_lines)):
        src, tr = str(src), str(tr)
        if len(src) > MAX_SUB_LENGTH or calc_len(tr) * TARGET_SUB_MULTIPLIER > MAX_SUB_LENGTH:
            to_split.append(i)
            table = Table(title=f"📏 Line {i} needs to be split")
            table.add_column("Type", style="cyan")
            table.add_column("Content", style="magenta")
            table.add_row("Source Line", src)
            table.add_row("Target Line", tr)
            console.print(table)

    efficiency_mode = load_key("efficiency_mode")

    from core.utils.progress_utils import get_progress, update_st_progress
    progress = get_progress()
    is_internal = not progress.live.is_started
    if is_internal: progress.start()

    if efficiency_mode and to_split:
        task_desc = "📏 正在批量分割和对齐长字幕..."
        task = progress.add_task(task_desc, total=2) # 2 steps: split and align
        
        word_limit = load_key("max_split_length")
        
        # 💡 读取用户的批量最大行数（句子数）设置来动态规划分块大小
        batch_size = load_key("batch_split_size")
        if batch_size is None or batch_size == 0:
            chunk_size = 10
        else:
            chunk_size = min(batch_size, 20)  # 对齐阶段对大批量数据极为敏感，最大不超过 20 以确保生成绝对不超时
            
        # 对待切分列表实施精细分块
        chunks = [to_split[i:i + chunk_size] for i in range(0, len(to_split), chunk_size)]
        console.print(f"[cyan]⚡ Efficiency mode: splitting {len(to_split)} lines in {len(chunks)} LLM chunks (chunk size: {chunk_size})[/cyan]")

        # Step 1: Batch split all source sentences in safe mini-batches
        update_st_progress(0, 2, f"{task_desc} (1/2)")
        split_src_map = {}
        
        for chunk_idx, chunk in enumerate(chunks, 1):
            chunk_items = [(str(src_lines[i]), 2, word_limit) for i in chunk]
            split_prompt = get_batch_split_prompt(chunk_items)

            def valid_bs(rd, c_len=len(chunk)):
                if not isinstance(rd, dict):
                    return {"status": "error", "message": "Not a dict"}
                for li in range(1, c_len + 1):
                    k = str(li)
                    if k not in rd or "split" not in rd.get(k, {}):
                        return {"status": "error", "message": f"Missing '{k}'"}
                return {"status": "success", "message": "OK"}

            try:
                split_resp = ask_gpt(split_prompt, resp_type='json', valid_def=valid_bs, log_title=f'batch_split_chunk_{chunk_idx}')
                for local_i, orig_i in enumerate(chunk, 1):
                    split_val = split_resp.get(str(local_i), {}).get("split", "")
                    if split_val and "[br]" in split_val:
                        split_src_map[orig_i] = _apply_split(str(src_lines[orig_i]), split_val).strip()
                    else:
                        console.print(f"[yellow]⚠️ Batch split missing [br] for line {orig_i}, using individual split[/yellow]")
                        split_src_map[orig_i] = split_sentence(str(src_lines[orig_i]), num_parts=2).strip()
            except Exception as e:
                console.print(f"[yellow]⚠️ Batch split chunk {chunk_idx} failed: {e}. Falling back to individual splits for this chunk.[/yellow]")
                for orig_i in chunk:
                    split_src_map[orig_i] = split_sentence(str(src_lines[orig_i]), num_parts=2).strip()
        
        progress.update(task, advance=1)
        
        # Step 2: Batch align all pairs in safe mini-batches (2/2)
        update_st_progress(1, 2, f"{task_desc} (2/2)")
        
        whisper_language = load_key("whisper.language")
        language = load_key("whisper.detected_language") if whisper_language == 'auto' else whisper_language
        joiner = get_joiner(language)

        align_chunks = [to_split[i:i + chunk_size] for i in range(0, len(to_split), chunk_size)]
        console.print(f"[cyan]⚡ Efficiency mode: aligning {len(to_split)} lines in {len(align_chunks)} LLM chunks (chunk size: {chunk_size})[/cyan]")

        for chunk_idx, chunk in enumerate(align_chunks, 1):
            chunk_items = [(str(src_lines[i]), str(tr_lines[i]), split_src_map[i]) for i in chunk]
            batch_resp = _batch_align_subs(chunk_items)

            for local_i, orig_i in enumerate(chunk, 1):
                if batch_resp and str(local_i) in batch_resp:
                    align_data = batch_resp[str(local_i)]["align"]
                    src_parts = split_src_map[orig_i].split('\n')
                    tr_parts = [item[f'target_part_{j+1}'].strip() for j, item in enumerate(align_data)]
                    src_lines[orig_i] = src_parts
                    tr_lines[orig_i] = tr_parts
                    remerged_tr_lines[orig_i] = joiner.join(tr_parts)
                else:
                    console.print(f"[yellow]⚠️ Batch align chunk {chunk_idx} missing result for line {orig_i}, using individual align[/yellow]")
                    src_parts, tr_parts, tr_remerged = align_subs(str(src_lines[orig_i]), str(tr_lines[orig_i]), split_src_map[orig_i])
                    src_lines[orig_i] = src_parts
                    tr_lines[orig_i] = tr_parts
                    remerged_tr_lines[orig_i] = tr_remerged
        
        progress.update(task, advance=1)
        update_st_progress(2, 2, task_desc)
        if not is_internal: progress.remove_task(task)

    elif to_split:
        @except_handler("Error in split_align_subs")
        def process(i):
            split_src = split_sentence(src_lines[i], num_parts=2).strip()
            src_parts, tr_parts, tr_remerged = align_subs(src_lines[i], tr_lines[i], split_src)
            return i, src_parts, tr_parts, tr_remerged

        task_desc = "📏 正在分割过长字幕..."
        task = progress.add_task(task_desc, total=len(to_split))
        with concurrent.futures.ThreadPoolExecutor(max_workers=load_key("max_workers")) as executor:
            futures = [executor.submit(process, i) for i in to_split]
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                idx, src_parts, tr_parts, tr_remerged = future.result()
                src_lines[idx] = src_parts
                tr_lines[idx] = tr_parts
                remerged_tr_lines[idx] = tr_remerged
                progress.update(task, advance=1)
                update_st_progress(i + 1, len(to_split), task_desc)
                
        if not is_internal: progress.remove_task(task)

    if is_internal: progress.stop()

    # Flatten `src_lines` and `tr_lines`
    src_lines = [item for sublist in src_lines for item in (sublist if isinstance(sublist, list) else [sublist])]
    tr_lines = [item for sublist in tr_lines for item in (sublist if isinstance(sublist, list) else [sublist])]

    return src_lines, tr_lines, remerged_tr_lines

def _propagate_corrected(orig_src, orig_corrected, new_src):
    """Distribute corrected source text proportionally when source lines are split.
    orig_src: list of source lines before splitting
    orig_corrected: list of corrected source lines aligned with orig_src
    new_src: list of source lines after splitting (may be longer due to splits)
    Returns list of corrected source lines aligned with new_src.
    """
    result = []
    new_idx = 0
    for orig_i, orig_line in enumerate(orig_src):
        orig_word_count = len(str(orig_line).split())
        corr_words = str(orig_corrected[orig_i]).split()

        # Collect new_src lines that together cover this orig_line
        new_lines = []
        consumed = 0
        while new_idx < len(new_src) and consumed < orig_word_count:
            nl = str(new_src[new_idx])
            new_lines.append(nl)
            consumed += len(nl.split())
            new_idx += 1

        if len(new_lines) <= 1:
            # No split happened — keep full corrected line
            result.append(' '.join(corr_words))
        else:
            # Split happened — distribute corrected words by each new line's word count
            prev = 0
            for j, nl in enumerate(new_lines):
                wc = len(nl.split())
                if j == len(new_lines) - 1:
                    result.append(' '.join(corr_words[prev:]) or nl)
                else:
                    result.append(' '.join(corr_words[prev:prev + wc]) or nl)
                prev += wc
    return result


def split_for_sub_main():
    console.print("[bold green]🚀 Start splitting subtitles...[/bold green]")

    df = pd.read_excel(_4_2_TRANSLATION)
    src = df['Source'].tolist()
    trans = df['Translation'].tolist()
    has_corrected = 'Corrected_Source' in df.columns
    corrected = df['Corrected_Source'].tolist() if has_corrected else None

    subtitle_set = load_key("subtitle")
    MAX_SUB_LENGTH = subtitle_set["max_length"]
    TARGET_SUB_MULTIPLIER = subtitle_set["target_multiplier"]

    for attempt in range(3):  # 多次切割
        console.print(Panel(f"🔄 Split attempt {attempt + 1}", expand=False))
        prev_src = src.copy()
        split_src, split_trans, remerged = split_align_subs(src.copy(), trans)

        # Propagate corrected source through splits
        if has_corrected:
            split_corrected = _propagate_corrected(prev_src, corrected, split_src)

        # 检查是否所有字幕都符合长度要求
        if all(len(s) <= MAX_SUB_LENGTH for s in split_src) and \
           all(calc_len(tr) * TARGET_SUB_MULTIPLIER <= MAX_SUB_LENGTH for tr in split_trans):
            break

        # 更新源数据继续下一轮分割
        src, trans = split_src, split_trans
        if has_corrected:
            corrected = split_corrected

    # 确保二者有相同的长度，防止报错
    if len(src) > len(remerged):
        remerged += [None] * (len(src) - len(remerged))
    elif len(remerged) > len(src):
        src += [None] * (len(remerged) - len(src))

    if has_corrected:
        pd.DataFrame({'Source': split_src, 'Translation': split_trans, 'Corrected_Source': split_corrected}).to_excel(_5_SPLIT_SUB, index=False)
    else:
        pd.DataFrame({'Source': split_src, 'Translation': split_trans}).to_excel(_5_SPLIT_SUB, index=False)
    pd.DataFrame({'Source': src, 'Translation': remerged}).to_excel(_5_REMERGED, index=False)

if __name__ == '__main__':
    split_for_sub_main()
