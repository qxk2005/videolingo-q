import concurrent.futures
from difflib import SequenceMatcher
import math
from core.prompts import get_split_prompt, get_batch_split_prompt
from core.spacy_utils.load_nlp_model import init_nlp
from core.utils import *
from rich.console import Console
from rich.table import Table
from core.utils.models import _3_1_SPLIT_BY_NLP, _3_2_SPLIT_BY_MEANING
console = Console()

def tokenize_sentence(sentence, nlp):
    doc = nlp(sentence)
    return [token.text for token in doc]

def find_split_positions(original, modified):
    split_positions = []
    parts = modified.split('[br]')
    start = 0
    whisper_language = load_key("whisper.language")
    language = load_key("whisper.detected_language") if whisper_language == 'auto' else whisper_language
    joiner = get_joiner(language)

    for i in range(len(parts) - 1):
        max_similarity = 0
        best_split = None

        for j in range(start, len(original)):
            original_left = original[start:j]
            modified_left = joiner.join(parts[i].split())

            left_similarity = SequenceMatcher(None, original_left, modified_left).ratio()

            if left_similarity > max_similarity:
                max_similarity = left_similarity
                best_split = j

        if max_similarity < 0.9:
            console.print(f"[yellow]Warning: low similarity found at the best split point: {max_similarity}[/yellow]")
        if best_split is not None:
            split_positions.append(best_split)
            start = best_split
        else:
            console.print(f"[yellow]Warning: Unable to find a suitable split point for the {i+1}th part.[/yellow]")

    return split_positions

def split_sentence_locally(sentence, num_parts=2):
    """本地标点与物理长度结合的兜底切分函数，在 GPT 请求彻底失败时保障系统绝不崩溃。"""
    words = sentence.split()
    if len(words) <= 1:
        return sentence
    
    total_words = len(words)
    mid = total_words // 2
    
    # 优先在靠近中点的逗号、分号或冒号位置切分以保证较好的语义可读性
    best_split_idx = mid
    for offset in range(min(5, mid)):
        for check_idx in [mid - offset, mid + offset]:
            if 0 < check_idx < total_words:
                prev_word = words[check_idx - 1]
                if prev_word.endswith(',') or prev_word.endswith(';') or prev_word.endswith(':'):
                    best_split_idx = check_idx
                    break
        else:
            continue
        break
    
    part1 = " ".join(words[:best_split_idx])
    part2 = " ".join(words[best_split_idx:])
    return f"{part1}\n{part2}"

def split_sentence(sentence, num_parts, word_limit=20, index=-1, retry_attempt=0):
    """Split a long sentence using GPT and return the result as a string."""
    split_prompt = get_split_prompt(sentence, num_parts, word_limit)
    def valid_split(response_data):
        if not isinstance(response_data, dict):
            return {"status": "error", "message": f"Response is not a dict: {type(response_data).__name__}"}
        choice = response_data["choice"]
        if f'split{choice}' not in response_data:
            return {"status": "error", "message": "Missing required key: `split`"}
        if "[br]" not in response_data[f"split{choice}"]:
            return {"status": "error", "message": "Split failed, no [br] found"}
        return {"status": "success", "message": "Split completed"}
    
    try:
        response_data = ask_gpt(split_prompt + " " * retry_attempt, resp_type='json', valid_def=valid_split, log_title='split_by_meaning')
        choice = response_data["choice"]
        best_split = response_data[f"split{choice}"]
        split_points = find_split_positions(sentence, best_split)
        # split the sentence based on the split points
        for i, split_point in enumerate(split_points):
            if i == 0:
                best_split = sentence[:split_point] + '\n' + sentence[split_point:]
            else:
                parts = best_split.split('\n')
                last_part = parts[-1]
                parts[-1] = last_part[:split_point - split_points[i-1]] + '\n' + last_part[split_point - split_points[i-1]:]
                best_split = '\n'.join(parts)
    except Exception as e:
        console.print(f"[yellow]⚠️ GPT split failed: {e}. Falling back to local physics-based split.[/yellow]")
        best_split = split_sentence_locally(sentence, num_parts)

    if index != -1:
        console.print(f'[green]✅ Sentence {index} has been successfully split[/green]')
    table = Table(title="")
    table.add_column("Type", style="cyan")
    table.add_column("Sentence")
    table.add_row("Original", sentence, style="yellow")
    table.add_row("Split", best_split.replace('\n', ' ||'), style="yellow")
    console.print(table)
    
    return best_split

def _apply_split(sentence, split_str_with_br):
    """Apply find_split_positions result to get a '\n'-split sentence string."""
    split_points = find_split_positions(sentence, split_str_with_br)
    result = sentence
    for i, split_point in enumerate(split_points):
        if i == 0:
            result = sentence[:split_point] + '\n' + sentence[split_point:]
        else:
            parts = result.split('\n')
            last_part = parts[-1]
            adj = split_point - split_points[i - 1]
            parts[-1] = last_part[:adj] + '\n' + last_part[adj:]
            result = '\n'.join(parts)
    return result


def batch_split_sentences(sentences, max_length, nlp, retry_attempt=0):
    """Batch all long sentences into a single LLM call.
    Returns flat sentence list on success, or None on LLM failure (signal to fall back).
    """
    new_sentences = [[s] for s in sentences]

    items_to_split = []
    for idx, sentence in enumerate(sentences):
        tokens = tokenize_sentence(sentence, nlp)
        if len(tokens) > max_length:
            num_parts = math.ceil(len(tokens) / max_length)
            items_to_split.append((idx, sentence, num_parts))

    if not items_to_split:
        return [s for sublist in new_sentences for s in sublist]

    console.print(f"[cyan]⚡ Efficiency mode: batch-splitting {len(items_to_split)} sentences in 1 LLM call[/cyan]")

    prompt_items = [(s, n, max_length) for _, s, n in items_to_split]
    batch_prompt = get_batch_split_prompt(prompt_items) + " " * retry_attempt

    def valid_batch(response_data):
        if not isinstance(response_data, dict):
            return {"status": "error", "message": "Not a dict"}
        for li in range(1, len(items_to_split) + 1):
            k = str(li)
            if k not in response_data:
                return {"status": "error", "message": f"Missing key '{k}'"}
            if not isinstance(response_data.get(k), dict) or "split" not in response_data[k]:
                return {"status": "error", "message": f"Key '{k}' missing 'split'"}
        return {"status": "success", "message": "Batch split completed"}

    try:
        response = ask_gpt(batch_prompt, resp_type='json', valid_def=valid_batch, log_title='batch_split_by_meaning')
    except Exception as e:
        console.print(f"[yellow]⚠️ Batch split LLM call failed: {e}. Will fall back to individual splits.[/yellow]")
        return None

    for local_i, (orig_idx, sentence, num_parts) in enumerate(items_to_split, 1):
        split_val = response.get(str(local_i), {}).get("split", "")
        if split_val and "[br]" in split_val:
            try:
                result_str = _apply_split(sentence, split_val)
                split_lines = [line.strip() for line in result_str.strip().split('\n') if line.strip()]
                new_sentences[orig_idx] = split_lines
                console.print(f"[green]✅ Batch split sentence {orig_idx}[/green]")
            except Exception as e:
                console.print(f"[yellow]⚠️ apply_split failed for sentence {orig_idx}: {e}[/yellow]")
        else:
            console.print(f"[yellow]⚠️ No [br] in batch response for sentence {orig_idx}[/yellow]")

    return [s for sublist in new_sentences for s in sublist]


def parallel_split_sentences(sentences, max_length, max_workers, nlp, retry_attempt=0):
    """Split sentences in parallel using a thread pool."""
    new_sentences = [None] * len(sentences)
    futures = []

    from core.utils.progress_utils import get_progress
    progress = get_progress()
    is_internal = not progress.live.is_started
    
    if is_internal:
        progress.start()
        
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for index, sentence in enumerate(sentences):
            # Use tokenizer to split the sentence
            tokens = tokenize_sentence(sentence, nlp)
            # print("Tokenization result:", tokens)
            num_parts = math.ceil(len(tokens) / max_length)
            if len(tokens) > max_length:
                future = executor.submit(split_sentence, sentence, num_parts, max_length, index=index, retry_attempt=retry_attempt)
                futures.append((future, index, num_parts, sentence))
            else:
                new_sentences[index] = [sentence]

        if futures:
            from core.utils.progress_utils import update_st_progress
            task_desc = "✂️ 正在根据语义分割句子..."
            task = progress.add_task(task_desc, total=len(futures))
            for i, (future, index, num_parts, sentence) in enumerate(futures):
                split_result = future.result()
                if split_result:
                    split_lines = split_result.strip().split('\n')
                    new_sentences[index] = [line.strip() for line in split_lines]
                else:
                    new_sentences[index] = [sentence]
                progress.update(task, advance=1)
                update_st_progress(i + 1, len(futures), task_desc)
            
            if is_internal:
                progress.stop()
            else:
                progress.remove_task(task)

    return [sentence for sublist in new_sentences for sentence in sublist]

@check_file_exists(_3_2_SPLIT_BY_MEANING)
def split_sentences_by_meaning():
    """The main function to split sentences by meaning."""
    with open(_3_1_SPLIT_BY_NLP, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f.readlines()]

    nlp = init_nlp()
    max_length = load_key("max_split_length")
    efficiency_mode = load_key("efficiency_mode")

    # 🔄 process sentences multiple times to ensure all are split
    for retry_attempt in range(3):
        if efficiency_mode:
            result = batch_split_sentences(sentences, max_length, nlp, retry_attempt=retry_attempt)
            if result is None:
                console.print("[yellow]⚠️ Batch split failed, falling back to parallel individual splits[/yellow]")
                sentences = parallel_split_sentences(sentences, max_length, load_key("max_workers"), nlp, retry_attempt=retry_attempt)
            else:
                sentences = result
        else:
            sentences = parallel_split_sentences(sentences, max_length, load_key("max_workers"), nlp, retry_attempt=retry_attempt)

    # 💾 save results
    with open(_3_2_SPLIT_BY_MEANING, 'w', encoding='utf-8') as f:
        f.write('\n'.join(sentences))
    console.print('[green]✅ All sentences have been successfully split![/green]')

if __name__ == '__main__':
    # print(split_sentence('Which makes no sense to the... average guy who always pushes the character creation slider all the way to the right.', 2, 22))
    split_sentences_by_meaning()