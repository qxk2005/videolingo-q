import os
import re
import json
import subprocess
from threading import Lock
import json_repair
from openai import OpenAI
from core.utils.config_utils import load_key
from rich import print as rprint
from core.utils.decorator import except_handler

# ------------
# cache gpt response
# ------------

LOCK = Lock()
GPT_LOG_FOLDER = 'output/gpt_log'

def _save_cache(model, prompt, resp_content, resp_type, resp, message=None, log_title="default"):
    with LOCK:
        logs = []
        file = os.path.join(GPT_LOG_FOLDER, f"{log_title}.json")
        os.makedirs(os.path.dirname(file), exist_ok=True)
        if os.path.exists(file):
            with open(file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        logs.append({"model": model, "prompt": prompt, "resp_content": resp_content, "resp_type": resp_type, "resp": resp, "message": message})
        with open(file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=4)

def _load_cache(prompt, resp_type, log_title):
    with LOCK:
        file = os.path.join(GPT_LOG_FOLDER, f"{log_title}.json")
        if os.path.exists(file):
            with open(file, 'r', encoding='utf-8') as f:
                for item in json.load(f):
                    if item["prompt"] == prompt and item["resp_type"] == resp_type:
                        return item["resp"]
        return False

# ------------
# ask gpt once
# ------------

def ask_antigravity_cli(prompt):
    try:
        # Use agy -p "<prompt>"
        result = subprocess.run(
            ["agy", "-p", prompt],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise ValueError(f"Antigravity CLI call failed: {e.stderr}")

def _normalize_keys_to_strings(data):
    if isinstance(data, dict):
        return {str(k): _normalize_keys_to_strings(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_normalize_keys_to_strings(item) for item in data]
    else:
        return data

@except_handler("GPT request failed", retry=3, delay=0.5)
def ask_gpt(prompt, resp_type=None, valid_def=None, log_title="default"):
    # check cache
    cached = _load_cache(prompt, resp_type, log_title)
    if cached:
        rprint("use cache response")
        return cached

    if load_key("api.use_antigravity_cli") or load_key("api.use_gemini_cli"):
        model = "antigravity-cli"
        resp_content = ask_antigravity_cli(prompt)
    else:
        if not load_key("api.key"):
            raise ValueError("API key is not set")
        model = load_key("api.model")
        base_url = load_key("api.base_url")
        if 'ark' in base_url:
            base_url = "https://ark.cn-beijing.volces.com/api/v3" # huoshan base url
        elif 'v1' not in base_url:
            base_url = base_url.strip('/') + '/v1'
        client = OpenAI(api_key=load_key("api.key"), base_url=base_url)
        # response_format = {"type": "json_object"} if resp_type == "json" and load_key("api.llm_support_json") else None

        messages = [{"role": "user", "content": prompt}]

        params = dict(
            model=model,
            messages=messages,
            #response_format=response_format,
            timeout=300
        )
        resp_raw = client.chat.completions.create(**params)

        # process and return full result
        resp_content = resp_raw.choices[0].message.content

    if resp_type == "json":
        # Extract JSON from ```json ... ``` code block if present (handles thinking models
        # that output analysis text before the actual JSON, with or without <think> tags).
        # Take the LAST block in case the thinking text contains example ```json``` snippets.
        json_matches = re.findall(r'```json\s*(.*?)\s*```', resp_content, flags=re.DOTALL)
        json_str = json_matches[-1] if json_matches else resp_content
        resp = json_repair.loads(json_str)
        if isinstance(resp, list):
            resp = resp[0] if resp and isinstance(resp[0], dict) else {}
        # If json_repair still returned a string (e.g. model returned plain text or a bare
        # JSON string), do a second-pass search for the last {...} block in the raw response.
        if isinstance(resp, str):
            obj_match = re.search(r'\{.*\}', resp_content, flags=re.DOTALL)
            if obj_match:
                resp = json_repair.loads(obj_match.group())
            if isinstance(resp, str):
                resp = {}  # force validation failure so the retry kicks in
        resp = _normalize_keys_to_strings(resp)
    else:
        # Strip <think> blocks for plain text responses
        resp = re.sub(r'<think>.*?</think>', '', resp_content, flags=re.DOTALL).strip()
    
    # check if the response format is valid
    if valid_def:
        valid_resp = valid_def(resp)
        if valid_resp['status'] != 'success':
            _save_cache(model, prompt, resp_content, resp_type, resp, log_title="error", message=valid_resp['message'])
            raise ValueError(f"❎ API response error: {valid_resp['message']}")

    _save_cache(model, prompt, resp_content, resp_type, resp, log_title=log_title)
    return resp


if __name__ == '__main__':
    from rich import print as rprint
    
    result = ask_gpt("""test respond ```json\n{\"code\": 200, \"message\": \"success\"}\n```""", resp_type="json")
    rprint(f"Test json output result: {result}")
