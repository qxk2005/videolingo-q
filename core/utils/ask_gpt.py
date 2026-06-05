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
CLI_LOCK = Lock()
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

def extract_auth_url(text):
    match = re.search(r'(https://accounts\.google\.com/o/oauth2/auth[^\s\'"]+)', text)
    return match.group(1) if match else None

def run_command_in_pty(cmd, stdin_data=None, timeout=60):
    """
    Runs a command inside a pseudo-terminal (PTY) to bypass interactive constraints.
    Supports feeding stdin_data if provided.
    """
    import os
    import select
    try:
        import pty
        has_pty = True
    except ImportError:
        has_pty = False

    if not has_pty:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE if stdin_data else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        try:
            stdout, stderr = process.communicate(input=stdin_data, timeout=timeout)
            return process.returncode, stdout, stderr
        except subprocess.TimeoutExpired:
            process.terminate()
            stdout, stderr = process.communicate()
            return -1, stdout or "", stderr or ""

    master, slave = pty.openpty()
    process = subprocess.Popen(
        cmd,
        stdout=slave,
        stderr=slave,
        stdin=slave,
        text=True,
        close_fds=True
    )
    os.close(slave)
    
    if stdin_data:
        try:
            os.write(master, stdin_data.encode('utf-8'))
        except OSError:
            pass
        
    output_chunks = []
    try:
        while True:
            r, w, x = select.select([master], [], [], timeout)
            if not r:
                process.terminate()
                break
            try:
                data = os.read(master, 4096)
                if not data:
                    break
                chunk = data.decode('utf-8', errors='ignore')
                output_chunks.append(chunk)
                
                # Check for interactive login trigger and abort early
                full_so_far = "".join(output_chunks)
                if "Authentication required" in full_so_far or "accounts.google.com" in full_so_far or "authorization code" in full_so_far:
                    process.terminate()
                    break
            except OSError:
                break
    finally:
        try:
            os.close(master)
        except OSError:
            pass
        process.wait()
        
    full_output = "".join(output_chunks)
    return process.returncode, full_output, ""

def login_antigravity_cli(token_code):
    try:
        returncode, output, _ = run_command_in_pty(["agy", "models"], stdin_data=f"{token_code}\n", timeout=20)
        if "Gemini" in output or "GPT" in output or "Claude" in output:
            return True, output
        if returncode == 0:
            return True, output
        return False, output
    except Exception as e:
        return False, str(e)

def ask_antigravity_cli(prompt):
    with CLI_LOCK:
        # ── 1. 执行常规 API 调用 ──
        try:
            returncode, stdout, _ = run_command_in_pty(["agy", "-p", prompt], timeout=120)
            stdout = stdout.strip()
            # 探测是否确实未登录
            if not ("Authentication required" in stdout or "accounts.google.com" in stdout or "authorization code" in stdout):
                return stdout
            auth_output = stdout
        except Exception as e:
            raise ValueError(f"Antigravity CLI call failed: {str(e)}")

        # ── 2. 若执行到此处说明 agy 报了“未登录”错误，我们尝试在后台静默自动登录 ──
        token_code = load_key("api.antigravity_token_code")
        if token_code and token_code.strip():
            rprint("[cyan]🔑 探测到 CLI 未登录/过期，正在使用侧边栏已保存的 Token Code 尝试静默激活授权...[/cyan]")
            try:
                from streamlit.runtime.scriptrunner import get_script_run_ctx
                if get_script_run_ctx() is not None:
                    import streamlit as st
                    st.toast("🔑 探测到 CLI 未登录/过期，正在使用侧边栏已保存的 Token Code 尝试静默激活授权...", icon="🔑")
            except Exception:
                pass
            success, info = login_antigravity_cli(token_code.strip())
            if success:
                rprint("[green]✅ 后台静默授权激活成功！正在重新执行 API 请求...[/green]")
                try:
                    from streamlit.runtime.scriptrunner import get_script_run_ctx
                    if get_script_run_ctx() is not None:
                        import streamlit as st
                        st.toast("✅ 后台静默授权激活成功！正在重新执行 API 请求...", icon="🎉")
                except Exception:
                    pass
                try:
                    # 重新执行刚才失败的 API 调用
                    retry_returncode, retry_stdout, _ = run_command_in_pty(["agy", "-p", prompt], timeout=120)
                    retry_stdout = retry_stdout.strip()
                    if not ("Authentication required" in retry_stdout or "accounts.google.com" in retry_stdout):
                        return retry_stdout
                    auth_output = retry_stdout
                except Exception as err:
                    raise ValueError(f"Antigravity CLI call failed after re-auth: {str(err)}")
            else:
                rprint(f"[red]❌ 使用保存的 Token Code 后台静默激活失败: {info.strip()}[/red]")
                try:
                    from streamlit.runtime.scriptrunner import get_script_run_ctx
                    if get_script_run_ctx() is not None:
                        import streamlit as st
                        st.toast("❌ 使用保存的 Token Code 后台静默激活失败，请在侧边栏填入新的 Token Code 重新授权", icon="⚠️")
                except Exception:
                    pass

        # ── 3. 若未填 Token Code 或后台静默登录失败，则向用户抛出含有 OAuth 链接的友好错误指引 ──
        auth_url = extract_auth_url(auth_output)
        url_msg = f"🔗 <b>授权链接 (Google OAuth URL)：</b><br/><a href='{auth_url}' target='_blank'>{auth_url}</a><br/><br/>" if auth_url else ""
        raise ValueError(
            "检测到您的 Antigravity 命令行工具 (agy) 尚未登录或登录已过期！❌\n\n"
            f"{url_msg}"
            "💡 <b>三步无缝恢复任务步骤：</b>\n"
            "1. <b>获取 Code</b>：点击上方授权链接（或在浏览器中登录 Google 账号），复制返回的 Authorization Code。\n"
            "2. <b>填入配置</b>：将 Code 粘贴到左侧 LLM 配置的「Antigravity Token Code」输入框中（后续系统会妥善用其自动重连，无需反复输入）。\n"
            "3. <b>恢复任务</b>：点击下方的「清除错误并重试」，任务即可无缝继续运行，不需要重新开始！"
        )

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
