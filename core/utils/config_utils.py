from contextlib import contextmanager
from ruamel.yaml import YAML
import threading

import os
import shutil


def get_app_root():
    """获取应用根目录。
    
    解析策略 (按优先级):
    1. 环境变量 VIDEOLINGO_ROOT (用于打包/自定义部署)
    2. 从本文件路径向上推导: core/utils/config_utils.py → 根目录 (../../..)
    """
    env_root = os.environ.get("VIDEOLINGO_ROOT")
    if env_root and os.path.isdir(env_root):
        return env_root
    
    # 从本文件位置推导: <root>/core/utils/config_utils.py
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


APP_ROOT = get_app_root()
CONFIG_PATH = os.path.join(APP_ROOT, 'config.yaml')
CONFIG_EXAMPLE_PATH = os.path.join(APP_ROOT, 'config.example.yaml')

# Automatically initialize config.yaml from config.example.yaml if missing
if not os.path.exists(CONFIG_PATH) and os.path.exists(CONFIG_EXAMPLE_PATH):
    shutil.copy(CONFIG_EXAMPLE_PATH, CONFIG_PATH)

lock = threading.Lock()

yaml = YAML()
yaml.preserve_quotes = True

def setup_proxy():
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as file:
            data = yaml.load(file)
        if data and 'proxy' in data and data['proxy']:
            proxy_val = str(data['proxy']).strip()
            if proxy_val:
                os.environ['http_proxy'] = proxy_val
                os.environ['https_proxy'] = proxy_val
                os.environ['HTTP_PROXY'] = proxy_val
                os.environ['HTTPS_PROXY'] = proxy_val
            else:
                for key in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
                    os.environ.pop(key, None)
        else:
            for key in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
                os.environ.pop(key, None)
    except Exception:
        pass

# Initialize proxy on load
setup_proxy()

# -----------------------
# load & update config
# -----------------------

_runtime_override_stack = []


@contextmanager
def runtime_config_overrides(overrides: dict = None):
    """Apply temporary in-memory config overrides without mutating `config.yaml` on disk."""
    clean = {str(k): v for k, v in (overrides or {}).items() if k and v is not None}
    if not clean:
        yield
        return
    with lock:
        _runtime_override_stack.append(clean)
    try:
        yield
    finally:
        with lock:
            for idx in range(len(_runtime_override_stack) - 1, -1, -1):
                if _runtime_override_stack[idx] is clean:
                    _runtime_override_stack.pop(idx)
                    break


def load_key(key, ignore_overrides: bool = False):
    with lock:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as file:
            data = yaml.load(file)
        merged_overrides = {}
        if _runtime_override_stack and not ignore_overrides:
            for layer in _runtime_override_stack:
                merged_overrides.update(layer)

    if merged_overrides and isinstance(data, dict):
        for ov_key, ov_val in merged_overrides.items():
            ov_parts = ov_key.split('.')
            cur = data
            for part in ov_parts[:-1]:
                if part not in cur or not isinstance(cur[part], dict):
                    cur[part] = {}
                cur = cur[part]
            cur[ov_parts[-1]] = ov_val

    keys = key.split('.')
    value = data
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            # Return None instead of raising KeyError to allow graceful handling in UI
            return None
    return value

def update_key(key, new_value):
    with lock:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as file:
            data = yaml.load(file)

        keys = key.split('.')
        current = data
        for k in keys[:-1]:
            if k not in current or not isinstance(current[k], dict):
                current[k] = {}
            current = current[k]

        current[keys[-1]] = new_value
        with open(CONFIG_PATH, 'w', encoding='utf-8') as file:
            yaml.dump(data, file)
        
        if key == 'proxy':
            setup_proxy()
        return True
        
# basic utils
def get_joiner(language):
    if language in load_key('language_split_with_space'):
        return " "
    elif language in load_key('language_split_without_space'):
        return ""
    else:
        raise ValueError(f"Unsupported language code: {language}")

if __name__ == "__main__":
    print(load_key('language_split_with_space'))
