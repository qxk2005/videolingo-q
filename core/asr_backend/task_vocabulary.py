"""Explicit selected-video vocabulary; absent marker preserves global behavior."""
import json
from pathlib import Path
from urllib.parse import urlparse, parse_qs
ALLOWED_IDS = {'gwduNjsOaNA', 'QER-0DaC-Gk', 'sNjFA7-_8k8', 'i0P56Pm1Q3U'}
def scoped_vocabulary(default_value):
    markers = [p for p in (Path('output/task-domain-vocab.json'), Path('output/.task-domain-vocab.json')) if p.exists()]
    if not markers:
        return default_value
    policies = [json.loads(p.read_text()) for p in markers]
    if any(p != policies[0] for p in policies):
        raise RuntimeError('Conflicting task vocabulary markers; stop for review')
    policy = policies[0]
    url = urlparse(Path('output/youtube_url.txt').read_text().strip())
    video_id = url.path.strip('/') if url.hostname in ('youtu.be', 'www.youtu.be') else parse_qs(url.query).get('v', [''])[0]
    if video_id not in ALLOWED_IDS or policy.get('video_id') != video_id:
        raise RuntimeError('Task vocabulary marker does not match current selected video')
    value = policy.get('domain_vocab', '')
    if not isinstance(value, str):
        raise RuntimeError('Invalid task vocabulary value')
    return value
