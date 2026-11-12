"""Read the existing public Doubao catalog without importing UI or configuration."""
import ast
from pathlib import Path

DEFAULT_VOICE = "zh_female_vv_uranus_bigtts"


def load_voice_catalog():
    source = Path(__file__).parent / "st_utils/sidebar_setting.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    catalog = {}
    for constant, gender in (("DOUBAO_MALE_VOICES", "male"),
                             ("DOUBAO_FEMALE_VOICES", "female")):
        assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == constant
                               for target in node.targets)]
        if len(assignments) != 1:
            raise ValueError("Doubao catalog unavailable")
        entries = ast.literal_eval(assignments[0].value)
        if not isinstance(entries, dict) or not entries:
            raise ValueError("Doubao catalog invalid")
        for label, voice_id in entries.items():
            if (not isinstance(label, str) or not label or not isinstance(voice_id, str)
                    or not voice_id or voice_id in catalog):
                raise ValueError("Doubao catalog invalid")
            catalog[voice_id] = {"name": label, "gender": gender}
    if DEFAULT_VOICE not in catalog:
        raise ValueError("Legacy VV voice missing from Doubao catalog")
    return catalog


VOICE_CATALOG = load_voice_catalog()
VOICE_CHOICES = tuple(VOICE_CATALOG)


def validate_voice(voice):
    if not isinstance(voice, str) or voice not in VOICE_CATALOG:
        raise ValueError("Select an exact voice ID from the existing Doubao catalog")
    return voice
