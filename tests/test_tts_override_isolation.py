import concurrent.futures
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from core.tts_backend.doubao_tts import resolve_doubao_voice
from core.tts_backend.edge_tts import edge_tts
from core.tts_backend.elevenlabs_tts import (
    DEFAULT_ELEVENLABS_FEMALE_VOICE,
    DEFAULT_ELEVENLABS_MALE_VOICE,
    resolve_elevenlabs_voice,
)
from core.utils import config_utils


SAMPLE_CONFIG_YAML = """tts_method: elevenlabs_tts
target_language: 简体中文
whisper:
  language: en
elevenlabs_tts:
  voice: el_default_voice_id
  male_voice: el_male_voice_id
  female_voice: el_female_voice_id
doubao_tts:
  voice: zh_female_vv_uranus_bigtts
  male_voice: zh_male_ruyayichen_uranus_bigtts
  female_voice: zh_female_vv_uranus_bigtts
edge_tts:
  voice: zh-CN-XiaoxiaoNeural
  male_voice: zh-CN-YunxiNeural
  female_voice: zh-CN-XiaoxiaoNeural
"""


class RuntimeConfigOverridesTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.cfg_path = os.path.join(self.tmpdir.name, "config.yaml")
        with open(self.cfg_path, "w", encoding="utf-8") as f:
            f.write(SAMPLE_CONFIG_YAML)
        self.cfg_patcher = patch.object(config_utils, "CONFIG_PATH", self.cfg_path)
        self.cfg_patcher.start()

    def tearDown(self):
        self.cfg_patcher.stop()
        self.tmpdir.cleanup()

    def test_runtime_overrides_do_not_mutate_disk_and_support_nested_dicts(self):
        with open(self.cfg_path, "r", encoding="utf-8") as f:
            before_disk = f.read()

        self.assertEqual(config_utils.load_key("tts_method"), "elevenlabs_tts")

        with config_utils.runtime_config_overrides(
            {
                "tts_method": "doubao_tts",
                "edge_tts.male_voice": "zh-CN-YunjianNeural",
            }
        ):
            self.assertEqual(config_utils.load_key("tts_method"), "doubao_tts")
            self.assertEqual(config_utils.load_key("tts_method", ignore_overrides=True), "elevenlabs_tts")
            edge_dict = config_utils.load_key("edge_tts")
            self.assertEqual(edge_dict["male_voice"], "zh-CN-YunjianNeural")
            self.assertEqual(edge_dict["female_voice"], "zh-CN-XiaoxiaoNeural")

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                worker_val = pool.submit(config_utils.load_key, "tts_method").result()
            self.assertEqual(worker_val, "doubao_tts")

            with open(self.cfg_path, "r", encoding="utf-8") as f:
                during_disk = f.read()
            self.assertEqual(before_disk, during_disk)

        self.assertEqual(config_utils.load_key("tts_method"), "elevenlabs_tts")
        self.assertEqual(config_utils.load_key("edge_tts.male_voice"), "zh-CN-YunxiNeural")

    def test_global_update_during_active_task_is_preserved_after_override_exits(self):
        config_utils.update_key("tts_method", "doubao_tts")
        self.assertEqual(config_utils.load_key("tts_method"), "doubao_tts")

        # Simulate a running batch task with doubao_tts override while user switches global config to elevenlabs_tts
        with config_utils.runtime_config_overrides({"tts_method": "doubao_tts"}):
            config_utils.update_key("tts_method", "elevenlabs_tts")
            # Running task continues seeing its own override
            self.assertEqual(config_utils.load_key("tts_method"), "doubao_tts")
            # Persisted config immediately reflects user's global save
            self.assertEqual(config_utils.load_key("tts_method", ignore_overrides=True), "elevenlabs_tts")

        # After task finishes, user's global elevenlabs_tts setting is NOT overwritten
        self.assertEqual(config_utils.load_key("tts_method"), "elevenlabs_tts")


class CrossEngineVoiceGuardTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.cfg_path = os.path.join(self.tmpdir.name, "config.yaml")
        with open(self.cfg_path, "w", encoding="utf-8") as f:
            f.write(SAMPLE_CONFIG_YAML)
        self.cfg_patcher = patch.object(config_utils, "CONFIG_PATH", self.cfg_path)
        self.cfg_patcher.start()

    def tearDown(self):
        self.cfg_patcher.stop()
        self.tmpdir.cleanup()

    def test_elevenlabs_rejects_foreign_doubao_and_edge_voices(self):
        self.assertEqual(
            resolve_elevenlabs_voice("zh_male_ruyayichen_uranus_bigtts", gender="male"),
            "el_male_voice_id",
        )
        self.assertEqual(
            resolve_elevenlabs_voice("BV700_streaming", gender="female"),
            "el_female_voice_id",
        )
        self.assertEqual(
            resolve_elevenlabs_voice("BV001", gender="male"),
            "el_male_voice_id",
        )
        self.assertEqual(
            resolve_elevenlabs_voice("zh-CN-YunxiNeural", gender="male"),
            "el_male_voice_id",
        )
        self.assertEqual(
            resolve_elevenlabs_voice("nPczCjzI2devNBz1zQrb", gender="male"),
            "nPczCjzI2devNBz1zQrb",
        )
        with config_utils.runtime_config_overrides(
            {
                "elevenlabs_tts.male_voice": "",
                "elevenlabs_tts.female_voice": "",
                "elevenlabs_tts.voice": "",
            }
        ):
            self.assertEqual(
                resolve_elevenlabs_voice("zh_male_ruyayichen_uranus_bigtts", gender="male"),
                DEFAULT_ELEVENLABS_MALE_VOICE,
            )
            self.assertEqual(
                resolve_elevenlabs_voice("zh-CN-XiaoxiaoNeural", gender="female"),
                DEFAULT_ELEVENLABS_FEMALE_VOICE,
            )

    def test_doubao_rejects_foreign_elevenlabs_and_edge_voices(self):
        self.assertEqual(
            resolve_doubao_voice("nPczCjzI2devNBz1zQrb", gender="male"),
            "zh_male_ruyayichen_uranus_bigtts",
        )
        self.assertEqual(
            resolve_doubao_voice("zh-CN-XiaoxiaoNeural", gender="female"),
            "zh_female_vv_uranus_bigtts",
        )
        self.assertEqual(
            resolve_doubao_voice("BV700_V2_streaming", gender="male"),
            "BV700_V2_streaming",
        )
        self.assertEqual(
            resolve_doubao_voice("BV001", gender="female"),
            "BV001",
        )

    def test_edge_tts_rejects_foreign_doubao_and_elevenlabs_voices(self):
        out_wav = os.path.join(self.tmpdir.name, "edge_out.mp3")
        captured_voices = []

        class FakeCommunicate:
            def __init__(self, text, voice):
                captured_voices.append(voice)

            async def save(self, path):
                with open(path, "wb") as f:
                    f.write(b"fake")

        with patch("core.tts_backend.edge_tts.edge_tts_sdk.Communicate", FakeCommunicate):
            edge_tts("测试文本", out_wav, voice="zh_male_ruyayichen_uranus_bigtts", gender="male")
            edge_tts("测试文本", out_wav, voice="nPczCjzI2devNBz1zQrb", gender="female")
            edge_tts("测试文本", out_wav, voice="zh-CN-YunjianNeural", gender="male")

        self.assertEqual(
            captured_voices,
            ["zh-CN-YunxiNeural", "zh-CN-XiaoxiaoNeural", "zh-CN-YunjianNeural"],
        )


class BatchTtsSyncAndMcpSchemaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._orig_cwd = os.getcwd()
        cls._boot_tmpdir = tempfile.TemporaryDirectory()
        orig_stdout, orig_stderr = sys.stdout, sys.stderr
        try:
            os.chdir(cls._boot_tmpdir.name)
            import backend.mcp_server as mcp_mod
            import backend.server as srv_mod

            cls.srv_mod = srv_mod
            cls.mcp_mod = mcp_mod
        finally:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr
            os.chdir(cls._orig_cwd)

    @classmethod
    def tearDownClass(cls):
        cls._boot_tmpdir.cleanup()

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.cfg_path = os.path.join(self.tmpdir.name, "config.yaml")
        with open(self.cfg_path, "w", encoding="utf-8") as f:
            f.write(SAMPLE_CONFIG_YAML)
        self.state_file = os.path.join(self.tmpdir.name, "batch", "queue_state.json")
        self.tasks_dir = os.path.join(self.tmpdir.name, "batch", "tasks")
        self.p_cfg = patch.object(config_utils, "CONFIG_PATH", self.cfg_path)
        self.p_state = patch.object(self.srv_mod, "BATCH_STATE_FILE", self.state_file)
        self.p_tasks = patch.object(self.srv_mod, "BATCH_TASKS_DIR", self.tasks_dir)
        self.p_cfg.start()
        self.p_state.start()
        self.p_tasks.start()

    def tearDown(self):
        self.p_tasks.stop()
        self.p_state.stop()
        self.p_cfg.stop()
        self.tmpdir.cleanup()

    def test_sync_tasks_on_global_tts_change(self):
        mgr = self.srv_mod.BatchQueueManager()
        mgr.studio_loaded_task_id = "t_studio"
        mgr.tasks = [
            {
                "id": "t_pending_old",
                "status": "pending",
                "tts_method": "doubao_tts",
                "male_voice": "zh_male_ruyayichen_uranus_bigtts",
                "female_voice": "zh_female_vv_uranus_bigtts",
            },
            {
                "id": "t_studio",
                "status": "paused",
                "tts_method": "edge_tts",
                "male_voice": "zh-CN-YunxiNeural",
                "female_voice": "zh-CN-XiaoxiaoNeural",
            },
            {
                "id": "t_custom_other",
                "status": "pending",
                "tts_method": "edge_tts",
                "male_voice": "zh-CN-YunxiNeural",
                "female_voice": "",
            },
            {
                "id": "t_completed",
                "status": "completed",
                "tts_method": "doubao_tts",
                "male_voice": "zh_male_ruyayichen_uranus_bigtts",
                "female_voice": "",
            },
            {
                "id": "t_running",
                "status": "running",
                "tts_method": "doubao_tts",
                "male_voice": "zh_male_ruyayichen_uranus_bigtts",
                "female_voice": "",
            },
        ]

        changed = mgr.sync_tasks_on_global_tts_change(
            old_tts_method="doubao_tts",
            new_tts_method="elevenlabs_tts",
            explicit_tts_update=True,
        )
        self.assertEqual(changed, 2)

        by_id = {t["id"]: t for t in mgr.tasks}
        # Unfinished task pinned to old global default follows new global default
        self.assertEqual(by_id["t_pending_old"]["tts_method"], "")
        self.assertEqual(by_id["t_pending_old"]["male_voice"], "")
        self.assertEqual(by_id["t_pending_old"]["female_voice"], "")
        # Studio-loaded task follows explicit user TTS switch
        self.assertEqual(by_id["t_studio"]["tts_method"], "")
        self.assertEqual(by_id["t_studio"]["male_voice"], "")
        # Unrelated third-engine pending task and completed/running tasks are untouched
        self.assertEqual(by_id["t_custom_other"]["tts_method"], "edge_tts")
        self.assertEqual(by_id["t_completed"]["tts_method"], "doubao_tts")
        self.assertEqual(by_id["t_running"]["tts_method"], "doubao_tts")

    def test_build_runtime_config_overrides_uses_persisted_tts_when_omitted(self):
        with config_utils.runtime_config_overrides({"tts_method": "doubao_tts"}):
            ov = self.srv_mod._build_runtime_config_overrides(
                tts_method="",
                male_voice="custom_el_male",
                female_voice="custom_el_female",
            )
        self.assertNotIn("tts_method", ov)
        self.assertEqual(ov["elevenlabs_tts.male_voice"], "custom_el_male")
        self.assertEqual(ov["elevenlabs_tts.female_voice"], "custom_el_female")
        self.assertEqual(ov["elevenlabs_tts.voice"], "custom_el_female")

    def test_mcp_batch_add_tasks_tts_fields_default_empty_with_descriptions(self):
        tool_map = {t["name"]: t for t in self.mcp_mod.MCP_TOOLS}
        for tool_name in ("videolingo_batch_add_tasks", "videolingo_one_shot_process"):
            props = tool_map[tool_name]["inputSchema"]["properties"]
            for field in ("tts_method", "male_voice", "female_voice"):
                self.assertIn(field, props)
                self.assertTrue(str(props[field].get("description") or "").strip())


if __name__ == "__main__":
    unittest.main()
