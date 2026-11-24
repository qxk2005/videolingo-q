import json
import os
import tempfile
import unittest
from unittest.mock import patch

from backend import mcp_server
from core.utils import config_utils, mcp_audit


SAMPLE_CFG = """tts_method: elevenlabs_tts
target_language: 简体中文
whisper:
  language: en
"""


class McpAuditLogTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.audit_path = os.path.join(self.tmpdir.name, "logs", "mcp_audit.jsonl")
        self.cfg_path = os.path.join(self.tmpdir.name, "config.yaml")
        with open(self.cfg_path, "w", encoding="utf-8") as f:
            f.write(SAMPLE_CFG)
        self.env_patcher = patch.dict(os.environ, {"VIDEOLINGO_MCP_AUDIT_PATH": self.audit_path})
        self.cfg_patcher = patch.object(config_utils, "CONFIG_PATH", self.cfg_path)
        self.env_patcher.start()
        self.cfg_patcher.start()

    def tearDown(self):
        self.cfg_patcher.stop()
        self.env_patcher.stop()
        self.tmpdir.cleanup()

    def test_redaction_masks_all_secrets_and_preserves_safe_fields(self):
        secret_el = "sk_fake_elevenlabs_secret_1234567890"
        secret_db = "db_access_token_abcdef123456"
        secret_cookie = "# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tTRUE\t1999999999\tSAPISID\tmy_secret_cookie"
        raw_payload = {
            "updates": {
                "tts_method": "elevenlabs_tts",
                "elevenlabs_tts.api_key": secret_el,
                "doubao_tts": {"access_token": secret_db, "male_voice": "zh_male_ruyayichen_uranus_bigtts"},
                "api": {"key": "sk-proj-supersecretkeyvalue12345"},
                "pyannote.hf_token": "hf_abcdefghijklmnopqrstuvwxyz",
                "youtube": {"cookies_from_browser": "chrome", "cookies_path": "output/youtube_cookies.txt"},
            },
            "cookies_content": secret_cookie,
            "note": "Authorization: Bearer my_bearer_secret_token_99999",
        }

        redacted = mcp_audit.redact_sensitive_data(raw_payload)
        serialized = json.dumps(redacted, ensure_ascii=False)

        self.assertNotIn(secret_el, serialized)
        self.assertNotIn(secret_db, serialized)
        self.assertNotIn("my_secret_cookie", serialized)
        self.assertNotIn("supersecretkeyvalue12345", serialized)
        self.assertNotIn("hf_abcdefghijklmnopqrstuvwxyz", serialized)
        self.assertNotIn("my_bearer_secret_token_99999", serialized)

        # Safe metadata remains readable for traceability
        self.assertEqual(redacted["updates"]["tts_method"], "elevenlabs_tts")
        self.assertEqual(redacted["updates"]["doubao_tts"]["male_voice"], "zh_male_ruyayichen_uranus_bigtts")
        self.assertEqual(redacted["updates"]["youtube"]["cookies_from_browser"], "chrome")
        self.assertEqual(redacted["updates"]["youtube"]["cookies_path"], "output/youtube_cookies.txt")
        self.assertEqual(redacted["updates"]["elevenlabs_tts.api_key"], mcp_audit.REDACTED_PLACEHOLDER)
        self.assertEqual(redacted["cookies_content"], mcp_audit.REDACTED_PLACEHOLDER)

    def test_jsonrpc_session_and_tool_calls_recorded_with_tts_trace(self):
        ctx = {
            "transport": "http",
            "session_id": "sess_trace_01",
            "client_ip": "127.0.0.1",
            "user_agent": "Claude-Desktop/1.5",
        }

        # 1. initialize records clientInfo
        init_resp = mcp_server.handle_jsonrpc_message(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "clientInfo": {"name": "claude-ai", "version": "1.5.0"},
                },
            },
            ctx,
        )
        self.assertEqual(init_resp["result"]["protocolVersion"], "2025-03-26")

        # 2. batch_add_tasks with task-level tts_method override
        def fake_api_request(method, path, payload=None, query=None, timeout=30.0):
            if path == "/api/batch/tasks":
                return {"tasks": [{"id": "task_1001"}], "queue": {"running": False}}
            if path == "/api/batch/start":
                return {"status": "started"}
            if path == "/api/config" and method == "POST":
                updates = (payload or {}).get("updates") or {}
                if "tts_method" in updates:
                    config_utils.update_key("tts_method", updates["tts_method"])
                return {"status": "ok", "tts_method": config_utils.load_key("tts_method")}
            return {"status": "ok", "task": {"running": False}}

        with patch.object(mcp_server, "_api_request", side_effect=fake_api_request):
            call_resp = mcp_server.handle_jsonrpc_message(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": "videolingo_batch_add_tasks",
                        "arguments": {
                            "urls": ["https://www.youtube.com/watch?v=dQw4w9WgXcQ"],
                            "tts_method": "doubao_tts",
                            "auto_start": False,
                        },
                    },
                },
                ctx,
            )
            self.assertFalse(call_resp["result"]["isError"])

            # 3. update_config with secret + global tts_method change
            upd_resp = mcp_server.handle_jsonrpc_message(
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {
                        "name": "videolingo_update_config",
                        "arguments": {
                            "updates": {
                                "tts_method": "edge_tts",
                                "elevenlabs_tts.api_key": "xi-never-write-to-disk-99999",
                            },
                            "cookies_content": "SID=secret_sid; SAPISID=secret_sapisid",
                        },
                    },
                },
                ctx,
            )
            self.assertFalse(upd_resp["result"]["isError"])

        # Verify raw JSONL file on disk never contains secrets
        with open(self.audit_path, "r", encoding="utf-8") as f:
            disk_raw = f.read()
        self.assertNotIn("xi-never-write-to-disk-99999", disk_raw)
        self.assertNotIn("secret_sapisid", disk_raw)

        # Query write-only operations via videolingo_get_mcp_history tool
        hist_resp = mcp_server.handle_jsonrpc_message(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "videolingo_get_mcp_history",
                    "arguments": {"category": "write", "limit": 10},
                },
            },
            ctx,
        )
        self.assertFalse(hist_resp["result"]["isError"])
        hist_data = json.loads(hist_resp["result"]["content"][0]["text"])
        self.assertEqual(hist_data["total_matched"], 2)

        # Newest first: item[0] is videolingo_update_config, item[1] is videolingo_batch_add_tasks
        upd_entry, batch_entry = hist_data["items"][0], hist_data["items"][1]
        self.assertEqual(upd_entry["name"], "videolingo_update_config")
        self.assertEqual(upd_entry["client"]["name"], "claude-ai")
        self.assertEqual(upd_entry["tts_trace"]["global_before"], "elevenlabs_tts")
        self.assertEqual(upd_entry["tts_trace"]["global_after"], "edge_tts")
        self.assertEqual(
            upd_entry["arguments"]["updates"]["elevenlabs_tts.api_key"],
            mcp_audit.REDACTED_PLACEHOLDER,
        )

        self.assertEqual(batch_entry["name"], "videolingo_batch_add_tasks")
        self.assertEqual(batch_entry["client"]["name"], "claude-ai")
        self.assertEqual(batch_entry["tts_trace"]["task_override"], "doubao_tts")
        self.assertEqual(batch_entry["tts_trace"]["global_before"], "elevenlabs_tts")
        self.assertIn("task_1001", batch_entry["result_summary"]["recent_task_ids"])

        # Resource `videolingo://mcp/history` also works without backend
        res_read = mcp_server.handle_jsonrpc_message(
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "resources/read",
                "params": {"uri": "videolingo://mcp/history"},
            },
            ctx,
        )
        res_payload = json.loads(res_read["result"]["contents"][0]["text"])
        self.assertGreaterEqual(res_payload["stats"]["total"], 4)

    def test_error_recording_and_log_rotation(self):
        ctx = {"transport": "stdio", "session_id": "stdio_test_rot"}

        # Record a failed tool call
        err_resp = mcp_server.handle_jsonrpc_message(
            {
                "jsonrpc": "2.0",
                "id": 99,
                "method": "tools/call",
                "params": {"name": "videolingo_nonexistent_tool", "arguments": {"foo": "bar"}},
            },
            ctx,
        )
        self.assertTrue(err_resp["result"]["isError"])

        err_hist = mcp_audit.query_mcp_audit_history(status="error")
        self.assertEqual(err_hist["total_matched"], 1)
        self.assertEqual(err_hist["items"][0]["name"], "videolingo_nonexistent_tool")
        self.assertIn("未找到名为", err_hist["items"][0]["error"])

        # Test size rotation preserves history across .1 and active file
        with patch.object(mcp_audit, "MAX_AUDIT_FILE_BYTES", 300):
            for idx in range(4):
                mcp_audit.record_mcp_audit_event(
                    event_type="tool_call",
                    name="videolingo_get_status",
                    arguments={"seq": idx},
                    transport="stdio",
                    session_id="stdio_test_rot",
                )
        self.assertTrue(os.path.exists(self.audit_path + ".1"))
        all_hist = mcp_audit.query_mcp_audit_history(limit=20)
        # Rotated + active entries are returned newest-first
        self.assertGreaterEqual(len(all_hist["items"]), 2)
        self.assertEqual(all_hist["items"][0]["arguments"].get("seq"), 3)


if __name__ == "__main__":
    unittest.main()
