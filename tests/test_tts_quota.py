import json
import unittest
from unittest.mock import Mock, patch
from core.tts_backend.quota_probe import get_tts_quota, SUBSCRIPTION_URL

SECRET = 'test-secret-must-never-be-returned'
CONFIG = {'elevenlabs_tts': {'api_key': SECRET}}


class QuotaTests(unittest.TestCase):
    def probe(self, data=None, code=200, config=None):
        response = Mock(status_code=code)
        response.json.return_value = data or {'character_count': 120, 'character_limit': 1000}
        get = Mock(return_value=response)
        result = get_tts_quota('elevenlabs', config_loader=lambda: config or CONFIG, http_get=get)
        self.assertNotIn(SECRET, json.dumps(result))
        return result['providers'][0], get, response

    def test_success_and_safe_request(self):
        result, get, response = self.probe()
        self.assertEqual(result['quota']['remaining'], 880)
        self.assertIsNone(result['balance'])
        self.assertEqual(get.call_args.args, (SUBSCRIPTION_URL,))
        self.assertFalse(get.call_args.kwargs['allow_redirects'])
        self.assertEqual(get.call_args.kwargs['timeout'], (5, 12))
        response.close.assert_called_once()

    def test_overage_is_not_negative_balance(self):
        r, _, _ = self.probe({'character_count': 1100, 'character_limit': 1000, 'max_credit_limit_extension': 'unlimited'})
        self.assertEqual(r['quota']['remaining'], 0)
        self.assertEqual(r['quota']['over_limit'], 100)
        self.assertEqual(r['max_credit_limit_extension'], 'unlimited')

    def test_invalid_data_never_zero(self):
        for data in [{'character_count': True, 'character_limit': 10}, {'character_count': -1, 'character_limit': 10}, {'character_count': '1', 'character_limit': 10}, {'other': SECRET}, []]:
            response = Mock(status_code=200)
            response.json.return_value = data
            r = get_tts_quota('elevenlabs', config_loader=lambda: CONFIG, http_get=Mock(return_value=response))
            self.assertEqual(r['providers'][0]['status'], 'invalid_response')
            self.assertIsNone(r['providers'][0]['quota'])
            self.assertNotIn(SECRET, json.dumps(r))

    def test_errors_no_response_body_or_retry(self):
        for code in (301, 401, 403, 429, 500):
            r, get, response = self.probe({'message': SECRET}, code)
            self.assertNotEqual(r['status'], 'ok')
            self.assertIsNone(r['quota'])
            get.assert_called_once()
            if code not in (401, 403):
                response.json.assert_not_called()

    def test_scoped_permission_can_be_http_401(self):
        r, _, _ = self.probe({'detail': {'status': 'missing_permissions', 'message': SECRET}}, 401)
        self.assertEqual(r['status'], 'permission_denied')

    def test_exception_no_leak(self):
        get = Mock(side_effect=RuntimeError(SECRET))
        r = get_tts_quota('elevenlabs', config_loader=lambda: CONFIG, http_get=get)
        self.assertEqual(r['providers'][0]['status'], 'request_failed')
        self.assertNotIn(SECRET, json.dumps(r))
        get.assert_called_once()

    def test_host_protection_before_credential_use(self):
        for base in ('https://evil.invalid', 'http://api.elevenlabs.io', 'https://api.elevenlabs.io@evil.invalid', 'https://api.elevenlabs.io/path'):
            r, get, _ = self.probe(config={'elevenlabs_tts': {'api_key': SECRET, 'base_url': base}})
            self.assertEqual(r['status'], 'unsupported_endpoint')
            get.assert_not_called()

    def test_missing_key(self):
        r, get, _ = self.probe(config={'elevenlabs_tts': {'api_key': 'YOUR_ELEVENLABS_API_KEY'}})
        self.assertEqual(r['status'], 'not_configured')
        get.assert_not_called()

    def test_unsupported_providers_do_not_read_credentials(self):
        loader = Mock(side_effect=AssertionError('must not read config'))
        get = Mock(side_effect=AssertionError('must not use network'))
        for provider in ('doubao', 'edge'):
            r = get_tts_quota(provider, config_loader=loader, http_get=get)
            self.assertIsNone(r['providers'][0]['quota'])
        loader.assert_not_called()
        get.assert_not_called()

    def test_config_error_safe(self):
        r = get_tts_quota(config_loader=Mock(side_effect=RuntimeError(SECRET)))
        self.assertEqual(len(r['providers']), 3)
        self.assertNotIn(SECRET, json.dumps(r))

    def test_unknown_arguments(self):
        for provider in ('other', None, [], 12):
            with self.assertRaises(ValueError):
                get_tts_quota(provider)

    def test_mcp_discovery_and_dispatch_no_backend(self):
        from backend import mcp_server
        listed = mcp_server.handle_jsonrpc_message({'jsonrpc': '2.0', 'id': 1, 'method': 'tools/list'})
        tool = next(t for t in listed['result']['tools'] if t['name'] == 'videolingo_get_tts_quota')
        self.assertTrue(tool['annotations']['readOnlyHint'])
        with patch.object(mcp_server, '_api_request', side_effect=AssertionError('backend forbidden')):
            r = mcp_server.handle_jsonrpc_message({'jsonrpc': '2.0', 'id': 2, 'method': 'tools/call', 'params': {'name': tool['name'], 'arguments': {'provider': 'edge'}}})
            self.assertFalse(r['result']['isError'])
            for args in ({'api_key': SECRET}, [], {'provider': 'other'}):
                with self.assertRaises(ValueError):
                    mcp_server.execute_mcp_tool(tool['name'], args)


if __name__ == '__main__':
    unittest.main()
