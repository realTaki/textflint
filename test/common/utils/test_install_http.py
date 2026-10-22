"""Offline download-boundary tests, without importing NLP models or datasets.

Only the download helper definitions are compiled from the actual source file.
External services are mocked; this is not an integration/model-runtime test.
"""
import ast
import io
import os
from pathlib import Path
import shutil
import tempfile
import types
import unittest
from unittest import mock
import zipfile


class HTTPError(Exception):
    def __init__(self, message='', response=None):
        super().__init__(message)
        self.response = response


class InstallHTTPTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cache = self.tmp.name
        self.response = mock.MagicMock()
        self.response.__enter__.return_value = self.response
        self.response.status_code = 200
        self.response.headers = {'Content-Length': '3'}
        self.response.iter_content.return_value = [b'a', b'', b'bc']
        self.requests = types.SimpleNamespace(
            get=mock.Mock(return_value=self.response), HTTPError=HTTPError)
        self.progress = mock.Mock()
        self.tqdm = types.SimpleNamespace(
            tqdm=mock.Mock(return_value=self.progress))
        self.lock = mock.MagicMock()
        self.filelock = types.SimpleNamespace(
            FileLock=mock.Mock(return_value=self.lock))
        source = Path(os.environ.get(
            'TEXTFLINT_INSTALL_SOURCE',
            str(Path(__file__).resolve().parents[3] /
                'textflint/common/utils/install.py')))
        parsed = ast.parse(source.read_text(encoding='utf-8'))
        names = {'textflint_url', 'path_in_cache', 'unzip_file',
                 'download_if_needed', 'http_get'}
        parsed.body = [node for node in parsed.body
                       if isinstance(node, ast.FunctionDef) and node.name in names]
        self.helpers = dict(os=os, shutil=shutil, tempfile=tempfile,
                            zipfile=zipfile, filelock=self.filelock,
                            requests=self.requests, tqdm=self.tqdm,
                            logger=mock.Mock(), CACHE_DIR=self.cache)
        exec(compile(parsed, str(source), 'exec'), self.helpers)

    def test_https_url(self):
        self.assertEqual(
            self.helpers['textflint_url']('folder/file.zip'),
            'https://textflint.oss-cn-beijing.aliyuncs.com/download/folder/file.zip')

    def test_success_stream_and_transport_options(self):
        output = io.BytesIO()
        proxies = {'https': 'http://example.invalid:8080'}
        self.helpers['http_get']('fixture.bin', output, proxies=proxies)
        self.assertEqual(output.getvalue(), b'abc')
        self.requests.get.assert_called_once_with(
            self.helpers['textflint_url']('fixture.bin'), stream=True,
            proxies=proxies, timeout=(10, 60), allow_redirects=False)
        self.response.raise_for_status.assert_called_once_with()
        self.response.__exit__.assert_called_once()
        self.progress.close.assert_called_once_with()

    def test_http_errors_and_redirects_never_read_or_write_body(self):
        for status in (204, 206, 301, 302, 304, 307, 308, 400, 403, 404, 500):
            with self.subTest(status=status):
                self.response.reset_mock()
                self.response.status_code = status
                self.response.raise_for_status.side_effect = (
                    HTTPError('HTTP error', response=self.response)
                    if status >= 400 else None)
                output = io.BytesIO()
                with self.assertRaises(HTTPError):
                    self.helpers['http_get']('fixture.bin', output)
                self.assertEqual(output.getvalue(), b'')
                self.response.iter_content.assert_not_called()
                self.response.__exit__.assert_called_once()

    def test_stream_failure_closes_response_and_progress(self):
        def broken():
            yield b'a'
            raise OSError('interrupted')
        self.response.iter_content.return_value = broken()
        with self.assertRaises(OSError):
            self.helpers['http_get']('fixture.bin', io.BytesIO())
        self.response.__exit__.assert_called_once()
        self.progress.close.assert_called_once_with()

    def test_tls_failure_has_no_insecure_fallback(self):
        self.requests.get.side_effect = OSError('certificate validation failed')
        with self.assertRaises(OSError):
            self.helpers['http_get']('fixture.bin', io.BytesIO())
        self.requests.get.assert_called_once()
        self.assertTrue(self.requests.get.call_args[0][0].startswith('https://'))
        self.assertNotIn('verify', self.requests.get.call_args[1])

    def test_failure_cleans_temp_and_releases_lock_then_retry_succeeds(self):
        self.response.raise_for_status.side_effect = HTTPError('not found')
        with self.assertRaises(HTTPError):
            self.helpers['download_if_needed']('fixture.bin')
        self.assertEqual(os.listdir(self.cache), [])
        self.lock.__exit__.assert_called_once()
        self.response.raise_for_status.side_effect = None
        result = self.helpers['download_if_needed']('fixture.bin')
        self.assertEqual(Path(result).read_bytes(), b'abc')
        self.assertEqual(os.listdir(self.cache), ['fixture.bin'])
        self.assertEqual(self.lock.__exit__.call_count, 2)

    def test_connection_failure_leaves_no_cache_or_temp(self):
        self.requests.get.side_effect = OSError('connection failed')
        with self.assertRaises(OSError):
            self.helpers['download_if_needed']('fixture.bin')
        self.assertEqual(os.listdir(self.cache), [])
        self.lock.__exit__.assert_called_once()

    def test_stream_failure_leaves_no_cache_or_temp(self):
        def broken():
            yield b'partial'
            raise OSError('interrupted')
        self.response.iter_content.return_value = broken()
        with self.assertRaises(OSError):
            self.helpers['download_if_needed']('fixture.bin')
        self.assertEqual(os.listdir(self.cache), [])
        self.lock.__exit__.assert_called_once()
        self.response.__exit__.assert_called_once()
        self.progress.close.assert_called_once_with()

    def test_zip_success_preserves_exact_fixture_bytes(self):
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, 'w') as packed:
            packed.writestr('data.txt', b'exact fixture bytes')
        payload = archive.getvalue()
        self.response.iter_content.return_value = [payload[:20], payload[20:]]
        self.response.headers = {'Content-Length': str(len(payload))}
        result = self.helpers['download_if_needed']('fixture')
        self.assertEqual((Path(result) / 'data.txt').read_bytes(),
                         b'exact fixture bytes')
        self.assertEqual(os.listdir(self.cache), ['fixture'])
        self.lock.__exit__.assert_called_once()

    def test_cache_hit_performs_no_request(self):
        cached = Path(self.cache) / 'fixture.bin'
        cached.write_bytes(b'cached')
        result = self.helpers['download_if_needed']('fixture.bin')
        self.assertEqual(result, str(cached))
        self.requests.get.assert_not_called()
        self.lock.__exit__.assert_called_once()


if __name__ == '__main__':
    unittest.main()
