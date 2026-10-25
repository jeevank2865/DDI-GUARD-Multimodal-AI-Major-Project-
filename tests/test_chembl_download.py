from __future__ import annotations

import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ddi_guard.chembl_download import download_archive


class ChemblDownloadTests(unittest.TestCase):
    class Response:
        def __init__(self, url, status, headers, body=b""):
            self.url, self.status, self.headers = url, status, headers
            self.body = io.BytesIO(body)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.body.close()

        def geturl(self):
            return self.url

        def getcode(self):
            return self.status

        def read(self, size=-1):
            return self.body.read(size)

    def fake_request(self, url, payload, *, ignore_range=False):
        def request(requested_url, method, headers=None, timeout=60):
            if method == "HEAD":
                return self.Response(url, 200, {"Content-Length": str(len(payload)), "Accept-Ranges": "bytes"})
            range_header = (headers or {}).get("Range")
            if range_header and not ignore_range:
                offset = int(range_header.removeprefix("bytes=").removesuffix("-"))
                return self.Response(url, 206,
                                     {"Content-Range": f"bytes {offset}-{len(payload) - 1}/{len(payload)}",
                                      "Content-Length": str(len(payload) - offset)}, payload[offset:])
            return self.Response(url, 200, {"Content-Length": str(len(payload))}, payload)
        return request

    def test_resumes_only_after_range_validation_then_atomically_installs_verified_file(self):
        payload = b"a realistic local archive fixture" * 100
        url = "http://127.0.0.1:1/archive.tar.gz"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            destination, partial = root / "archive.tar.gz", root / "archive.partial"
            partial.write_bytes(payload[:57])
            with patch("ddi_guard.chembl_download._request", self.fake_request(url, payload)):
                result = download_archive(url, destination, expected_size=len(payload),
                                          expected_sha256=hashlib.sha256(payload).hexdigest(),
                                          partial_path=partial, chunk_size=31, official_only=False)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["resumed_from_bytes"], 57)
            self.assertEqual(result["downloaded_this_run_bytes"], len(payload) - 57)
            self.assertEqual(destination.read_bytes(), payload)
            self.assertFalse(partial.exists())

    def test_refuses_server_that_ignores_range_without_touching_partial(self):
        payload = b"archive contents"
        url = "http://127.0.0.1:1/archive.tar.gz"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            destination, partial = root / "archive.tar.gz", root / "archive.partial"
            partial.write_bytes(payload[:4])
            with patch("ddi_guard.chembl_download._request",
                       self.fake_request(url, payload, ignore_range=True)):
                with self.assertRaisesRegex(RuntimeError, "did not honor resume offset"):
                    download_archive(url, destination, expected_size=len(payload),
                                     expected_sha256=hashlib.sha256(payload).hexdigest(),
                                     partial_path=partial, official_only=False)
            self.assertEqual(partial.read_bytes(), payload[:4])
            self.assertFalse(destination.exists())

    def test_retries_read_timeout_from_the_exact_saved_offset(self):
        payload = b"large enough to require several reads" * 8
        url = "http://127.0.0.1:1/archive.tar.gz"
        requests = []

        class TimeoutOnceResponse(ChemblDownloadTests.Response):
            def __init__(self, url, status, headers, body=b""):
                super().__init__(url, status, headers, body)
                self.read_calls = 0

            def read(self, size=-1):
                self.read_calls += 1
                if self.read_calls == 2:
                    raise TimeoutError("temporary network stall")
                return self.body.read(min(size, 11))

        def request(requested_url, method, headers=None, timeout=60):
            if method == "HEAD":
                return self.Response(url, 200, {"Content-Length": str(len(payload)),
                                                 "Accept-Ranges": "bytes"})
            range_header = (headers or {}).get("Range")
            offset = int(range_header.removeprefix("bytes=").removesuffix("-")) if range_header else 0
            requests.append(offset)
            remaining = payload[offset:]
            status = 206 if offset else 200
            response_headers = {"Content-Length": str(len(remaining))}
            if offset:
                response_headers["Content-Range"] = f"bytes {offset}-{len(payload) - 1}/{len(payload)}"
            response_type = TimeoutOnceResponse if len(requests) == 1 else self.Response
            return response_type(url, status, response_headers, remaining)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            destination, partial = root / "archive.tar.gz", root / "archive.partial"
            partial.write_bytes(payload[:5])
            with patch("ddi_guard.chembl_download._request", request):
                result = download_archive(url, destination, expected_size=len(payload),
                                          expected_sha256=hashlib.sha256(payload).hexdigest(),
                                          partial_path=partial, chunk_size=13,
                                          retry_backoff_seconds=0, official_only=False)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(requests, [5, 16])
            self.assertEqual(destination.read_bytes(), payload)

    def test_lock_and_official_host_guards(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            partial = root / "archive.part"
            partial.with_name(partial.name + ".download.lock").touch()
            with self.assertRaisesRegex(RuntimeError, "holds the lock"):
                download_archive("https://ftp.ebi.ac.uk/archive", root / "archive",
                                 partial_path=partial)
            with self.assertRaisesRegex(ValueError, "non-official"):
                download_archive("https://example.com/archive", root / "archive")


if __name__ == "__main__":
    unittest.main()
