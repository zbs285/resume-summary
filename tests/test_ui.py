from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from resume_summarizer.ui import _unique_upload_name, create_app


class UiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = create_app(prefill_path="/tmp/example resumes")
        self.app.testing = True
        self.client = self.app.test_client()

    def test_home_contains_all_three_input_methods(self) -> None:
        response = self.client.get("/")
        page = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("把 PDF 简历或文件夹拖到这里", page)
        self.assertIn("webkitdirectory", page)
        self.assertIn("/tmp/example resumes", page)
        self.assertIn("退出工具", page)

    def test_api_rejects_missing_local_token(self) -> None:
        response = self.client.post("/api/jobs")
        self.assertEqual(response.status_code, 403)

    def test_shutdown_rejects_missing_local_token(self) -> None:
        response = self.client.post("/api/shutdown")
        self.assertEqual(response.status_code, 403)

    def test_empty_job_returns_actionable_error(self) -> None:
        token = self.app.config["RESUME_TOKEN"]
        response = self.client.post(
            "/api/jobs",
            headers={"X-Resume-Token": token},
            data={},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("没有找到 PDF", response.get_json()["error"])

    def test_upload_name_preserves_chinese_filename(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _unique_upload_name(Path(temp_dir), "简历-王栋.pdf")
        self.assertEqual("简历-王栋.pdf", path.name)


if __name__ == "__main__":
    unittest.main()
