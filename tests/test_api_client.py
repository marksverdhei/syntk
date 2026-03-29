"""Tests for syntk.api_client.save_raw_api_call."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from syntk.api_client import save_raw_api_call


# ---------------------------------------------------------------------------
# save_raw_api_call
# ---------------------------------------------------------------------------

class TestSaveRawApiCall:
    def test_writes_jsonl_entry(self, tmp_path):
        path = str(tmp_path / "raw.jsonl")
        result = {"content": "hello", "raw": {"request": {"model": "gpt-4"}, "response": {"id": "abc"}}}
        save_raw_api_call(path, row_index=0, result=result)
        lines = open(path).readlines()
        assert len(lines) == 1

    def test_entry_is_valid_json(self, tmp_path):
        path = str(tmp_path / "raw.jsonl")
        result = {"content": "hi", "raw": {"request": {}, "response": {}}}
        save_raw_api_call(path, row_index=5, result=result)
        record = json.loads(open(path).read().strip())
        assert isinstance(record, dict)

    def test_entry_contains_row_index(self, tmp_path):
        path = str(tmp_path / "raw.jsonl")
        result = {"raw": {"request": {}, "response": {}}}
        save_raw_api_call(path, row_index=42, result=result)
        record = json.loads(open(path).read().strip())
        assert record["row_index"] == 42

    def test_entry_contains_request_and_response(self, tmp_path):
        path = str(tmp_path / "raw.jsonl")
        req = {"model": "test", "messages": []}
        resp = {"choices": [{"message": {"content": "ok"}}]}
        result = {"raw": {"request": req, "response": resp}}
        save_raw_api_call(path, row_index=0, result=result)
        record = json.loads(open(path).read().strip())
        assert record["request"] == req
        assert record["response"] == resp

    def test_entry_contains_timestamp(self, tmp_path):
        path = str(tmp_path / "raw.jsonl")
        result = {"raw": {"request": {}, "response": {}}}
        save_raw_api_call(path, row_index=0, result=result)
        record = json.loads(open(path).read().strip())
        assert "timestamp" in record
        assert isinstance(record["timestamp"], (int, float))

    def test_appends_multiple_entries(self, tmp_path):
        path = str(tmp_path / "raw.jsonl")
        result = {"raw": {"request": {}, "response": {}}}
        save_raw_api_call(path, row_index=0, result=result)
        save_raw_api_call(path, row_index=1, result=result)
        save_raw_api_call(path, row_index=2, result=result)
        lines = open(path).readlines()
        assert len(lines) == 3

    def test_each_appended_entry_is_valid_json(self, tmp_path):
        path = str(tmp_path / "raw.jsonl")
        result = {"raw": {"request": {}, "response": {}}}
        for i in range(5):
            save_raw_api_call(path, row_index=i, result=result)
        for line in open(path):
            record = json.loads(line.strip())
            assert "row_index" in record

    def test_no_raw_key_does_nothing(self, tmp_path):
        """If result has no 'raw' key, the file should not be created."""
        path = str(tmp_path / "raw.jsonl")
        result = {"content": "hello", "stop_reason": "stop"}
        save_raw_api_call(path, row_index=0, result=result)
        assert not os.path.exists(path)

    def test_creates_file_if_not_exists(self, tmp_path):
        path = str(tmp_path / "subdir" / "raw.jsonl")
        result = {"raw": {"request": {}, "response": {}}}
        # Parent directory doesn't exist — should raise or create?
        # The function uses open(file_path, "a") which will fail if parent is missing.
        # Verify it raises rather than silently doing nothing.
        with pytest.raises(FileNotFoundError):
            save_raw_api_call(path, row_index=0, result=result)

    def test_row_indices_preserved_in_order(self, tmp_path):
        path = str(tmp_path / "raw.jsonl")
        result = {"raw": {"request": {}, "response": {}}}
        for i in [10, 20, 30]:
            save_raw_api_call(path, row_index=i, result=result)
        records = [json.loads(l) for l in open(path)]
        assert [r["row_index"] for r in records] == [10, 20, 30]
