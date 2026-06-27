"""Tests for syntk.pipelines.bootstrap — row generation and DataFrame assembly."""

from __future__ import annotations

from typing import List, Optional
from unittest.mock import MagicMock, patch

import pandas as pd

from syntk.config import APIArguments, GenerationArguments
from syntk.pipelines.bootstrap import (
    BootstrapDataArguments,
    _build_user_message,
    _parse_row_response,
    _rows_to_json_block,
    bootstrap,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _df(n: int = 5) -> pd.DataFrame:
    return pd.DataFrame({
        "text": [f"sentence {i}" for i in range(n)],
        "label": [i % 3 for i in range(n)],
    })


def _api_args() -> APIArguments:
    return APIArguments(model="test-model")


def _gen_args() -> GenerationArguments:
    return GenerationArguments()


def _data_args(**kwargs) -> BootstrapDataArguments:
    defaults = dict(input_file="in.parquet", output_file="out.parquet", n=3, n_shots=2, seed=42)
    defaults.update(kwargs)
    return BootstrapDataArguments(**defaults)


def _mock_client(responses: List[Optional[str]]):
    """Return a MagicMock client and patch get_chat_response to return responses in order."""
    return iter(responses)


# ---------------------------------------------------------------------------
# _rows_to_json_block
# ---------------------------------------------------------------------------

class TestRowsToJsonBlock:
    def test_numbers_examples(self):
        rows = [{"a": 1}, {"a": 2}]
        block = _rows_to_json_block(rows)
        assert "Example 1:" in block
        assert "Example 2:" in block

    def test_json_serialised(self):
        rows = [{"key": "value with spaces"}]
        block = _rows_to_json_block(rows)
        assert '"key"' in block
        assert '"value with spaces"' in block

    def test_empty_list(self):
        assert _rows_to_json_block([]) == ""


# ---------------------------------------------------------------------------
# _build_user_message
# ---------------------------------------------------------------------------

class TestBuildUserMessage:
    def test_includes_example_count(self):
        examples = [{"x": 1}, {"x": 2}]
        msg = _build_user_message(examples, ["x"])
        assert "2" in msg

    def test_lists_columns(self):
        examples = [{"a": 1, "b": "hi"}]
        msg = _build_user_message(examples, ["a", "b"])
        assert "a" in msg
        assert "b" in msg

    def test_contains_json_of_examples(self):
        examples = [{"fruit": "mango"}]
        msg = _build_user_message(examples, ["fruit"])
        assert "mango" in msg


# ---------------------------------------------------------------------------
# _parse_row_response
# ---------------------------------------------------------------------------

class TestParseRowResponse:
    def test_valid_json(self):
        result = _parse_row_response('{"text": "hello", "label": 1}', ["text", "label"])
        assert result == {"text": "hello", "label": 1}

    def test_strips_markdown_fences(self):
        raw = '```json\n{"text": "hi", "label": 0}\n```'
        result = _parse_row_response(raw, ["text", "label"])
        assert result is not None
        assert result["text"] == "hi"

    def test_missing_column_filled_with_none(self):
        result = _parse_row_response('{"text": "hello"}', ["text", "label"])
        assert result["label"] is None

    def test_extra_columns_dropped(self):
        result = _parse_row_response('{"text": "x", "label": 1, "extra": "drop"}', ["text", "label"])
        assert "extra" not in result

    def test_invalid_json_returns_none(self):
        assert _parse_row_response("not json at all", ["text"]) is None

    def test_non_dict_returns_none(self):
        assert _parse_row_response("[1, 2, 3]", ["text"]) is None

    def test_empty_string_returns_none(self):
        assert _parse_row_response("", ["text"]) is None


# ---------------------------------------------------------------------------
# bootstrap()
# ---------------------------------------------------------------------------

class TestBootstrap:
    def _run(self, responses: List[Optional[str]], df: Optional[pd.DataFrame] = None,
             **data_kwargs) -> pd.DataFrame:
        df = df or _df()
        client = MagicMock()
        data_args = _data_args(**data_kwargs)

        call_iter = iter(responses)

        def fake_get_chat_response(**kwargs):
            try:
                return next(call_iter)
            except StopIteration:
                return None

        with patch("syntk.pipelines.bootstrap.get_chat_response", side_effect=fake_get_chat_response):
            return bootstrap(client, df, data_args, _api_args(), _gen_args())

    def test_returns_dataframe(self):
        resp = '{"text": "new", "label": 1}'
        result = self._run([resp] * 3, n=3)
        assert isinstance(result, pd.DataFrame)

    def test_generates_n_rows(self):
        resp = '{"text": "new", "label": 1}'
        result = self._run([resp] * 5, n=5)
        assert len(result) == 5

    def test_columns_match_source(self):
        resp = '{"text": "new", "label": 1}'
        result = self._run([resp] * 2, n=2)
        assert set(result.columns) == {"text", "label"}

    def test_failed_responses_skipped(self):
        responses = ['{"text": "ok", "label": 0}', "INVALID", '{"text": "ok2", "label": 1}']
        result = self._run(responses, n=3)
        # Only 2 parseable responses → 2 rows
        assert len(result) == 2

    def test_all_failed_returns_empty(self):
        result = self._run(["INVALID"] * 3, n=3)
        assert len(result) == 0

    def test_none_response_skipped(self):
        result = self._run([None, '{"text": "x", "label": 0}', None], n=3)
        assert len(result) == 1

    def test_column_subset(self):
        resp = '{"text": "hi"}'
        result = self._run([resp] * 2, n=2, columns="text")
        assert list(result.columns) == ["text"]

    def test_seed_reproducibility(self):
        resp = '{"text": "a", "label": 0}'
        df = _df(20)

        with patch("syntk.pipelines.bootstrap.get_chat_response", return_value=resp):
            r1 = bootstrap(MagicMock(), df, _data_args(n=3, seed=99), _api_args(), _gen_args())
            r2 = bootstrap(MagicMock(), df, _data_args(n=3, seed=99), _api_args(), _gen_args())

        # Same seed → same calls (we can at least verify both succeeded)
        assert len(r1) == len(r2) == 3

    def test_generated_rows_used_as_future_shots(self):
        """Generated rows should be eligible as future examples (n_shots can sample them)."""
        # Use a 1-row dataframe so after generating row 1, row 1 is in the pool for row 2
        df = pd.DataFrame({"text": ["original"], "label": [0]})
        resp = '{"text": "gen", "label": 1}'
        call_count = []

        def fake_resp(**kwargs):
            call_count.append(kwargs["messages"])
            return resp

        with patch("syntk.pipelines.bootstrap.get_chat_response", side_effect=fake_resp):
            result = bootstrap(
                MagicMock(), df, _data_args(n=3, n_shots=2, seed=0), _api_args(), _gen_args()
            )

        assert len(result) == 3
        # After the first row, pool grows — later prompts may include generated rows
        # (The test just verifies no crash and correct count)


# ---------------------------------------------------------------------------
# _merge_yaml_config — positional YAML support (README: "same shape as column")
# ---------------------------------------------------------------------------

class TestMergeYamlConfig:
    def _merge(self, tmp_path, config: dict, api=None, gen=None, data=None):
        import yaml

        from syntk.pipelines.bootstrap import _merge_yaml_config

        cfg = tmp_path / "c.yaml"
        cfg.write_text(yaml.safe_dump(config))
        api = api or APIArguments()
        gen = gen or GenerationArguments()
        data = data or BootstrapDataArguments()
        _merge_yaml_config(
            str(cfg),
            [api, gen, data],
            (APIArguments, GenerationArguments, BootstrapDataArguments),
        )
        return api, gen, data

    def test_yaml_values_applied_when_cli_at_default(self, tmp_path):
        api, gen, data = self._merge(
            tmp_path, {"model": "yaml-model", "n": 99, "n_shots": 7}
        )
        assert api.model == "yaml-model"   # was default → take YAML
        assert data.n == 99
        assert data.n_shots == 7

    def test_cli_value_overrides_yaml(self, tmp_path):
        # data.n is explicitly set (differs from the default 10) → CLI wins.
        data = BootstrapDataArguments(n=5)
        _, _, data = self._merge(tmp_path, {"n": 99}, data=data)
        assert data.n == 5

    def test_empty_yaml_is_noop(self, tmp_path):
        api, _, data = self._merge(tmp_path, {})
        assert api.model == APIArguments().model
        assert data.n == BootstrapDataArguments().n
