"""Tests for __main__.py CLI config parsing."""

from __future__ import annotations

import json

import pytest

from agentica_mcp_runtime.__main__ import _parse_configs


def test_parse_inline_json():
    raw = json.dumps({"test-server": {"command": "echo", "args": ["hi"]}})
    configs = _parse_configs(raw)
    assert "test-server" in configs


def test_parse_file_path(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"file-server": {"command": "echo", "args": ["hello"]}}))
    configs = _parse_configs(str(config_file))
    assert "file-server" in configs


def test_parse_invalid_json_exits():
    with pytest.raises(SystemExit):
        _parse_configs("/nonexistent/path/config.json")


def test_parse_non_object_exits():
    with pytest.raises(SystemExit):
        _parse_configs('["not", "an", "object"]')


def test_parse_multiple_servers():
    raw = json.dumps({
        "server-a": {"command": "echo", "args": ["a"]},
        "server-b": {"command": "echo", "args": ["b"]},
    })
    configs = _parse_configs(raw)
    assert "server-a" in configs
    assert "server-b" in configs


def test_parse_skips_invalid_server(capsys):
    """Invalid server config is skipped but others still load."""
    raw = json.dumps({
        "good-server": {"command": "echo", "args": ["good"]},
    })
    configs = _parse_configs(raw)
    assert "good-server" in configs
