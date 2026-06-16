"""
Tests for the GET /result2 integration endpoint.

Run with:
    python -m pytest tests/test_result2.py -v
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import api.main as main_module
from api.main import RESULT_FILE, _SAMPLE_RESULT, app

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_result_file(tmp_path, monkeypatch):
    """Redirect RESULT_FILE to a temp path so tests never touch output/."""
    tmp_file = tmp_path / "result2.json"
    monkeypatch.setattr(main_module, "RESULT_FILE", tmp_file)
    monkeypatch.setattr(main_module, "OUTPUT_DIR", tmp_path)
    yield tmp_file


# ── /result2 ──────────────────────────────────────────────────────────────────

def test_result2_returns_sample_when_no_file_exists(clean_result_file):
    """Before any scan, /result2 must return the eight-field sample so Team 4's
    integration works immediately — even before the model is loaded."""
    assert not clean_result_file.exists()
    resp = client.get("/result2")
    assert resp.status_code == 200
    data = resp.json()
    assert data == _SAMPLE_RESULT


def test_result2_returns_stored_file_when_present(clean_result_file):
    """After a scan has run, /result2 must return whatever was persisted."""
    stored = {
        "name": "Sushi",
        "sodium_mg": 410.0,
        "sugar_g": 5.0,
        "carbs_g": 28.0,
        "calories": 150.0,
        "fat_g": 2.0,
        "confidence": 0.91,
        "confidence_tier": "high",
    }
    clean_result_file.write_text(json.dumps(stored))
    resp = client.get("/result2")
    assert resp.status_code == 200
    assert resp.json() == stored


def test_result2_falls_back_to_sample_when_file_is_corrupt(clean_result_file):
    """A corrupt JSON file must not crash the endpoint — fall back to sample."""
    clean_result_file.write_text("not valid json{{{")
    resp = client.get("/result2")
    assert resp.status_code == 200
    assert resp.json() == _SAMPLE_RESULT


def test_result2_eight_fields_present(clean_result_file):
    """The contract must always contain exactly the eight agreed-upon keys."""
    expected_keys = {
        "name", "sodium_mg", "sugar_g", "carbs_g",
        "calories", "fat_g", "confidence", "confidence_tier",
    }
    resp = client.get("/result2")
    assert resp.status_code == 200
    assert set(resp.json().keys()) == expected_keys


# ── _store_result ─────────────────────────────────────────────────────────────

def test_store_result_writes_eight_fields(clean_result_file):
    """_store_result must strip internal fields and write only the contract."""
    from api.main import _store_result

    full_payload = {
        "name": "Tacos",
        "sodium_mg": 320.0,
        "sugar_g": 2.0,
        "carbs_g": 22.0,
        "calories": 210.0,
        "fat_g": 9.0,
        "confidence": 0.75,
        "confidence_tier": "high",
        # internal fields that must be stripped:
        "source": "mobilenetv3_food101",
        "food_unrecognized": None,
        "top_3_candidates": None,
    }
    _store_result(full_payload)

    assert clean_result_file.exists()
    saved = json.loads(clean_result_file.read_text())
    assert "source" not in saved
    assert "food_unrecognized" not in saved
    assert "top_3_candidates" not in saved
    assert saved["name"] == "Tacos"
    assert saved["confidence_tier"] == "high"


def test_store_result_then_result2_serves_it(clean_result_file):
    """Full round-trip: store a result, then verify /result2 returns it."""
    from api.main import _store_result

    payload = {
        "name": "Ramen",
        "sodium_mg": 800.0,
        "sugar_g": 4.0,
        "carbs_g": 55.0,
        "calories": 430.0,
        "fat_g": 14.0,
        "confidence": 0.68,
        "confidence_tier": "high",
    }
    _store_result(payload)

    resp = client.get("/result2")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "Ramen"
    assert data["calories"] == 430.0
    assert data["confidence_tier"] == "high"


# ── /health ───────────────────────────────────────────────────────────────────

def test_health_endpoint():
    """/health must be reachable and report status ok."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
