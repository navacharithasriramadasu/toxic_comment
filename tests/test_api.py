"""
Tests for Toxic Comment Detection API
Run with: pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.main import app

client = TestClient(app)


# ── Health check ─────────────────────────────────────────────────

def test_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "model" in data


# ── Analyze endpoint ─────────────────────────────────────────────

def test_analyze_clean_comment():
    res = client.post("/api/analyze", json={"text": "This is a great tutorial, thank you!"})
    assert res.status_code == 200
    data = res.json()
    assert data["is_toxic"] is False
    assert data["action"] == "ALLOW"
    assert 0.0 <= data["toxicity_score"] <= 1.0
    assert data["confidence"] > 0.5


def test_analyze_toxic_comment():
    res = client.post("/api/analyze", json={"text": "You are a stupid idiot, go die!"})
    assert res.status_code == 200
    data = res.json()
    assert data["is_toxic"] is True
    assert data["action"] in ["WARN", "FLAG", "BLOCK"]


def test_analyze_returns_categories():
    res = client.post("/api/analyze", json={"text": "Hello world"})
    assert res.status_code == 200
    cats = res.json()["categories"]
    for key in ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]:
        assert key in cats
        assert 0.0 <= cats[key] <= 1.0


def test_analyze_empty_text():
    res = client.post("/api/analyze", json={"text": ""})
    assert res.status_code == 400


def test_analyze_too_long():
    res = client.post("/api/analyze", json={"text": "x" * 5001})
    assert res.status_code == 400


# ── Stats endpoint ───────────────────────────────────────────────

def test_stats():
    # Ensure at least one analysis has been done first
    client.post("/api/analyze", json={"text": "test"})
    res = client.get("/api/stats")
    assert res.status_code == 200
    data = res.json()
    assert "total_analyzed" in data
    assert "toxic_detected" in data
    assert "toxic_rate" in data
    assert data["total_analyzed"] >= 1


# ── Preprocessor ─────────────────────────────────────────────────

def test_preprocessor_removes_html():
    from backend.preprocessor import TextPreprocessor
    pp = TextPreprocessor()
    result = pp.clean("<b>Hello</b> <script>bad</script> world")
    assert "<b>" not in result
    assert "<script>" not in result
    assert "Hello" in result


def test_preprocessor_removes_urls():
    from backend.preprocessor import TextPreprocessor
    pp = TextPreprocessor()
    result = pp.clean("Check out https://example.com for more info")
    assert "https://" not in result
    assert "[URL]" in result


def test_preprocessor_normalizes_repeated_chars():
    from backend.preprocessor import TextPreprocessor
    pp = TextPreprocessor()
    result = pp.clean("sooooooo goooood")
    assert len(result) < len("sooooooo goooood")
