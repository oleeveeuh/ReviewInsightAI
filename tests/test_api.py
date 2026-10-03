"""API tests: health, tools, error handling, CORS, request limits.

All offline — /analyze without an API key must return a clear 503, and no
endpoint may leak internal exception details.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import MAX_REVIEW_CHARS, app


@pytest.fixture
def client():
    with TestClient(app) as c:  # triggers lifespan -> agent initialized
        yield c


def test_health_ok(client):
    r = client.get('/health')
    assert r.status_code == 200
    body = r.json()
    assert body['status'] == 'healthy'
    assert body['agent'] == 'ready'
    assert body['llm_configured'] is False  # conftest strips the key


def test_root_lists_docs(client):
    r = client.get('/')
    assert r.status_code == 200
    assert r.json()['docs'] == '/docs'


def test_tools_endpoint_lists_registry(client):
    r = client.get('/tools')
    assert r.status_code == 200
    body = r.json()
    assert body['count'] == 3
    names = {t['name'] for t in body['tools']}
    assert {'retrieve_similar_reviews', 'analyze_sentiment',
            'detect_anomaly'} <= names


def test_analyze_without_llm_returns_clear_503(client):
    r = client.post('/analyze', json={'text': 'overtime is exhausting'})
    assert r.status_code == 503
    assert 'OPENAI_API_KEY' in r.json()['detail'] or 'LLM' in r.json()['detail']


def test_analyze_never_leaks_exception_details(client, monkeypatch):
    """A generic failure must return a sanitized message."""
    from api import main as api_main

    class Boom:
        def analyze(self, *a, **kw):
            raise ValueError('SECRET internal path /Users/x/secret.db')

    monkeypatch.setattr(api_main, 'agent', Boom())
    r = client.post('/analyze', json={'text': 'anything'})
    assert r.status_code == 500
    assert 'SECRET' not in r.json()['detail']
    assert r.json()['detail'] == 'Analysis failed. See server logs for details.'


def test_memory_stats_endpoint(client):
    r = client.get('/memory/stats')
    assert r.status_code == 200
    body = r.json()
    assert 'total_analyses' in body and 'theme_distribution' in body


def test_oversized_review_rejected(client):
    r = client.post('/analyze', json={'text': 'x' * (MAX_REVIEW_CHARS + 1)})
    assert r.status_code == 422


def test_blank_review_rejected(client):
    assert client.post('/analyze', json={'text': ''}).status_code == 422


def test_cors_disallows_unconfigured_origins(client):
    r = client.get('/health', headers={'Origin': 'https://evil.example.com'})
    assert r.status_code == 200
    assert 'access-control-allow-origin' not in {k.lower() for k in r.headers}


def test_cors_origin_parser_rejects_wildcard():
    """'*' must never be honored (wildcard + credentials is unsafe)."""
    from api.main import parse_allowed_origins
    assert parse_allowed_origins('') == []
    assert parse_allowed_origins(None) == []
    assert parse_allowed_origins('*') == []
    assert parse_allowed_origins(' http://a.com, http://b.com ') == \
        ['http://a.com', 'http://b.com']
    assert parse_allowed_origins('*,http://a.com') == ['http://a.com']
