"""Agent tool registry tests (offline; no API key, no network)."""


from agent.tools import (AnalyzeSentimentTool, DetectAnomalyTool,
                         RetrieveSimilarTool, ToolRegistry)


def test_registry_registers_core_tools():
    registry = ToolRegistry()
    assert set(registry.tools) == {'retrieve_similar_reviews', 'analyze_sentiment',
                                   'detect_anomaly'}
    specs = registry.list_tools()
    assert len(specs) == 3
    for spec in specs:
        assert spec['name'] and spec['description'] and 'parameters' in spec


def test_unknown_tool_returns_error_not_exception():
    registry = ToolRegistry()
    out = registry.execute('definitely_not_a_tool')
    assert 'error' in out
    assert 'definitely_not_a_tool' in out['error']


def test_registry_survives_tool_exception():
    registry = ToolRegistry()
    # detect_anomaly requires themes/sentiment; calling it with bad args
    # must produce an error dict, not raise
    out = registry.execute('detect_anomaly', themes='not-a-list', sentiment=None)
    assert isinstance(out, dict)


def test_retrieve_similar_uses_fixture_store(fixture_reviews_path):
    tool = RetrieveSimilarTool()
    out = tool.execute(query_text="mandatory overtime every week", k=3)
    assert out['count'] <= 3
    assert len(out['results']) >= 1
    assert all('similarity' in r and 'review_id' in r for r in out['results'])


def test_detect_anomaly_rules():
    tool = DetectAnomalyTool()
    out = tool.execute(themes=['safety', 'management'], sentiment=2)
    assert out['is_anomalous'] is True
    assert out['critical_combinations']

    out_ok = tool.execute(themes=['pay_benefits'], sentiment=4)
    assert out_ok['is_anomalous'] is False
    assert out_ok['critical_combinations'] is None


def test_detect_anomaly_flags_unknown_theme_as_rare():
    tool = DetectAnomalyTool()
    out = tool.execute(themes=['totally_unknown_theme_xyz'], sentiment=3)
    assert out['rare_themes'] == ['totally_unknown_theme_xyz']


def test_analyze_sentiment_without_key_returns_actionable_error(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    tool = AnalyzeSentimentTool()
    out = tool.execute(review_text="some review text")
    assert 'error' in out
    # The message must tell the user what to do, not leak a traceback
    assert 'OPENAI_API_KEY' in out['error'] or 'openai' in out['error'].lower()
