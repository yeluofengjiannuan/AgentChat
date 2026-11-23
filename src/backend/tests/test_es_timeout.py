"""ES 检索超时与熔断：ES 不可达时不能每条检索都等满超时。

覆盖：
- 多次失败达到阈值后熔断，之后直接返回空、不再请求 ES；
- 单次失败不熔断；成功会重置失败计数；
- 超时预算通过 client.options(request_timeout=...) 传入。
"""
import asyncio
from types import SimpleNamespace

import pytest

from agentchat.schemas.common import Rag
from agentchat.services.rag import es_client as es_module
from agentchat.settings import app_settings


def run(coro):
    return asyncio.run(coro)


class FakeSearch:
    """模拟 elasticsearch.Elasticsearch.search 的行为。"""

    def __init__(self, behavior):
        self.behavior = behavior
        self.calls = 0
        self.last_timeout = None

    def options(self, request_timeout=None):
        self.last_timeout = request_timeout
        return self

    def search(self, index=None, body=None):
        self.calls += 1
        return self.behavior()


def make_client(behavior, monkeypatch, threshold=3):
    fake = FakeSearch(behavior)
    client = object.__new__(es_module.ESClient)
    client.client = fake
    monkeypatch.setattr(app_settings, "rag", Rag(
        enable_elasticsearch=True, es_timeout=0.2, es_cooldown=5.0, es_failure_threshold=threshold,
    ))
    monkeypatch.setattr(es_module, "_unavailable_until", 0.0)
    monkeypatch.setattr(es_module, "_consecutive_failures", 0)
    return client, fake


def test_single_failure_returns_empty_without_breaker(monkeypatch):
    def boom():
        raise ConnectionError("es down")

    client, fake = make_client(boom, monkeypatch)
    result = run(client.search_documents("问题", "kb"))
    assert result == []
    assert fake.calls == 1
    assert not es_module.breaker_open(), "单次失败不应熔断"


def test_breaker_opens_after_threshold_failures(monkeypatch):
    def boom():
        raise ConnectionError("es down")

    client, fake = make_client(boom, monkeypatch, threshold=2)
    for _ in range(2):
        run(client.search_documents("问题", "kb"))
    assert es_module.breaker_open(), "连续失败达到阈值应熔断"

    calls_before = fake.calls
    result = run(client.search_documents("问题", "kb"))
    assert result == []
    assert fake.calls == calls_before, "熔断期不应再请求 ES"


def test_success_resets_failure_counter(monkeypatch):
    outcomes = {"fail": True}

    def behavior():
        if outcomes["fail"]:
            raise ConnectionError("es down")
        return {"hits": {"max_score": 1.0, "hits": []}}

    client, fake = make_client(behavior, monkeypatch, threshold=3)
    run(client.search_documents("问题", "kb"))  # 失败 1 次
    outcomes["fail"] = False
    run(client.search_documents("问题", "kb"))  # 成功，重置计数
    assert es_module._consecutive_failures == 0
    assert not es_module.breaker_open()


def test_timeout_budget_is_passed(monkeypatch):
    def ok():
        return {"hits": {"max_score": 1.0, "hits": []}}

    client, fake = make_client(ok, monkeypatch)
    run(client.search_documents("问题", "kb"))
    assert fake.last_timeout == 0.2


def test_breaker_open_short_circuits_summary(monkeypatch):
    def ok():
        return {"hits": {"max_score": 1.0, "hits": []}}

    client, fake = make_client(ok, monkeypatch)
    monkeypatch.setattr(es_module, "_unavailable_until", __import__("time").monotonic() + 10)

    assert run(client.search_documents_summary("问题", "kb")) == []
    assert fake.calls == 0, "熔断期不应请求 ES"
