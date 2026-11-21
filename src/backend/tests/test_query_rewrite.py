"""查询重写：配置开关与输出结构校验。"""
import asyncio
from types import SimpleNamespace

from agentchat.schemas.common import Rag
from agentchat.services.rag.config_flags import should_query_rewrite
from agentchat.settings import app_settings

# query_write 在 import 时就会实例化 QueryRewrite（读 multi_models），
# 无配置环境下补一个占位，避免 import 期报错。
if app_settings.multi_models is None:
    app_settings.multi_models = SimpleNamespace(
        chat_model=SimpleNamespace(
            model_name="test-model",
            api_key="test-only",
            base_url="http://127.0.0.1:1/v1",
        )
    )

from agentchat.services.rewrite import query_write  # noqa: E402


def test_default_enabled():
    assert Rag().enable_query_rewrite is True


def test_flag_reads_config(monkeypatch):
    monkeypatch.setattr(app_settings, "rag", Rag(enable_query_rewrite=False))
    assert should_query_rewrite(None) is False

    monkeypatch.setattr(app_settings, "rag", Rag(enable_query_rewrite=True))
    assert should_query_rewrite(None) is True


def test_explicit_flag_overrides_config(monkeypatch):
    monkeypatch.setattr(app_settings, "rag", Rag(enable_query_rewrite=True))
    assert should_query_rewrite(False) is False

    monkeypatch.setattr(app_settings, "rag", Rag(enable_query_rewrite=False))
    assert should_query_rewrite(True) is True


def test_missing_rag_section_defaults_to_enabled(monkeypatch):
    # 简易模式/CI 没有 rag 段时保持原来的“开启重写”行为
    monkeypatch.setattr(app_settings, "rag", None)
    assert should_query_rewrite(None) is True


def test_normalize_list_of_strings():
    assert query_write.QueryRewrite._normalize_queries(["a", " b "], "orig") == ["a", "b"]


def test_normalize_dict_variations():
    payload = {"original_query": "q", "variations": ["改法一", "改法二"]}
    assert query_write.QueryRewrite._normalize_queries(payload, "q") == ["改法一", "改法二"]


def test_normalize_malformed_payload():
    assert query_write.QueryRewrite._normalize_queries({"foo": 1}, "orig") == ["orig"]
    assert query_write.QueryRewrite._normalize_queries("not a list", "orig") == ["orig"]
    assert query_write.QueryRewrite._normalize_queries(["", "  "], "orig") == ["orig"]


def test_rewrite_accepts_dict_output(monkeypatch):
    class FakeClient:
        def invoke(self, messages):
            return SimpleNamespace(
                content='{"original_query": "q", "variations": ["改法一", "改法二"]}'
            )

    monkeypatch.setattr(query_write.query_rewriter, "client", FakeClient())
    assert asyncio.run(query_write.query_rewriter.rewrite("q")) == ["改法一", "改法二"]


def test_rewrite_falls_back_on_bad_json(monkeypatch):
    class FakeClient:
        def invoke(self, messages):
            return SimpleNamespace(content="不是 JSON")

    monkeypatch.setattr(query_write.query_rewriter, "client", FakeClient())
    assert asyncio.run(query_write.query_rewriter.rewrite("q")) == ["q"]
