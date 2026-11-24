"""混合检索的批量向量化：重写出的多条问法只调一次 embedding。

覆盖：
- N 条问法只调用一次 get_embedding，向量按问法分发给各知识库；
- 同一问法在多个知识库间复用同一个向量（向量化次数与知识库数解耦）；
- 重复问法只向量化一次；
- 向量化失败或返回条数不符时降级为 ES-only，不触发二次 embedding；
- ES 关闭时只查向量库的路径同样只向量化一次。
"""
import asyncio

import pytest

from agentchat.services.rag import retrieval as retrieval_module
from agentchat.services.rag.retrieval import MixRetrival


def run(coro):
    return asyncio.run(coro)


class FakeVectorClient:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    async def _record(self, field, query, collection_name, query_embedding):
        if self.fail:
            raise RuntimeError("vector store down")
        self.calls.append({"field": field, "query": query, "kb": collection_name,
                           "embedding": query_embedding})
        return []

    async def search(self, query, collection_name, top_k=10, query_embedding=None):
        return await self._record("content", query, collection_name, query_embedding)

    async def search_summary(self, query, collection_name, top_k=10, query_embedding=None):
        return await self._record("summary", query, collection_name, query_embedding)


class FakeESClient:
    def __init__(self):
        self.calls = []

    async def search_documents(self, query, index_name):
        self.calls.append(("content", query, index_name))
        return []

    async def search_documents_summary(self, query, index_name):
        self.calls.append(("summary", query, index_name))
        return []


@pytest.fixture
def env(monkeypatch):
    """替身环境：记录 embedding 调用、向量检索与 ES 检索。"""
    state = {"embedding_inputs": [], "embedding_error": None, "vectors": {"a": [1.0], "b": [2.0], "c": [3.0]}}
    vector = FakeVectorClient()
    es = FakeESClient()

    async def fake_embedding(queries):
        state["embedding_inputs"].append(list(queries))
        if state["embedding_error"]:
            raise state["embedding_error"]
        return [state["vectors"][query] for query in queries]

    monkeypatch.setattr(retrieval_module, "get_embedding", fake_embedding)
    monkeypatch.setattr(retrieval_module, "milvus_client", vector)
    monkeypatch.setattr(retrieval_module, "es_client", es)
    return state, vector, es


def test_rewritten_queries_use_single_embedding_call(env):
    state, vector, es = env

    es_documents, vector_documents = run(
        MixRetrival.mix_retrival_documents(["a", "b", "c"], ["kb1"], "content")
    )

    assert state["embedding_inputs"] == [["a", "b", "c"]], "三条问法应合并为一次 embedding 调用"
    assert len(vector.calls) == 3
    assert [call["embedding"] for call in vector.calls] == [[1.0], [2.0], [3.0]]
    assert len(es.calls) == 3, "ES 仍按问法逐条检索"
    assert es_documents == [] and vector_documents == []


def test_embedding_reused_across_knowledge_bases(env):
    state, vector, es = env

    run(MixRetrival.mix_retrival_documents(["a", "b"], ["kb1", "kb2"], "summary"))

    assert state["embedding_inputs"] == [["a", "b"]], "多个知识库不应重复向量化"
    assert len(vector.calls) == 4, "2 条问法 × 2 个知识库"
    assert {call["kb"] for call in vector.calls} == {"kb1", "kb2"}
    assert {call["field"] for call in vector.calls} == {"summary"}


def test_duplicate_queries_embedded_once(env):
    state, vector, _ = env

    run(MixRetrival.mix_retrival_documents(["a", "b", "a"], ["kb1"], "content"))

    assert state["embedding_inputs"] == [["a", "b"]], "重复问法只向量化一次"
    assert [call["embedding"] for call in vector.calls] == [[1.0], [2.0], [1.0]]


def test_embedding_failure_degrades_to_es_only(env):
    state, vector, es = env
    state["embedding_error"] = RuntimeError("embedding down")

    es_documents, vector_documents = run(
        MixRetrival.mix_retrival_documents(["a", "b"], ["kb1"], "content")
    )

    assert state["embedding_inputs"] == [["a", "b"]], "失败时也只尝试一次，不再逐条重试"
    assert vector.calls == [], "向量化失败后不应再调向量库（避免二次 embedding）"
    assert len(es.calls) == 2, "ES 结果仍要返回，检索不整体失败"
    assert es_documents == [] and vector_documents == []


def test_vector_count_mismatch_degrades_to_es_only(env, monkeypatch):
    state, vector, _es = env

    async def short_embedding(queries):
        state["embedding_inputs"].append(list(queries))
        return [[1.0]]

    monkeypatch.setattr(retrieval_module, "get_embedding", short_embedding)

    _, vector_documents = run(MixRetrival.mix_retrival_documents(["a", "b"], ["kb1"], "content"))

    assert vector_documents == []
    assert vector.calls == []


def test_es_disabled_path_embeds_once(env):
    state, vector, _ = env

    run(MixRetrival.retrival_milvus_documents(["a", "b", "c"], ["kb1"], "content"))

    assert state["embedding_inputs"] == [["a", "b", "c"]]
    assert len(vector.calls) == 3
