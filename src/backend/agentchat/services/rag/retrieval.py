from typing import Dict, List

from loguru import logger

from agentchat.services.rag.embedding import get_embedding
from agentchat.services.rag.es_client import client as es_client
from agentchat.services.rag.vector_stores import milvus_client


class MixRetrival:
    """ES 关键词检索 + 向量检索的混合召回。

    查询重写会把一个问题扩成多条问法，逐条检索时原实现每条问法都要单独调一次
    embedding。这里改成"先批量向量化、再逐条检索"：同一批问法只调一次 embedding，
    算出的向量按问法复用给各个知识库，网络往返从 N 次降为 1 次。
    """

    @classmethod
    async def embed_queries(cls, queries: List[str]) -> Dict[str, List[float]]:
        """批量向量化，返回 {问法: 向量}；失败或条数不符时返回空表。

        向量化失败时返回空表，调用方会跳过向量检索、退回只用 ES 结果——
        与逐条检索时 embedding 异常被吞掉的效果一致，但只失败一次。
        """
        unique = list(dict.fromkeys(query for query in queries if query))
        if not unique:
            return {}
        try:
            vectors = await get_embedding(unique)
        except Exception as err:  # 超时/熔断/服务异常都按降级处理
            logger.warning(f"批量向量化失败（{type(err).__name__}），本次检索降级为 ES-only")
            return {}
        if not vectors or len(vectors) != len(unique):
            logger.warning(f"向量化返回 {len(vectors) if vectors else 0} 条，与查询数 {len(unique)} 不一致，按 ES-only 处理")
            return {}
        return dict(zip(unique, vectors))

    @classmethod
    async def retrival_vector_documents(cls, query, query_embedding, knowledges_id, search_field):
        """在指定知识库中做向量检索，复用已算好的查询向量。"""
        documents = []
        if query_embedding is None:
            # 向量化没成功：不再触发第二次 embedding，直接交给 ES 结果
            return documents
        for knowledge_id in knowledges_id:
            if search_field == "summary":
                documents += await milvus_client.search_summary(query, knowledge_id, query_embedding=query_embedding)
            else:
                documents += await milvus_client.search(query, knowledge_id, query_embedding=query_embedding)
        return documents

    @classmethod
    async def retrival_milvus_documents(cls, query, knowledges_id, search_field):
        """只查向量库（ES 关闭时走这里），同样只做一次向量化。"""
        queries = query if isinstance(query, list) else [query]
        embedding_map = await cls.embed_queries(queries)

        documents = []
        for item in queries:
            documents += await cls.retrival_vector_documents(item, embedding_map.get(item), knowledges_id, search_field)
        return documents

    @classmethod
    async def retrival_es_documents(cls, query, knowledges_id, search_field):
        """从Elasticsearch检索文档"""
        documents = []
        queries = query if isinstance(query, list) else [query]

        for query in queries:
            for knowledge_id in knowledges_id:
                if search_field == "summary":
                    documents += await es_client.search_documents_summary(query, knowledge_id)
                else:
                    documents += await es_client.search_documents(query, knowledge_id)

        return documents

    @classmethod
    async def mix_retrival_documents(cls, query_list, knowledges_id, search_field):
        queries = list(query_list)
        # 每条问法只向量化一次，多个知识库共用同一个查询向量
        embedding_map = await cls.embed_queries(queries)

        es_documents = []
        milvus_documents = []
        for query in queries:
            es_documents += await cls.retrival_es_documents(query, knowledges_id, search_field)
            milvus_documents += await cls.retrival_vector_documents(
                query, embedding_map.get(query), knowledges_id, search_field
            )

        return es_documents, milvus_documents
