import json
import time
from typing import List

from elasticsearch import Elasticsearch
from loguru import logger

from agentchat.config.es_index import ESIndex
from agentchat.schemas.chunk import ChunkModel
from agentchat.schemas.search import SearchModel
from agentchat.settings import app_settings

# ES 熔断状态：ES 不可达时快速失败，避免每条检索都等满超时（默认 1s）
_unavailable_until = 0.0
_consecutive_failures = 0


def _config_value(name, default, cast):
    value = getattr(app_settings.rag, name, None)
    try:
        value = cast(value)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _es_timeout() -> float:
    return _config_value("es_timeout", 1.0, float)


def _cooldown() -> float:
    return _config_value("es_cooldown", 30.0, float)


def _failure_threshold() -> int:
    return _config_value("es_failure_threshold", 3, int)


def breaker_open() -> bool:
    return time.monotonic() < _unavailable_until


def _note_success() -> None:
    global _consecutive_failures
    _consecutive_failures = 0


def _note_failure(reason: str) -> None:
    global _consecutive_failures, _unavailable_until
    _consecutive_failures += 1
    threshold = _failure_threshold()
    if _consecutive_failures < threshold:
        logger.warning(f"ES 检索失败（{reason}，连续 {_consecutive_failures}/{threshold}）")
        return
    cooldown = _cooldown()
    _unavailable_until = time.monotonic() + cooldown
    logger.warning(f"ES 连续失败 {_consecutive_failures} 次（{reason}），"
                   f"{cooldown:.0f}s 内只用向量检索")


def reset_breaker() -> None:
    """测试或手动恢复时清空熔断状态。"""
    global _unavailable_until, _consecutive_failures
    _unavailable_until = 0.0
    _consecutive_failures = 0


class ESClient:
    def __init__(self):
        hosts = getattr(app_settings.rag, "elasticsearch", {}) or {}
        self.client = Elasticsearch(hosts=hosts.get('hosts'))

    async def insert_documents(self, index_name, chunks: List[ChunkModel]):
        # 构造查询条件
        index_config = json.loads(ESIndex.index_config)

        if not self.client.indices.exists(index=index_name):

            try:
                self.client.indices.create(index=index_name, body=index_config)
                logger.info(f'index name: {index_name} 创建成功')
            except Exception as e:
                logger.error(f"index name {index_name} error: {e}")
                raise ValueError(f"index create error")
        try:
            for chunk in chunks:
                self.client.index(
                    index=index_name,
                    body=chunk.to_dict()
                )
                logger.info(f'chunk id: {chunk.chunk_id} 已存到索引中')
        except Exception as e:
            logger.error(f"索引增加数据失败：{e}")
        finally:
            await self.close()

    async def index_documents(self, index_name, chunks):
        await self.insert_documents(index_name, chunks)

    async def search_documents(self, query, index_name):
        if breaker_open():
            logger.warning("ES 处于熔断期，跳过关键词检索（本次只用向量检索）")
            return []

        # 模板里用 %s 占位，不能走 str.format（JSON 的花括号会被当成字段）。
        index_search = json.loads(ESIndex.index_search_content % query)

        documents = []
        try:
            response = self.client.options(request_timeout=_es_timeout()).search(
                index=index_name, body=index_search
            )
            _note_success()
            hits = response['hits']
            if not hits.get("max_score"):
                return documents
            for hit in hits.get("hits", []):
                documents.append(
                    SearchModel(
                        score=hit['_score'], chunk_id=hit['_source']['chunk_id'],
                        update_time=hit['_source']['update_time'],
                        content=hit['_source']['content'], file_name=hit['_source']['file_name'],
                        summary=hit['_source']['summary'],
                        file_id=hit['_source']['file_id'],
                        knowledge_id=hit['_source']['knowledge_id']
                    )
                )
        except Exception as e:
            _note_failure(type(e).__name__)
            logger.error(f'Search documents error: {e}')
        finally:
            await self.close()
            return documents

    async def search_documents_summary(self, query, index_name):
        if breaker_open():
            logger.warning("ES 处于熔断期，跳过摘要关键词检索（本次只用向量检索）")
            return []

        index_search = json.loads(ESIndex.index_search_summary % query)

        documents = []
        try:
            response = self.client.options(request_timeout=_es_timeout()).search(
                index=index_name, body=index_search
            )
            _note_success()

            for hit in response['hits'].get("hits", []):
                documents.append(
                    SearchModel(
                        score=hit['_score'], chunk_id=hit['_source']['chunk_id'],
                        update_time=hit['_source']['update_time'],
                        content=hit['_source']['content'], file_name=hit['_source']['file_name'],
                        summary=hit['_source']['summary'],
                        file_id=hit['_source']['file_id'],
                        knowledge_id=hit['_source']['knowledge_id']
                    )
                )

        except Exception as e:
            _note_failure(type(e).__name__)
            logger.error(f'Search documents summary error: {e}')
        finally:
            await self.close()
            return documents

    async def delete_documents(self, file_id, index_name):
        try:
            # 构造查询条件
            delete_query = json.loads(ESIndex.index_delete % file_id)
            self.client.delete_by_query(index=index_name, body=delete_query)
            logger.info(f'Success delete documents in file id: {file_id}')
        except Exception as e:
            logger.error(f'Delete documents Error: {e}')

    async def delete_index(self, index_name):
        try:
            if self.client.indices.exists(index=index_name):
                self.client.indices.delete(index=index_name)
                logger.info(f'Success delete index: {index_name}')
        except Exception as e:
            logger.error(f'Delete index Error: {e}')

    async def close(self):
        pass


# 未启用全文检索（或没有 rag 配置段）时不创建外部客户端。
client = ESClient() if getattr(app_settings.rag, "enable_elasticsearch", False) else None

# ⭐Elasticsearch 在7.11版本之前不支持异步，因本地部署的7.0.0版本，所以使用的同步，如果ES版本较高，建议使用下面的异步代码⭐
# import json
# from typing import List
# from elasticsearch import AsyncElasticsearch
# from agentchat.schemas.chunk import ChunkModel
# from agentchat.schemas.search import SearchModel
# from agentchat.settings import app_settings
# from loguru import logger
#
# class AsyncESClient:
#     def __init__(self):
#         self.client = AsyncElasticsearch(hosts=app_settings.elasticsearch.get('hosts'))
#
#     async def insert_documents(self, index_name, chunks: List[ChunkModel]):
#         with open(app_settings.elasticsearch.get('index_config_path'), 'r') as f:
#             index_config = json.loads(f.read())
#
#         if not await self.client.indices.exists(index=index_name):
#
#             try:
#                 await self.client.indices.create(index=index_name, body=index_config)
#                 logger.info(f'index name: {index_name} 创建成功')
#             except Exception as e:
#                 logger.error(f"index name {index_name} error: {e}")
#                 raise ValueError(f"index create error")
#         try:
#             for chunk in chunks:
#                 await self.client.index(
#                     index=index_name,
#                     body=chunk.to_dict()
#                 )
#                 logger.info(f'chunk id: {chunk.chunk_id} 已存到索引中')
#         except Exception as e:
#             logger.error(f"索引增加数据失败：{e}")
#         finally:
#             await self.close()
#
#     async def index_documents(self, index_name, chunks):
#         await self.insert_documents(index_name, chunks)
#
#     async def search_documents(self, query, index_name):
#         with open(app_settings.elasticsearch.get('index_search_content_path'), 'r') as f:
#             content = f.read()
#             content = content.format(query=query)
#             index_search = json.loads(content)
#
#         try:
#             response = await self.client.search(index=index_name, body=index_search)
#
#             documents = []
#             for hit in response['hits']:
#                 documents.append(SearchModel(score=hit['_score'], chunk_id=hit['_source']['chunk_id'], update_time=hit['_source']['update_time'],
#                                              content=hit['_source']['content'], file_name=hit['_source']['file_name'], summary=hit['_source']['summary'],
#                                              file_id=hit['_source']['file_id'], knowledge_id=hit['_source']['knowledge_id']))
#
#         except Exception as e:
#             logger.error(f'Search documents error: {e}')
#         finally:
#             await self.close()
#
#     async def search_documents_summary(self, query, index_name):
#         with open(app_settings.elasticsearch.get('index_search_summary_path'), 'r') as f:
#             content = f.read()
#             content = content.format(query=query)
#             index_search = json.loads(content)
#
#         try:
#             response = await self.client.search(index=index_name, body=index_search)
#
#             documents = []
#             for hit in response['hits']:
#                 documents.append(SearchModel(score=hit['_score'], chunk_id=hit['_source']['chunk_id'], update_time=hit['_source']['update_time'],
#                                              content=hit['_source']['content'], file_name=hit['_source']['file_name'], summary=hit['_source']['summary'],
#                                              file_id=hit['_source']['file_id'], knowledge_id=hit['_source']['knowledge_id']))
#
#         except Exception as e:
#             logger.error(f'Search documents summary error: {e}')
#         finally:
#             await self.close()
#
#     async def delete_documents(self, file_id, index_name):
#         try:
#             # 构造查询条件
#             with open(app_settings.elasticsearch.get('index_delete_path'), 'r') as f:
#                 content = f.read()
#                 content = content.format(file_id=file_id)
#                 delete_query = json.loads(content)
#
#             await self.client.delete_by_query(index=index_name, body=delete_query)
#             logger.info(f'Success delete documents in file id: {file_id}')
#         except Exception as e:
#             logger.error(f'Delete documents in file id error: {e}')
#
#
#     async def close(self):
#         await self.client.close()
#
# client = AsyncESClient()
