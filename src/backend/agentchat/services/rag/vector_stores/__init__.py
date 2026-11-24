from agentchat.services.rag.vector_stores.milvus import MilvusClient
from agentchat.services.rag.vector_stores.chroma import ChromaClient
from agentchat.services.rag.vector_stores.milvus_lite import MilvusLiteClient
from agentchat.settings import app_settings


def _create_vector_client():
    """按配置创建向量库客户端。

    单测里 app_settings.rag 尚未初始化（没有 vector_db 段），此时返回 None，
    避免 import 阶段就尝试连 Milvus 让整个测试收集失败；真实启动前一定先读配置。
    """
    rag = getattr(app_settings, "rag", None)
    vector_db = getattr(rag, "vector_db", None) or {}
    mode = vector_db.get("mode")
    if mode == "chroma":
        return ChromaClient()
    if mode == "lite":
        return MilvusLiteClient()
    if rag is None:
        return None
    return MilvusClient()


milvus_client = _create_vector_client()
