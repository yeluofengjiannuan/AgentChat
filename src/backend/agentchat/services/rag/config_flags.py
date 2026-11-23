"""RAG 行为开关。

放在独立小模块里，避免为了读一个配置项去 import 向量库/ES 客户端（那些在 import 期就会建连接）。
"""
from typing import Optional

from agentchat.settings import app_settings


def should_query_rewrite(needs_query_rewrite: Optional[bool] = None) -> bool:
    """显式传参优先，否则读 rag.enable_query_rewrite（默认开启，保持原有行为）。"""
    if needs_query_rewrite is not None:
        return bool(needs_query_rewrite)
    return bool(getattr(app_settings.rag, "enable_query_rewrite", True))
