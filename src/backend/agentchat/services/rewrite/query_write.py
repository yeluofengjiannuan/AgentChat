import json

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from agentchat.core.models.manager import ModelManager
from agentchat.prompts.rewrite import system_query_rewrite, user_query_write


class QueryRewrite:
    def __init__(self):
        self.client = ModelManager.get_chat_model()

    @classmethod
    def _normalize_queries(cls, payload, user_input):
        """把模型输出统一成非空字符串列表。

        小模型偶尔会返回 {"original_query": ..., "variations": [...]} 这种对象，
        直接往下传会让 ES/向量检索拿到 dict 而静默失败，所以这里做一次结构校验。
        """
        if isinstance(payload, dict):
            payload = payload.get("variations") or payload.get("queries") or []
        if not isinstance(payload, list):
            return [user_input]
        queries = [item.strip() for item in payload if isinstance(item, str) and item.strip()]
        return queries or [user_input]

    async def rewrite(self, user_input):
        rewrite_prompt = user_query_write.format(user_input=user_input)
        response = self.client.invoke([SystemMessage(content=system_query_rewrite), HumanMessage(content=rewrite_prompt)])
        cleaned_response = response.content.replace("```json", "")
        cleaned_response = cleaned_response.replace("```", "").strip()

        try:
            result = json.loads(cleaned_response)
        except Exception as e:
            logger.info(f"json loads error: {e}")
            return [user_input]
        return self._normalize_queries(result, user_input)

query_rewriter = QueryRewrite()