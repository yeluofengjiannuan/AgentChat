"""向量化调用（embedding）。

外部 embedding 服务会间歇性排队变慢，这里做两层保护：
1. 单次调用预算 rag.embedding_timeout（默认 1.2s），超时立即失败，避免整条检索被拖死；
2. 失败后熔断 rag.embedding_cooldown（默认 30s），熔断期内直接快速失败，
   检索层会把这条查询降级为只查 ES，而不是每条重写查询都再等一次超时。
"""
import asyncio
import time
from typing import Union, List

from loguru import logger

from agentchat.core.models.manager import ModelManager
from agentchat.settings import app_settings

# 熔断到期时间戳（monotonic）；0 表示未熔断
_unavailable_until = 0.0
# 连续失败次数：单次抖动不应熔断，连续失败达到阈值才熔断
_consecutive_failures = 0


def _config_float(name: str, default: float) -> float:
    value = getattr(app_settings.rag, name, None)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _config_int(name: str, default: int) -> int:
    value = getattr(app_settings.rag, name, None)
    try:
        value = int(value)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _timeout() -> float:
    return _config_float("embedding_timeout", 1.2)


def _cooldown() -> float:
    return _config_float("embedding_cooldown", 30.0)


def _failure_threshold() -> int:
    return _config_int("embedding_failure_threshold", 3)


def _breaker_open() -> bool:
    return time.monotonic() < _unavailable_until


def _note_success() -> None:
    global _consecutive_failures
    _consecutive_failures = 0


def _note_failure(reason: str) -> None:
    global _consecutive_failures, _unavailable_until
    _consecutive_failures += 1
    threshold = _failure_threshold()
    if _consecutive_failures < threshold:
        logger.warning(f"embedding 调用失败（{reason}，连续 {_consecutive_failures}/{threshold}）")
        return
    cooldown = _cooldown()
    _unavailable_until = time.monotonic() + cooldown
    logger.warning(f"embedding 连续失败 {_consecutive_failures} 次（{reason}），"
                   f"{cooldown:.0f}s 内检索降级为 ES 结果")


def reset_breaker() -> None:
    """测试或手动恢复时清空熔断状态。"""
    global _unavailable_until, _consecutive_failures
    _unavailable_until = 0.0
    _consecutive_failures = 0


async def _create_embeddings(embedding_client, inputs):
    """带预算和熔断的单次 embeddings.create 调用。"""
    if _breaker_open():
        raise RuntimeError("embedding 处于熔断期，直接降级")

    deadline = time.monotonic() + _timeout()
    last_error: Exception | None = None
    for _ in range(2):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            response = await asyncio.wait_for(
                embedding_client.embeddings.create(
                    model=app_settings.multi_models.embedding.model_name,
                    input=inputs,
                    encoding_format="float",
                ),
                timeout=remaining,
            )
            _note_success()
            return response
        except asyncio.TimeoutError as err:
            # 预算已被这次超时用完，重试没有意义
            last_error = err
            break
        except Exception as err:
            # 快速失败（如连接被拒）才允许在剩余预算内重试一次
            last_error = err
            if deadline - time.monotonic() <= 0.05:
                break

    _note_failure(type(last_error).__name__ if last_error else "unknown")
    raise last_error if last_error else RuntimeError("embedding 调用失败")


async def get_embedding(query: Union[str, List[str]]):
    embedding_client = ModelManager.get_embedding_openai_model()

    # 如果是字符串或长度小于等于10的列表，直接处理
    if isinstance(query, str) or (isinstance(query, list) and len(query) <= 10):
        responses = await _create_embeddings(embedding_client, query)

        if isinstance(query, str):
            return responses.data[0].embedding
        else:
            return [response.embedding for response in responses.data]

    # 处理超过10条的情况
    semaphore = asyncio.Semaphore(5)  # 限制并发数为5

    async def process_batch(batch):
        async with semaphore:
            responses = await _create_embeddings(embedding_client, batch)
            return [response.embedding for response in responses.data]

    # 将查询分成每组10条
    batches = [query[i:i + 10] for i in range(0, len(query), 10)]

    # 并发处理所有批次
    tasks = [process_batch(batch) for batch in batches]
    results = await asyncio.gather(*tasks)

    # 将所有结果合并
    return [embedding for batch_result in results for embedding in batch_result]
