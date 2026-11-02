"""
大语言模型服务
负责与LLM的交互和问答生成
"""

from typing import List, Dict, Any, Optional, AsyncIterator
from openai import AsyncOpenAI, APITimeoutError, APIConnectionError
import time
import asyncio
from urllib.parse import urlsplit
from app.core.config import settings
from app.core.app_logging import get_logger

logger = get_logger(__name__)


class GenerationAPIError(RuntimeError):
    """Safe diagnosis; provider bodies and credentials are never included."""

    def __init__(self, code: str, *, retryable: bool = False):
        self.code = code
        self.retryable = retryable
        super().__init__(code)


def classify_generation_error(exc):
    if isinstance(exc, GenerationAPIError):
        return exc
    if isinstance(exc, (APITimeoutError, asyncio.TimeoutError)):
        return GenerationAPIError("timeout", retryable=True)
    if isinstance(exc, APIConnectionError):
        return GenerationAPIError("connection", retryable=True)
    status = getattr(exc, "status_code", None)
    if status in {401, 403}:
        return GenerationAPIError("credentials")
    if status == 408:
        return GenerationAPIError("timeout", retryable=True)
    if status == 429:
        body = getattr(exc, "body", None)
        error = body.get("error", body) if isinstance(body, dict) else {}
        code = error.get("code") or error.get("type") if isinstance(error, dict) else None
        if code in {"insufficient_quota", "quota_exceeded", "billing_hard_limit_reached"}:
            return GenerationAPIError("quota")
        return GenerationAPIError("rate_limit", retryable=True)
    if status is not None and status >= 500:
        return GenerationAPIError("server", retryable=True)
    if status in {400, 404, 405, 413, 422}:
        return GenerationAPIError("configuration")
    return GenerationAPIError("unexpected")


class LLMService:
    """大语言模型服务：与LLM交互并生成问答"""

    def __init__(self, *, model_name: Optional[str] = None, max_tokens: Optional[int] = None, temperature: Optional[float] = None):
        api_key = settings.generation_api_key
        base_url = settings.generation_api_base_url
        self._api_key = api_key
        self._base_url = base_url
        self.model = model_name if model_name is not None else settings.generation_model()
        self.core_model_name = self.model
        self.light_model_name = (model_name if model_name is not None
                                 else settings.generation_model("light"))
        self.max_tokens = max_tokens if max_tokens is not None else settings.OPENAI_MAX_TOKENS
        self.temperature = temperature if temperature is not None else settings.OPENAI_TEMPERATURE
        self.client = None
        self._client_configuration_error = False
        if api_key and base_url and self.model:
            try:
                parsed = urlsplit(base_url)
                if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                    raise ValueError("Invalid generation API URL")
                _ = parsed.port  # Validate an explicitly supplied port before making requests.
                self.client = AsyncOpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    timeout=float(settings.OPENAI_TIMEOUT_SECS),
                    max_retries=0,
                )
            except Exception:
                self._client_configuration_error = True

        # 并发限制（全局 Semaphore，跨实例共享，确保总并发受控）
        if not hasattr(LLMService, "_global_semaphore") or LLMService._global_semaphore is None:
            LLMService._global_semaphore = asyncio.Semaphore(settings.LLM_MAX_CONCURRENCY)
        self.semaphore = LLMService._global_semaphore
        # 熔断状态（类级别共享）
        if not hasattr(LLMService, "_failure_count"):
            LLMService._failure_count = 0
        if not hasattr(LLMService, "_circuit_opened_at"):
            LLMService._circuit_opened_at = None

    def has_credentials(self) -> bool:
        return self.client is not None

    @classmethod
    def with_model(cls, model_name: str, *, max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> "LLMService":
        """工厂方法：按需指定模型/参数创建实例。"""
        return cls(model_name=model_name, max_tokens=max_tokens, temperature=temperature)

    # --- 可靠性：熔断与退避重试 ---
    def _is_circuit_open(self) -> bool:
        opened_at = getattr(LLMService, "_circuit_opened_at", None)
        if opened_at is None:
            return False
        elapsed = time.time() - opened_at
        if elapsed >= settings.LLM_CIRCUIT_RESET_TIMEOUT:
            # 熔断自动恢复
            LLMService._circuit_opened_at = None
            LLMService._failure_count = 0
            return False
        return True

    def _record_success(self) -> None:
        LLMService._failure_count = 0
        LLMService._circuit_opened_at = None

    def _record_failure(self) -> None:
        LLMService._failure_count += 1
        if LLMService._failure_count >= settings.LLM_CIRCUIT_FAILURE_THRESHOLD:
            if LLMService._circuit_opened_at is None:
                LLMService._circuit_opened_at = time.time()
                logger.warning(
                    f"LLM 熔断开启（失败次数 {LLMService._failure_count}）— 冷却 {settings.LLM_CIRCUIT_RESET_TIMEOUT}s"
                )

    def _backoff_seconds(self, attempt_index: int) -> float:
        base = settings.LLM_BACKOFF_BASE
        factor = settings.LLM_BACKOFF_FACTOR
        max_v = settings.LLM_BACKOFF_MAX
        delay = base * (factor ** attempt_index)
        return min(delay, max_v)

    async def _request(self, messages, *, model=None, stream=False, **kwargs):
        if not self.has_credentials():
            raise GenerationAPIError("configuration" if self._client_configuration_error else "not_configured")
        if self._is_circuit_open():
            raise GenerationAPIError("circuit_open")
        try:
            await asyncio.wait_for(self.semaphore.acquire(), timeout=settings.LLM_QUEUE_TIMEOUT_SECS)
        except asyncio.TimeoutError:
            raise GenerationAPIError("queue_timeout") from None
        try:
            attempts = min(10, max(1, settings.LLM_MAX_RETRIES))
            for attempt in range(attempts):
                failure = None
                try:
                    response = await self.client.chat.completions.create(
                        model=model or self.model, messages=messages, stream=stream, **kwargs)
                    if stream:
                        if not hasattr(response, "__aiter__"):
                            raise GenerationAPIError("invalid_response")
                    else:
                        choices = getattr(response, "choices", None)
                        content = (getattr(getattr(choices[0], "message", None), "content", None)
                                   if isinstance(choices, (list, tuple)) and choices else None)
                        if not isinstance(content, str) or not content.strip():
                            raise GenerationAPIError("invalid_response")
                    self._record_success()
                    return response
                except Exception as exc:
                    failure = classify_generation_error(exc)
                # Only transient failures retry, and no provider body escapes the boundary.
                if failure.retryable:
                    self._record_failure()
                if not failure.retryable or attempt + 1 == attempts:
                    raise failure from None
                delay = self._backoff_seconds(attempt)
                logger.warning("Generation API %s; retry %d/%d in %.2fs",
                               failure.code, attempt + 1, attempts, delay)
                await asyncio.sleep(delay)
        finally:
            self.semaphore.release()

    async def _chat_completion(self, messages, **kwargs):
        stream = kwargs.pop("stream", False)
        return await self._request(messages, stream=stream, **kwargs)

    async def _chat_completion_stream(self, messages, **kwargs):
        return await self._request(messages, stream=True, **kwargs)

    async def _stream_events(self, stream):
        failure = None
        has_text = False
        try:
            async for event in stream:
                choices = getattr(event, "choices", None)
                if choices is not None and not isinstance(choices, (list, tuple)):
                    raise GenerationAPIError("invalid_response")
                if choices:
                    delta = getattr(choices[0], "delta", None)
                    content = getattr(delta, "content", None)
                    if content is not None and not isinstance(content, str):
                        raise GenerationAPIError("invalid_response")
                    if isinstance(content, str) and content.strip():
                        has_text = True
                yield event
        except Exception as exc:
            failure = classify_generation_error(exc)
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                try:
                    await close()
                except Exception:
                    logger.warning("Generation stream close failed")
        if failure:
            raise failure from None  # Never replay an already partially emitted answer.
        if not has_text:
            raise GenerationAPIError("invalid_response")

    async def preflight(self):
        try:
            await self._request([{"role": "user", "content": "Reply with OK."}],
                                max_tokens=8, temperature=0)
            return {"status": "available", "model": self.model}
        except GenerationAPIError as exc:
            return {"status": "unavailable", "code": exc.code, "retryable": exc.retryable,
                    "model": self.model}

    def _build_system_prompt(self) -> str:
        """构建系统提示词：条款证据问答，非保险顾问。"""
        return (
            "你是本地保险条款检索问答助手，只根据给定条款片段作答。\n"
            "硬性规则：\n"
            "1. 只使用提供的条款片段，禁止编造保额、疾病、责任或监管结论。\n"
            "2. 答案必须使用中文，并严格按三段输出：【结论】【条款依据】【不确定/边界】。\n"
            "3. 在【条款依据】中用 [1][2] 引用片段编号，编号对应输入顺序。\n"
            "4. 若片段不足，明确写“未在已入库条款中找到充分依据”。\n"
            "5. 禁止给出购买建议、配置方案或“一定能获赔”的承诺。\n"
            "6. 最后一行必须包含：本系统不构成保险销售或理赔承诺。"
        )

    def _build_user_prompt(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """构建用户提示词（编号片段，便于 [1][2] 引用）。"""
        doc_parts: List[str] = []
        for i, chunk in enumerate(context_chunks, 1):
            content = chunk.get("content", "")
            page_num = chunk.get("page_number", "N/A")
            doc_name = chunk.get("document_name", "Unknown")
            document_id = chunk.get("document_id", "Unknown")
            doc_parts.append(
                f"[{i}] 来源: {doc_name} | 文档 ID: {document_id} | 页码: {page_num}\n内容: {content}\n"
            )
        context = "\n".join(doc_parts) if doc_parts else "(无上下文)"
        prompt = (
            "以下是已检索到的保险条款片段：\n\n"
            f"{context}\n\n"
            f"用户问题：{query}\n\n"
            "请按【结论】【条款依据】【不确定/边界】三段作答，并在依据中使用 [1][2] 引用。"
        )
        return prompt

    async def agenerate_response(self, query: str, context_chunks: List[Dict[str, Any]], *, max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> Dict[str, Any]:
        """非流式：生成完整回复（满足 T2.3/T2.4）。"""
        start_time = time.time()

        try:
            messages = [
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user", "content": self._build_user_prompt(query, context_chunks)},
            ]

            response = await self._request(
                messages, model=self.core_model_name,
                max_tokens=(max_tokens or self.max_tokens),
                temperature=(temperature if temperature is not None else self.temperature))

            answer = response.choices[0].message.content.strip()
            tokens_used = getattr(response.usage, "total_tokens", 0)
            response_time = time.time() - start_time

            return {
                "answer": answer,
                "model_used": self.core_model_name,
                "tokens_used": tokens_used,
                "response_time": response_time,
                "success": True,
            }

        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"LLM回复生成失败: {type(e).__name__}")
            return {
                "answer": "抱歉, 生成回复时出现错误, 请稍后重试。",
                "model_used": self.core_model_name,
                "tokens_used": 0,
                "response_time": response_time,
                "success": False,
                "error": classify_generation_error(e).code,
            }

    async def agenerate_stream(self, query: str, context_chunks: List[Dict[str, Any]], *, max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> AsyncIterator[str]:
        """流式：异步生成器，逐个 yield 文本增量（满足 T2.5/T2.6）。"""
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": self._build_user_prompt(query, context_chunks)},
        ]

        stream = await self._request(
            messages, model=self.core_model_name, stream=True,
            max_tokens=(max_tokens or self.max_tokens),
            temperature=(temperature if temperature is not None else self.temperature))

        async for event in self._stream_events(stream):
            try:
                if hasattr(event, "choices") and event.choices:
                    delta = event.choices[0].delta
                    content_delta = getattr(delta, "content", None)
                    if content_delta:
                        yield content_delta
            except Exception:
                # 忽略单次解析异常，继续流
                pass

    async def agenerate_structured_decision(
        self,
        messages: List[Dict[str, Any]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> Any:
        """
        轻量任务：路由/意图/重写 等结构化决策调用，强制使用 light 模型。
        返回 OpenAI ChatCompletion 响应对象，由调用方自行解析 JSON。
        """
        return await self._request(messages, model=self.light_model_name,
                                   max_tokens=max_tokens, temperature=temperature)

    async def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成问答回复"""
        start_time = time.time()

        try:
            messages = [
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user", "content": self._build_user_prompt(query, context_chunks)},
            ]

            response = await self._chat_completion(
                messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=False,
            )

            answer = response.choices[0].message.content.strip()
            tokens_used = getattr(response.usage, "total_tokens", 0)
            response_time = time.time() - start_time

            result = {
                "answer": answer,
                "model_used": self.model,
                "tokens_used": tokens_used,
                "response_time": response_time,
                "success": True,
            }

            logger.info(
                f"LLM回复生成成功, 用时 {response_time:.2f}s, 使用 {tokens_used} tokens"
            )
            return result

        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"LLM回复生成失败: {type(e).__name__}")

            return {
                "answer": "抱歉, 生成回复时出现错误, 请稍后重试。",
                "model_used": self.model,
                "tokens_used": 0,
                "response_time": response_time,
                "success": False,
                "error": classify_generation_error(e).code,
            }

    async def stream_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> AsyncIterator[Dict[str, Any]]:
        """以流式方式生成问答回复。

        事件流结构：
        - {"type": "start", "model_used": ..., "timestamp": ...}
        - {"type": "token", "content": "..."}  // 多次
        - {"type": "end", "answer": "...", "tokens_used": N, "response_time": S, "success": True}
        - 若失败：{"type": "error", "message": "...", "response_time": S}
        """
        started_at = time.time()
        # 构建消息
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": self._build_user_prompt(query, context_chunks)},
        ]

        # 起始事件
        yield {
            "type": "start",
            "model_used": self.model,
            "timestamp": started_at,
        }

        final_text: str = ""
        try:
            stream = await self._chat_completion_stream(
                messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            async for event in self._stream_events(stream):
                try:
                    # OpenAI v1: ChatCompletionChunk -> choices[0].delta.content
                    if hasattr(event, "choices") and event.choices:
                        delta = event.choices[0].delta
                        content_delta = getattr(delta, "content", None)
                        if content_delta:
                            final_text += content_delta
                            yield {"type": "token", "content": content_delta}
                except Exception as parse_err:
                    # 忽略解析异常，继续流（不终止）
                    logger.debug(f"流事件解析异常: {parse_err}")

            # 结束后尝试获取最终响应以读取用量
            tokens_used = 0
            try:
                final_resp = await stream.get_final_response()
                tokens_used = getattr(getattr(final_resp, "usage", None), "total_tokens", 0) or 0
            except Exception:
                tokens_used = 0

            response_time = time.time() - started_at
            yield {
                "type": "end",
                "answer": final_text.strip(),
                "model_used": self.model,
                "tokens_used": tokens_used,
                "response_time": response_time,
                "success": True,
            }

        except Exception as e:
            response_time = time.time() - started_at
            logger.error(f"LLM流式生成失败: {type(e).__name__}")
            yield {
                "type": "error",
                "message": classify_generation_error(e).code,
                "model_used": self.model,
                "response_time": response_time,
                "success": False,
            }

    async def generate_summary(self, text: str, max_length: int = 200) -> str:
        """生成文本摘要"""
        try:
            messages = [
                {
                    "role": "system",
                    "content": f"请为以下文本生成一个简洁的摘要, 长度不超过 {max_length} 个字符。",
                },
                {"role": "user", "content": text},
            ]

            response = await self._chat_completion(
                messages,
                max_tokens=max_length // 2,
                temperature=0.3,
            )

            summary = response.choices[0].message.content.strip()
            logger.info("文本摘要生成成功")
            return summary

        except Exception as e:
            logger.error(f"文本摘要生成失败: {type(e).__name__}")
            return text[:max_length] + "..." if len(text) > max_length else text

    async def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """提取关键词"""
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        f"请从以下文本中提取最重要的 {max_keywords} 个关键词, 用逗号分隔。"
                    ),
                },
                {"role": "user", "content": text},
            ]

            response = await self._chat_completion(
                messages,
                max_tokens=100,
                temperature=0.3,
            )

            keywords_text = response.choices[0].message.content.strip()
            keywords = [kw.strip() for kw in keywords_text.split(",")]

            logger.info(f"关键词提取成功, 共 {len(keywords)} 个")
            return keywords[:max_keywords]

        except Exception as e:
            # 回退到本地关键词提取（jieba.analyse）以避免429等限流问题
            try:
                import jieba.analyse as analyse  # type: ignore
                kws = analyse.extract_tags(text or "", topK=max_keywords)
                logger.info(f"关键词本地回退成功, 共 {len(kws)} 个")
                return kws
            except Exception as e2:
                logger.warning(f"关键词提取失败且本地回退异常: {type(e).__name__} / {type(e2).__name__}")
                return []

    async def classify_query(self, query: str) -> Dict[str, Any]:
        """查询分类"""
        try:
            categories = [
                "保险条款查询",
                "理赔相关",
                "保费计算",
                "保险产品对比",
                "投保流程",
                "其他",
            ]

            messages = [
                {
                    "role": "system",
                    "content": (
                        f"请将以下查询分类到这些类别中的一个: {', '.join(categories)}。只返回类别名称。"
                    ),
                },
                {"role": "user", "content": query},
            ]

            response = await self._chat_completion(
                messages,
                max_tokens=50,
                temperature=0.1,
            )

            category = response.choices[0].message.content.strip()
            if category not in categories:
                category = "其他"

            logger.info(f"查询分类完成: {category}")
            return {"category": category, "confidence": 0.8, "success": True}

        except Exception as e:
            logger.error(f"查询分类失败: {type(e).__name__}")
            return {
                "category": "其他",
                "confidence": 0.0,
                "success": False,
                "error": classify_generation_error(e).code,
            }

    async def health_check(self) -> bool:
        """Credentials present (demo does not ping provider on every health)."""
        return self.has_credentials()


    async def rewrite_query(
        self,
        user_query: str,
        ontology: Dict[str, Any],
        max_output_tokens: int = 256,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        """口语化查询重写：将用户原始问题转换为专业的、语义更丰富的检索查询。

        输入：用户原始查询与保险术语本体库（JSON结构）
        输出：字典，包含 rewritten_query、intent_tags、keywords、constraints 等字段
        失败时返回 success=False 并提供错误信息。
        """
        try:
            # 将本体库压缩为有限长度的字符串，避免超长提示
            import json as _json
            ontology_text = _json.dumps(ontology, ensure_ascii=False)
            if len(ontology_text) > 4000:
                ontology_text = ontology_text[:4000] + "..."

            system_prompt = (
                "你是保险查询重写专家。你的任务：基于保险术语本体库，将用户口语化问题重写为专业、语义丰富的检索查询。"
                "要求：\n"
                "- 使用术语同义词与标准化名称（本体库提供）\n"
                "- 明确产品类型、保障责任、限制条件与关键数字\n"
                "- 保持中文，避免增加未给出的新事实\n"
                "- 输出JSON结构，字段：rewritten_query、intent_tags、keywords、constraints\n"
            )

            user_content = (
                f"保险术语本体库（片段）：\n{ontology_text}\n\n"
                f"用户原始问题：\n{user_query}\n\n"
                "请重写为用于向量检索的专业查询，并返回如下JSON：\n"
                "{\n"
                "  \"rewritten_query\": \"...\",\n"
                "  \"intent_tags\": [\"产品类型\", \"保障责任\", \"条款/限制\"],\n"
                "  \"keywords\": [\"标准术语或关键短语\"],\n"
                "  \"constraints\": {\"等待期\": \"30天\", \"免赔额\": \"1000元\"}\n"
                "}"
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]

            response = await self._chat_completion(
                messages,
                max_tokens=max_output_tokens,
                temperature=temperature,
            )

            content = response.choices[0].message.content.strip()
            # 尝试抽取JSON
            import json as _json2
            parsed: Optional[Dict[str, Any]] = None
            try:
                if "```json" in content:
                    start = content.find("```json") + 7
                    end = content.find("```", start)
                    if end != -1:
                        content = content[start:end].strip()
                elif content.startswith("{"):
                    pass
                parsed = _json2.loads(content)
            except Exception:
                # 尝试第二种方式：提取第一段以 { 开头的块
                l = content.find("{")
                r = content.rfind("}")
                if l != -1 and r != -1 and r > l:
                    try:
                        parsed = _json2.loads(content[l : r + 1])
                    except Exception:
                        parsed = None

            if not isinstance(parsed, dict):
                logger.warning("重写结果解析失败，回退到原始查询")
                return {
                    "rewritten_query": user_query,
                    "intent_tags": [],
                    "keywords": [],
                    "constraints": {},
                    "success": False,
                    "error": "解析失败",
                }

            rewritten = parsed.get("rewritten_query") or user_query
            return {
                "rewritten_query": rewritten,
                "intent_tags": parsed.get("intent_tags", []),
                "keywords": parsed.get("keywords", []),
                "constraints": parsed.get("constraints", {}),
                "success": True,
            }

        except Exception as e:
            logger.error(f"查询重写失败: {type(e).__name__}")
            return {
                "rewritten_query": user_query,
                "intent_tags": [],
                "keywords": [],
                "constraints": {},
                "success": False,
                "error": classify_generation_error(e).code,
            }

