"""
HTTP-backed web search tool that proxies queries to external providers.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import requests

from agent_rl.tools.base_tool import BaseTool

logger = logging.getLogger(__name__)


class WebSearchTool(BaseTool):
    """
    Execute live web searches via a configurable HTTP API.

    The implementation is intentionally provider-agnostic: different SaaS search
    engines (Tavily, Serper, Bing, custom proxies, etc.) can be supported by
    tweaking the `provider`, `endpoint`, and `default_params` arguments.
    """

    name = "search"
    description = (
        "Call an external web search API and return the top results with titles, URLs, "
        "and short snippets. Use this tool whenever you need fresh information."
    )

    _PROVIDER_ENDPOINTS = {
        "tavily": "https://api.tavily.com/search",
        "serper": "https://google.serper.dev/search",
        "bing": "https://api.bing.microsoft.com/v7.0/search",
    }
    _REQUIRES_API_KEY = {"tavily", "serper", "bing"}

    def __init__(
        self,
        provider: str = "tavily",
        *,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        default_params: Optional[Dict[str, Any]] = None,
        timeout: float = 15.0,
        max_retries: int = 2,
        backoff_factor: float = 0.8,
        rate_limit_per_min: Optional[int] = 8,
        max_snippet_chars: int = 480,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.provider = provider.lower()
        self.endpoint = endpoint or self._PROVIDER_ENDPOINTS.get(self.provider)
        self.api_key = api_key
        self.default_params = default_params or {}
        self.timeout = timeout
        self.max_retries = max(0, int(max_retries))
        self.backoff_factor = max(0.1, float(backoff_factor))
        self.rate_limit_per_min = rate_limit_per_min
        self.max_snippet_chars = max(120, max_snippet_chars)
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", "MiniAgent-WebSearchTool/1.0")
        self._last_request_ts: float = 0.0

        if not self.endpoint:
            raise ValueError(
                f"No endpoint configured for provider '{self.provider}'. "
                "Pass `endpoint=` explicitly to use custom proxies."
            )
        if self.provider in self._REQUIRES_API_KEY and not self.api_key:
            raise ValueError(
                f"Provider '{self.provider}' requires an API key. "
                "Configure `api_key` or `api_key_env` in the training config."
            )

    def __call__(self, query: str, top_k: int = 5, **kwargs: Any) -> str:
        query = (query or "").strip()
        if not query:
            return "Search aborted: empty query."
        top_k = max(1, int(top_k or 5))

        call_params = dict(self.default_params)
        call_params.update({k: v for k, v in kwargs.items() if v is not None})

        error_message = ""
        for attempt in range(self.max_retries + 1):
            try:
                self._respect_rate_limit()
                request_args = self._build_request(query, top_k, call_params)
                response = self.session.request(timeout=self.timeout, **request_args)
                response.raise_for_status()
                data = response.json()
                self._last_request_ts = time.monotonic()
                return self._format_results(data, provider=self.provider)
            except requests.RequestException as exc:
                error_message = f"{type(exc).__name__}: {exc}"
                sleep_s = self.backoff_factor * (2**attempt)
                logger.warning(
                    "WebSearchTool request failed (attempt %s/%s): %s",
                    attempt + 1,
                    self.max_retries + 1,
                    error_message,
                )
                time.sleep(sleep_s)
            except ValueError as exc:
                # JSON decoding or provider-specific parsing issues.
                return f"Search failed: {exc}"

        return (
            "Search failed after multiple retries. "
            f"Last error: {error_message or 'unknown error'}"
        )

    def _respect_rate_limit(self) -> None:
        if not self.rate_limit_per_min:
            return
        interval = 60.0 / max(self.rate_limit_per_min, 1)
        elapsed = time.monotonic() - self._last_request_ts
        if elapsed < interval:
            time.sleep(interval - elapsed)

    def _build_request(
        self, query: str, top_k: int, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        headers = dict(self.session.headers)
        req: Dict[str, Any] = {"method": "get", "url": self.endpoint, "headers": headers}

        if self.provider == "tavily":
            payload = {
                "api_key": self.api_key,
                "query": query,
                "max_results": top_k,
                "search_depth": params.get("search_depth", "basic"),
            }
            payload.update({k: v for k, v in params.items() if k not in payload})
            req.update({"method": "post", "json": payload})

        elif self.provider == "serper":
            headers["X-API-KEY"] = self.api_key or ""
            payload = {"q": query, "num": top_k}
            payload.update(params)
            req.update({"method": "post", "json": payload})

        elif self.provider == "bing":
            headers["Ocp-Apim-Subscription-Key"] = self.api_key or ""
            query_params = {"q": query, "count": top_k}
            query_params.update(params)
            req.update({"method": "get", "params": query_params})

        else:
            # Generic JSON POST contract: the remote proxy is expected to
            # understand {query, top_k, extra params} and return {results:[...]}.
            payload = {"query": query, "top_k": top_k}
            payload.update(params)
            req.update({"method": "post", "json": payload})

        return req

    def _format_results(self, data: Dict[str, Any], *, provider: str) -> str:
        results = self._extract_results(data, provider)
        if not results:
            answer = data.get("answer")
            if isinstance(answer, str) and answer.strip():
                return answer.strip()
            return "Search completed but no results were returned."

        formatted: List[str] = []
        for idx, item in enumerate(results, start=1):
            title = item.get("title") or "Untitled result"
            url = item.get("url") or item.get("link") or ""
            snippet = (
                item.get("snippet")
                or item.get("content")
                or item.get("body")
                or ""
            ).strip()
            if len(snippet) > self.max_snippet_chars:
                snippet = f"{snippet[: self.max_snippet_chars].rstrip()}..."
            block = f"[{idx}] {title}\nURL: {url}\n{snippet or '(no snippet provided)'}"
            formatted.append(block)
        return "\n\n".join(formatted)

    def _extract_results(
        self, data: Dict[str, Any], provider: str
    ) -> List[Dict[str, Any]]:
        if provider == "tavily":
            results = data.get("results", [])
        elif provider == "serper":
            results = data.get("organic", [])
        elif provider == "bing":
            results = data.get("webPages", {}).get("value", [])
        else:
            results = data.get("results", [])

        if not isinstance(results, list):
            return []
        return [item for item in results if isinstance(item, dict)]


__all__ = ["WebSearchTool"]


