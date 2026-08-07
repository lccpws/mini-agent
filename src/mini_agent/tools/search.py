from mini_agent.tools.base_tool import BaseTool
import requests
import time


class SearchTool(BaseTool):
    name = "search"
    description = "搜索信息，使用 Wikipedia 获取知识"
    capabilities = ["search"]

    version = "1.0.0"
    author = "MiniAgent"
    allow_roles = ["admin", "user"]
    required_permissions = ["network"]
    risk_level = 1

    MAX_RETRIES = 3
    RETRY_DELAY = 1

    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词"
            }
        },
        "required": ["query"]
    }

    def execute(self, query):
        if not query:
            return {"error": "请输入搜索关键词"}

        search_result = self._search_with_fallback(query)
        if not search_result:
            return {"error": f"未找到关于 '{query}' 的信息"}

        page_content = self._request_with_retry(
            self._get_page_content, search_result["pageid"], search_result.get("lang", "zh")
        )
        if not page_content:
            return {"error": f"无法获取 '{query}' 的详细内容"}

        return self._format_result_dict(search_result, page_content)

    def _search_with_fallback(self, query):
        search_result = self._request_with_retry(self._search_wikipedia, query, "zh")
        if search_result:
            search_result["lang"] = "zh"
            return search_result

        search_result = self._request_with_retry(self._search_wikipedia, query, "en")
        if search_result:
            search_result["lang"] = "en"
            return search_result

        return None

    def _request_with_retry(self, func, *args, **kwargs):
        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                result = func(*args, **kwargs)
                if result is not None:
                    return result
                last_error = "返回结果为空"
            except Exception as e:
                last_error = str(e)

            if attempt < self.MAX_RETRIES - 1:
                time.sleep(self.RETRY_DELAY * (attempt + 1))

        return None

    def _get_headers(self):
        return {
            "User-Agent": "MiniAgent/1.0 (https://github.com/lccpws/mini-agent; mini-agent@example.com)"
        }

    def _search_wikipedia(self, query, lang="zh"):
        url = f"https://{lang}.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": 1,
            "format": "json"
        }

        resp = requests.get(url, params=params, headers=self._get_headers(), timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data.get("query", {}).get("search"):
            return data["query"]["search"][0]
        return None

    def _get_page_content(self, pageid, lang="zh"):
        url = f"https://{lang}.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "pageids": pageid,
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "exsectionformat": "plain",
            "format": "json"
        }

        resp = requests.get(url, params=params, headers=self._get_headers(), timeout=10)
        resp.raise_for_status()
        data = resp.json()

        pages = data.get("query", {}).get("pages", {})
        if pages:
            page = next(iter(pages.values()))
            return page.get("extract", "")
        return None

    def _format_result_dict(self, search_result, content) -> dict:
        title = search_result.get("title", "")
        snippet = search_result.get("snippet", "")

        clean_content = content[:2000] if len(content) > 2000 else content

        return {
            "title": title,
            "content": clean_content,
            "snippet": snippet,
            "source": "Wikipedia",
            "pageid": search_result.get("pageid"),
        }

    def _format_result(self, search_result, content) -> str:
        data = self._format_result_dict(search_result, content)
        return f"## {data['title']}\n\n{data['content']}\n\n来源: {data['source']}"


def search(query: str):
    tool = SearchTool()
    return tool.execute(query)
