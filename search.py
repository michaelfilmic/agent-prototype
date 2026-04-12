from ddgs import DDGS

MIN_BODY_LENGTH = 80  # filter out results with too little content

def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Search DuckDuckGo and return raw results."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    return results


def filter_results(results: list[dict]) -> list[dict]:
    """Remove low-quality results using simple rules (no LLM needed)."""
    filtered = []
    seen_urls = set()

    for r in results:
        # Skip duplicates
        if r.get("href") in seen_urls:
            continue
        # Skip results with too little content
        if len(r.get("body", "")) < MIN_BODY_LENGTH:
            continue

        filtered.append(r)
        seen_urls.add(r.get("href"))

    return filtered


def format_for_llm(results: list[dict]) -> str:
    """Format filtered results into a string for the LLM."""
    if not results:
        return "No relevant search results found."

    sections = []
    for i, r in enumerate(results, 1):
        sections.append(
            f"[{i}] {r.get('title', 'No title')}\n"
            f"URL: {r.get('href', '')}\n"
            f"Content: {r.get('body', '')}"
        )
    return "\n\n".join(sections)
