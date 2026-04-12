from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END

from model import get_llm
from search import web_search, filter_results, format_for_llm

llm = get_llm()


# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    question: str           # original user question
    should_search: bool     # router decision
    search_query: str       # rewritten query for search engine
    raw_results: list       # raw DuckDuckGo results
    filtered_results: list  # after quality filtering
    answer: str             # generated answer
    final_answer: str       # answer + citations


# ── Node 0: Router ────────────────────────────────────────────────────────────

SEARCH_KEYWORDS = [
    # Chinese
    "今天", "最新", "现在", "价格", "新闻", "天气", "几点", "多少钱", "实时",
    # English
    "today", "latest", "current", "price", "news", "weather", "right now",
    "how much", "live", "real-time", "recently", "this week", "2026",
]

def router_node(state: AgentState) -> AgentState:
    """Decide whether web search is needed."""
    question = state["question"]

    # Layer 1: keyword rules (fast, no LLM call)
    if any(kw in question for kw in SEARCH_KEYWORDS):
        return {**state, "should_search": True}

    # Layer 2: ask the LLM
    prompt = (
        "Does this question require up-to-date information from the internet to answer accurately? "
        "Reply with YES or NO only. The question may be in English or Chinese.\n"
        f"Question: {question}"
    )
    response = llm.invoke(prompt).content.strip().upper()
    return {**state, "should_search": "YES" in response}


# ── Node 1: Query Rewriter ────────────────────────────────────────────────────

def query_rewriter_node(state: AgentState) -> AgentState:
    """Rewrite the user question into search-engine-friendly keywords."""
    prompt = (
        "Rewrite the following question into concise search engine keywords. "
        "Output the keywords only, no explanation.\n"
        f"Question: {state['question']}"
    )
    search_query = llm.invoke(prompt).content.strip()
    return {**state, "search_query": search_query}


# ── Node 2: Web Search ────────────────────────────────────────────────────────

def web_search_node(state: AgentState) -> AgentState:
    """Run DuckDuckGo search using the rewritten query."""
    results = web_search(state["search_query"], max_results=5)
    return {**state, "raw_results": results}


# ── Node 3: Result Filter ─────────────────────────────────────────────────────

def result_filter_node(state: AgentState) -> AgentState:
    """Filter out low-quality results using rules (no LLM)."""
    filtered = filter_results(state["raw_results"])
    return {**state, "filtered_results": filtered}


# ── Node 4: Answer Generator ──────────────────────────────────────────────────

def answer_generator_node(state: AgentState) -> AgentState:
    """Generate an answer, with or without search context."""
    if state.get("filtered_results"):
        context = format_for_llm(state["filtered_results"])
        prompt = (
            f"Use the following search results to answer the question accurately.\n\n"
            f"Search Results:\n{context}\n\n"
            f"Question: {state['question']}\n"
            f"Answer:"
        )
    else:
        prompt = state["question"]

    answer = llm.invoke(prompt).content.strip()
    return {**state, "answer": answer}


# ── Node 5: Citation Formatter ────────────────────────────────────────────────

def citation_formatter_node(state: AgentState) -> AgentState:
    """Append source URLs to the answer."""
    if not state.get("filtered_results"):
        return {**state, "final_answer": state["answer"]}

    sources = "\n".join([
        f"  [{i+1}] {r.get('title', 'Source')} - {r.get('href', '')}"
        for i, r in enumerate(state["filtered_results"])
    ])
    final_answer = f"{state['answer']}\n\nSources:\n{sources}"
    return {**state, "final_answer": final_answer}


# ── Routing Function ──────────────────────────────────────────────────────────

def route_after_router(state: AgentState) -> Literal["query_rewriter", "answer_generator"]:
    return "query_rewriter" if state["should_search"] else "answer_generator"


def route_after_answer(state: AgentState) -> Literal["citation_formatter", "__end__"]:
    return "citation_formatter" if state.get("filtered_results") else "__end__"


# ── Build Graph ───────────────────────────────────────────────────────────────

graph = StateGraph(AgentState)

graph.add_node("router",             router_node)
graph.add_node("query_rewriter",     query_rewriter_node)
graph.add_node("web_search",         web_search_node)
graph.add_node("result_filter",      result_filter_node)
graph.add_node("answer_generator",   answer_generator_node)
graph.add_node("citation_formatter", citation_formatter_node)

graph.set_entry_point("router")

graph.add_conditional_edges("router", route_after_router, {
    "query_rewriter":   "query_rewriter",
    "answer_generator": "answer_generator",
})
graph.add_edge("query_rewriter",   "web_search")
graph.add_edge("web_search",       "result_filter")
graph.add_edge("result_filter",    "answer_generator")
graph.add_conditional_edges("answer_generator", route_after_answer, {
    "citation_formatter": "citation_formatter",
    "__end__":            END,
})
graph.add_edge("citation_formatter", END)

agent = graph.compile()


# ── Main Chat Loop ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Agent ready. Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() == "exit":
            break

        initial_state: AgentState = {
            "question":         question,
            "should_search":    False,
            "search_query":     "",
            "raw_results":      [],
            "filtered_results": [],
            "answer":           "",
            "final_answer":     "",
        }

        result = agent.invoke(initial_state)

        mode = "🔍 Searched web" if result["should_search"] else "💭 Direct answer"
        print(f"\n[{mode}]")
        print(f"Agent: {result['final_answer'] or result['answer']}\n")
