from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END

from model import get_llm
from search import web_search, filter_results, format_for_llm as format_web
from knowledge import query_knowledge, format_for_llm as format_local

llm = get_llm()


# ── State ─────────────────────────────────────────────────────

class AgentState(TypedDict):
    question:         str
    route:            str   # "local", "web", "direct"
    search_query:     str
    raw_results:      list
    filtered_results: list
    local_results:    list
    answer:           str
    final_answer:     str


# ── Node 0: Router (3-way) ────────────────────────────────────

WEB_KEYWORDS = [
    # Chinese
    "今天", "最新", "现在", "价格", "新闻", "天气", "几点", "多少钱", "实时",
    # English
    "today", "latest", "current", "price", "news", "weather", "right now",
    "how much", "live", "real-time", "recently", "this week", "2026",
]

LOCAL_KEYWORDS = [
    # Chinese
    "我之前", "我的笔记", "我记录", "之前存的", "我写的", "本地文件",
    # English
    "my notes", "i saved", "i wrote", "my docs", "previously", "local",
]

def router_node(state: AgentState) -> AgentState:
    question = state["question"].lower()

    # Layer 1: keyword rules
    if any(kw in question for kw in LOCAL_KEYWORDS):
        return {**state, "route": "local"}
    if any(kw in question for kw in WEB_KEYWORDS):
        return {**state, "route": "web"}

    # Layer 2: ask LLM
    prompt = (
        "Classify this question into one of three categories:\n"
        "- LOCAL: asks about the user's personal notes, saved files, or past work\n"
        "- WEB: requires real-time or up-to-date information from the internet\n"
        "- DIRECT: can be answered from general knowledge\n\n"
        "Reply with LOCAL, WEB, or DIRECT only.\n"
        f"Question: {state['question']}"
    )
    response = llm.invoke(prompt).content.strip().upper()

    if "LOCAL" in response:
        route = "local"
    elif "WEB" in response:
        route = "web"
    else:
        route = "direct"

    return {**state, "route": route}


# ── Node 1a: Local Search ─────────────────────────────────────

def local_search_node(state: AgentState) -> AgentState:
    """Search local knowledge base (ChromaDB)."""
    results = query_knowledge(state["question"], top_k=3)
    return {**state, "local_results": results}


# ── Node 1b: Query Rewriter (web path) ───────────────────────

def query_rewriter_node(state: AgentState) -> AgentState:
    prompt = (
        "Rewrite the following question into concise search engine keywords. "
        "Output the keywords only, no explanation.\n"
        f"Question: {state['question']}"
    )
    search_query = llm.invoke(prompt).content.strip()
    return {**state, "search_query": search_query}


# ── Node 2: Web Search ────────────────────────────────────────

def web_search_node(state: AgentState) -> AgentState:
    results = web_search(state["search_query"], max_results=5)
    return {**state, "raw_results": results}


# ── Node 3: Result Filter ─────────────────────────────────────

def result_filter_node(state: AgentState) -> AgentState:
    filtered = filter_results(state["raw_results"])
    return {**state, "filtered_results": filtered}


# ── Node 4: Answer Generator ──────────────────────────────────

def answer_generator_node(state: AgentState) -> AgentState:
    if state.get("local_results"):
        context = format_local(state["local_results"])
        prompt = (
            f"Answer the question using the following information from the local knowledge base.\n\n"
            f"Knowledge Base:\n{context}\n\n"
            f"Question: {state['question']}\nAnswer:"
        )
    elif state.get("filtered_results"):
        context = format_web(state["filtered_results"])
        prompt = (
            f"Use the following search results to answer the question accurately.\n\n"
            f"Search Results:\n{context}\n\n"
            f"Question: {state['question']}\nAnswer:"
        )
    else:
        prompt = state["question"]

    answer = llm.invoke(prompt).content.strip()
    return {**state, "answer": answer}


# ── Node 5: Citation Formatter ────────────────────────────────

def citation_formatter_node(state: AgentState) -> AgentState:
    if state.get("local_results"):
        sources = "\n".join([
            f"  [{i+1}] {r['source']}"
            for i, r in enumerate(state["local_results"])
        ])
        final = f"{state['answer']}\n\nSources (local):\n{sources}"
    elif state.get("filtered_results"):
        sources = "\n".join([
            f"  [{i+1}] {r.get('title', 'Source')} - {r.get('href', '')}"
            for i, r in enumerate(state["filtered_results"])
        ])
        final = f"{state['answer']}\n\nSources (web):\n{sources}"
    else:
        final = state["answer"]

    return {**state, "final_answer": final}


# ── Routing Functions ─────────────────────────────────────────

def route_after_router(state: AgentState) -> Literal["local_search", "query_rewriter", "answer_generator"]:
    return {
        "local":  "local_search",
        "web":    "query_rewriter",
        "direct": "answer_generator",
    }[state["route"]]


def route_after_answer(state: AgentState) -> Literal["citation_formatter", "__end__"]:
    has_sources = bool(state.get("local_results") or state.get("filtered_results"))
    return "citation_formatter" if has_sources else "__end__"


# ── Build Graph ───────────────────────────────────────────────

graph = StateGraph(AgentState)

graph.add_node("router",             router_node)
graph.add_node("local_search",       local_search_node)
graph.add_node("query_rewriter",     query_rewriter_node)
graph.add_node("web_search",         web_search_node)
graph.add_node("result_filter",      result_filter_node)
graph.add_node("answer_generator",   answer_generator_node)
graph.add_node("citation_formatter", citation_formatter_node)

graph.set_entry_point("router")

graph.add_conditional_edges("router", route_after_router, {
    "local_search":    "local_search",
    "query_rewriter":  "query_rewriter",
    "answer_generator":"answer_generator",
})
graph.add_edge("local_search",     "answer_generator")
graph.add_edge("query_rewriter",   "web_search")
graph.add_edge("web_search",       "result_filter")
graph.add_edge("result_filter",    "answer_generator")
graph.add_conditional_edges("answer_generator", route_after_answer, {
    "citation_formatter": "citation_formatter",
    "__end__":            END,
})
graph.add_edge("citation_formatter", END)

agent = graph.compile()


# ── Main Chat Loop ────────────────────────────────────────────

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
            "route":            "",
            "search_query":     "",
            "raw_results":      [],
            "filtered_results": [],
            "local_results":    [],
            "answer":           "",
            "final_answer":     "",
        }

        result = agent.invoke(initial_state)

        icons = {"local": "📚 Local KB", "web": "🔍 Web search", "direct": "💭 Direct"}
        mode = icons.get(result["route"], "💭 Direct")
        print(f"\n[{mode}]")
        print(f"Agent: {result['final_answer'] or result['answer']}\n")
