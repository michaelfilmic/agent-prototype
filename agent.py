import os
import re
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END

from model import get_llm
from search import web_search, filter_results, format_for_llm as format_web
from knowledge import query_knowledge, format_for_llm as format_local
from scrubber import validate_and_normalize, scrub_and_save
from scrubber import _read_file                        # internal helper
from excel_filter import extract_filter_criteria, correct_criteria, apply_filters, format_filter_report

llm = get_llm()


# ── State ──────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    question:         str
    route:            str    # "local" | "web" | "direct" | "file"
    intent:           str    # "scrub" | "filter" | "filter+scrub"
    search_query:     str
    raw_results:      list
    filtered_results: list
    local_results:    list
    answer:           str
    final_answer:     str
    file_path:        str    # normalized path; updated by excel_process after filtering
    file_ext:         str    # ".csv" / ".xlsx" / ".xls"
    filter_criteria:  dict   # structured filter rules extracted by LLM


# ── Keyword lists ──────────────────────────────────────────────────────────────

WEB_KEYWORDS = [
    "今天", "最新", "现在", "价格", "新闻", "天气", "几点", "多少钱", "实时",
    "today", "latest", "current", "price", "news", "weather", "right now",
    "how much", "live", "real-time", "recently", "this week", "2026",
]

LOCAL_KEYWORDS = [
    "我之前", "我的笔记", "我记录", "之前存的", "我写的", "本地文件",
    "my notes", "i saved", "i wrote", "my docs", "previously", "local",
]

SCRUB_KEYWORDS = [
    "清除", "脱敏", "去除敏感", "隐藏", "敏感信息",
    "scrub", "redact", "remove sensitive", "anonymize", "anonymise",
    "clean sensitive", "mask", "sanitize", "sanitise",
]

FILTER_KEYWORDS = [
    # Chinese
    "筛选", "过滤", "找出", "显示", "只要", "只看", "查找",
    "月", "月份", "一月", "二月", "三月", "四月", "五月", "六月",
    "七月", "八月", "九月", "十月", "十一月", "十二月",
    # English
    "filter", "show me", "find", "only", "between", "date range",
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    "month", "year", "greater than", "less than", "over", "under", "above", "below",
]

# File path regex — prefix is REQUIRED to avoid swallowing preceding text
_FILE_PATH_RE = re.compile(
    r"""
      (?:^|(?<=[^/\\A-Za-z0-9_.~-]))   # preceded by a non-path char (or start)
      (
        (?:
          [A-Za-z]:[/\\]                # Windows absolute:  C:\  or  C:/
        | /[A-Za-z]/                    # Git Bash / MSYS2:  /c/
        | /                             # Unix absolute:     /home/...
        | \.{1,2}[/\\]                  # relative:          ./  or  ../
        )
        [^\s"']+                        # rest of path
        \.(?:csv|xlsx|xls)              # supported extension
      )
      (?=$|\s|["'])                     # followed by end / space / quote
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _extract_file_path(text: str) -> str | None:
    m = _FILE_PATH_RE.search(text)
    return m.group(1).strip() if m else None


# ── Node 0: Router ─────────────────────────────────────────────────────────────

def router_node(state: AgentState) -> AgentState:
    question = state["question"]
    q_lower  = question.lower()

    file_path    = _extract_file_path(question)
    has_scrub_kw = any(kw in q_lower for kw in SCRUB_KEYWORDS)
    has_filter_kw = any(kw in q_lower for kw in FILTER_KEYWORDS)

    if file_path or has_scrub_kw or has_filter_kw:
        # Determine intent
        if has_scrub_kw and has_filter_kw:
            intent = "filter+scrub"
        elif has_filter_kw:
            intent = "filter"
        else:
            intent = "scrub"

        return {**state,
                "route": "file",
                "intent": intent,
                "search_query": file_path or ""}

    # Standard local / web / direct routing
    if any(kw in q_lower for kw in LOCAL_KEYWORDS):
        return {**state, "route": "local",  "intent": ""}
    if any(kw in q_lower for kw in WEB_KEYWORDS):
        return {**state, "route": "web",    "intent": ""}

    prompt = (
        "Classify this question into one of three categories:\n"
        "- LOCAL: asks about the user's personal notes, saved files, or past work\n"
        "- WEB: requires real-time or up-to-date information from the internet\n"
        "- DIRECT: can be answered from general knowledge\n\n"
        "Reply with LOCAL, WEB, or DIRECT only.\n"
        f"Question: {state['question']}"
    )
    response = llm.invoke(prompt).content.strip().upper()
    route = "local" if "LOCAL" in response else "web" if "WEB" in response else "direct"
    return {**state, "route": route, "intent": ""}


# ── Node 1: Excel Input ────────────────────────────────────────────────────────

def excel_input_node(state: AgentState) -> AgentState:
    """Extract the file path from the question, normalize it, and validate it exists."""
    raw_path = state.get("search_query", "").strip()
    if not raw_path:
        raw_path = _extract_file_path(state["question"]) or ""

    if not raw_path:
        msg = (
            "Please provide the full path to a .csv, .xlsx, or .xls file.\n"
            "Example:  scrub C:\\Users\\you\\Downloads\\statement.csv"
        )
        return {**state, "answer": msg, "final_answer": msg,
                "file_path": "", "file_ext": ""}

    result = validate_and_normalize(raw_path)
    if isinstance(result, str):          # error string
        return {**state, "answer": result, "final_answer": result,
                "file_path": "", "file_ext": ""}

    normalized_path, ext = result
    print(f"  [excel_input] Validated: {normalized_path}  ({state['intent']})")
    return {**state, "file_path": normalized_path, "file_ext": ext}


# ── Node 2: Excel Process (filter) ─────────────────────────────────────────────

def excel_process_node(state: AgentState) -> AgentState:
    """
    Read the file and apply LLM-extracted filters when intent includes 'filter'.
    Saves the (possibly filtered) file and updates file_path in state so the
    downstream scrub node always works on the correct data.
    """
    file_path = state["file_path"]
    ext       = state["file_ext"]
    intent    = state["intent"]

    df = _read_file(file_path, ext)
    original_count = len(df)
    criteria = {"filters": [], "description": "No filter applied"}

    if "filter" in intent:
        print("  [excel_process] Asking LLM to extract filter criteria…")
        sample   = df.head(3).astype(str).to_dict(orient="records")
        criteria = extract_filter_criteria(
            state["question"], df.columns.tolist(), sample, llm
        )
        print(f"  [excel_process] LLM raw criteria : {criteria}")
        criteria = correct_criteria(criteria, df)
        print(f"  [excel_process] Corrected criteria: {criteria}")
        df = apply_filters(df, criteria)
        print(f"  [excel_process] {original_count} → {len(df)} rows after filter")

    # Save filtered result (or pass-through) to a working file
    base     = os.path.splitext(file_path)[0]
    out_path = base + "_filtered" + ext
    if ext == ".csv":
        df.to_csv(out_path, index=False)
    else:
        df.to_excel(out_path, index=False, engine="openpyxl")

    report = format_filter_report(criteria, original_count, len(df), out_path)

    # If intent is filter-only, set final_answer now; scrub node won't run
    if intent == "filter":
        return {**state,
                "file_path":       out_path,
                "file_ext":        ext,
                "filter_criteria": criteria,
                "answer":          report,
                "final_answer":    report}

    # filter+scrub: carry the report forward, scrub will append its own report
    return {**state,
            "file_path":       out_path,
            "file_ext":        ext,
            "filter_criteria": criteria,
            "answer":          report}


# ── Node 3: Scrub ──────────────────────────────────────────────────────────────

def scrub_node(state: AgentState) -> AgentState:
    """Run LLM + regex detection and redact sensitive data from the working file."""
    scrub_report = scrub_and_save(state["file_path"], state["file_ext"], llm=llm)

    # Prepend filter report if it exists
    prior = state.get("answer", "")
    full_answer = f"{prior}\n\n{scrub_report}" if prior.strip() else scrub_report
    return {**state, "answer": full_answer, "final_answer": full_answer}


# ── Nodes: Local / Web / Direct ────────────────────────────────────────────────

def local_search_node(state: AgentState) -> AgentState:
    results = query_knowledge(state["question"], top_k=3)
    return {**state, "local_results": results}


def query_rewriter_node(state: AgentState) -> AgentState:
    prompt = (
        "Rewrite the following question into concise search engine keywords. "
        "Output the keywords only, no explanation.\n"
        f"Question: {state['question']}"
    )
    return {**state, "search_query": llm.invoke(prompt).content.strip()}


def web_search_node(state: AgentState) -> AgentState:
    results = web_search(state["search_query"], max_results=5)
    return {**state, "raw_results": results}


def result_filter_node(state: AgentState) -> AgentState:
    return {**state, "filtered_results": filter_results(state["raw_results"])}


def answer_generator_node(state: AgentState) -> AgentState:
    if state.get("local_results"):
        context = format_local(state["local_results"])
        prompt  = (
            f"Answer the question using the following information from the local knowledge base.\n\n"
            f"Knowledge Base:\n{context}\n\n"
            f"Question: {state['question']}\nAnswer:"
        )
    elif state.get("filtered_results"):
        context = format_web(state["filtered_results"])
        prompt  = (
            f"Use the following search results to answer the question accurately.\n\n"
            f"Search Results:\n{context}\n\n"
            f"Question: {state['question']}\nAnswer:"
        )
    else:
        prompt = state["question"]
    return {**state, "answer": llm.invoke(prompt).content.strip()}


def citation_formatter_node(state: AgentState) -> AgentState:
    if state.get("local_results"):
        sources = "\n".join(
            f"  [{i+1}] {r['source']}"
            for i, r in enumerate(state["local_results"])
        )
        final = f"{state['answer']}\n\nSources (local):\n{sources}"
    elif state.get("filtered_results"):
        sources = "\n".join(
            f"  [{i+1}] {r.get('title','Source')} - {r.get('href','')}"
            for i, r in enumerate(state["filtered_results"])
        )
        final = f"{state['answer']}\n\nSources (web):\n{sources}"
    else:
        final = state["answer"]
    return {**state, "final_answer": final}


# ── Routing functions ──────────────────────────────────────────────────────────

def route_after_router(state: AgentState) -> Literal[
    "excel_input", "local_search", "query_rewriter", "answer_generator"
]:
    return {
        "file":   "excel_input",
        "local":  "local_search",
        "web":    "query_rewriter",
        "direct": "answer_generator",
    }[state["route"]]


def route_after_excel_input(state: AgentState) -> Literal["excel_process", "__end__"]:
    return "excel_process" if state.get("file_path") else "__end__"


def route_after_excel_process(state: AgentState) -> Literal["scrub", "__end__"]:
    return "scrub" if "scrub" in state.get("intent", "") else "__end__"


def route_after_answer(state: AgentState) -> Literal["citation_formatter", "__end__"]:
    has_sources = bool(state.get("local_results") or state.get("filtered_results"))
    return "citation_formatter" if has_sources else "__end__"


# ── Build graph ────────────────────────────────────────────────────────────────

graph = StateGraph(AgentState)

graph.add_node("router",             router_node)
graph.add_node("excel_input",        excel_input_node)
graph.add_node("excel_process",      excel_process_node)
graph.add_node("scrub",              scrub_node)
graph.add_node("local_search",       local_search_node)
graph.add_node("query_rewriter",     query_rewriter_node)
graph.add_node("web_search",         web_search_node)
graph.add_node("result_filter",      result_filter_node)
graph.add_node("answer_generator",   answer_generator_node)
graph.add_node("citation_formatter", citation_formatter_node)

graph.set_entry_point("router")

graph.add_conditional_edges("router", route_after_router, {
    "excel_input":     "excel_input",
    "local_search":    "local_search",
    "query_rewriter":  "query_rewriter",
    "answer_generator":"answer_generator",
})
graph.add_conditional_edges("excel_input", route_after_excel_input, {
    "excel_process": "excel_process",
    "__end__":       END,
})
graph.add_conditional_edges("excel_process", route_after_excel_process, {
    "scrub":   "scrub",
    "__end__": END,
})
graph.add_edge("scrub",            END)
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


# ── Main chat loop ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Agent ready. Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() == "exit":
            break

        initial_state: AgentState = {
            "question":        question,
            "route":           "",
            "intent":          "",
            "search_query":    "",
            "raw_results":     [],
            "filtered_results":[],
            "local_results":   [],
            "answer":          "",
            "final_answer":    "",
            "file_path":       "",
            "file_ext":        "",
            "filter_criteria": {},
        }

        result = agent.invoke(initial_state)

        icons = {
            "file":   f"📂 File ({result.get('intent', '')})",
            "local":  "📚 Local KB",
            "web":    "🔍 Web search",
            "direct": "💭 Direct",
        }
        mode = icons.get(result["route"], "💭 Direct")
        print(f"\n[{mode}]")
        print(f"Agent: {result['final_answer'] or result['answer']}\n")
