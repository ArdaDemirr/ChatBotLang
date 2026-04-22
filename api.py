import os
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import TypedDict, Optional, Literal
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

# LLMs
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

# Errors
from google.api_core.exceptions import ResourceExhausted, TooManyRequests

# Your DB utils
from database_utils import get_schema, run_query

load_dotenv()
app = FastAPI()

user_memory_bank: dict[int, list[dict]] = {}

# ---------------------------------------------------------------------------
# REAL FALLBACK CHAIN (fixed)
# ---------------------------------------------------------------------------

LLM_CHAIN = [
    # ── 1. Groq (Fastest & Free) ──────────────────────────────────────
    {"name": "llama-3.3-70b", "model": "llama-3.3-70b-versatile", "provider": "groq"}, # Updated to 3.3
    {"name": "llama-3.1-8b", "model": "llama-3.1-8b-instant", "provider": "groq"},

    # ── 2. Google Gemini (Backup Engine) ──────────────────────────────
    {"name": "gemini-2.5-flash", "model": "gemini-2.5-flash", "provider": "google"},

    # ── 3. Local Fallback (Safety net if offline) ─────────────────────
    {"name": "phi4-mini", "model": "phi4-mini", "provider": "ollama"},
]

_QUOTA_KEYWORDS = ("quota", "rate", "limit", "429", "exhausted")

# ---------------------------------------------------------------------------
# LLM FACTORY
# ---------------------------------------------------------------------------

def _make_llm(entry: dict):
    provider = entry["provider"]

    if provider == "ollama":
        return ChatOllama(model=entry["model"], temperature=0)

    if provider == "groq":
        return ChatGroq(
            model=entry["model"],
            groq_api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
        )

    if provider == "google":
        return ChatGoogleGenerativeAI(
            model=entry["model"],
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0,
        )


def llm_invoke(prompt: str) -> str:
    for entry in LLM_CHAIN:
        try:
            llm = _make_llm(entry)
            response = llm.invoke(prompt)

            print(f"[LLM] OK → {entry['name']}")
            return response.content.strip()

        except (ResourceExhausted, TooManyRequests) as e:
            print(f"[LLM] LIMIT → {entry['name']}: {e}")
            continue

        except Exception as e:
            msg = str(e).lower()

            if any(k in msg for k in _QUOTA_KEYWORDS):
                print(f"[LLM] QUOTA → {entry['name']}: {e}")
                continue

            print(f"[LLM] ERROR → {entry['name']}: {type(e).__name__} | {e}")
            continue

    raise RuntimeError("All LLM backends failed.")

# ---------------------------------------------------------------------------
# STATE
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    question: str
    user_role: str
    user_id: int
    history_text: str
    intent: Literal["greeting", "db_query"]
    sql_query: str
    db_results: Optional[dict]
    error: Optional[str]
    final_answer: str

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def is_greeting(text: str) -> bool:
    return text.lower().strip() in {"hi", "hello", "hey", "selam", "merhaba"}

def detect_language(text: str) -> str:
    return "tr" if re.search(r"[çşğüöıÇŞĞÜÖİ]", text) else "en"

# ---------------------------------------------------------------------------
# NODES
# ---------------------------------------------------------------------------

def intent_classifier(state: AgentState):
    return {"intent": "greeting" if is_greeting(state["question"]) else "db_query"}

def greeting_handler(state: AgentState):
    return {"final_answer": "Merhaba! Ne öğrenmek istersiniz?"}

def sql_writer(state: AgentState):
    schema = get_schema()
    role = state["user_role"]
    user_id = state["user_id"]

    if role == "GUEST":
        role_context = (
            "This is a guest (non-logged-in) user. "
            "Only answer questions about general public data: products, categories, prices, and bestsellers. "
            "Do NOT filter by user_id or include personal/order data."
        )
    elif role == "INDIVIDUAL":
        role_context = (
            f"The current user is a CUSTOMER with user_id = {user_id}. "
            f"Always filter results to only show this user's data "
            f"(use user_id or customer_id = {user_id} where applicable)."
        )
    elif role == "ADMIN":
        role_context = (
            "This user is a PLATFORM ADMINISTRATOR. "
            "They have FULL unrestricted access to ALL tables and ALL data. "
            "Do NOT add any user_id or store_id filters unless explicitly asked. "
            "They can see: all orders, all users, all products, all reviews, all stores, all shipments. "
            "For analytics questions like 'last 5 orders', 'best selling product', 'total revenue', "
            "write aggregate queries with ORDER BY, COUNT, SUM, GROUP BY as needed. "
            "Never restrict data access for ADMIN users."
        )
    else:
        role_context = (
            f"The current user is a STORE OWNER with user_id = {user_id}. "
            f"Filter results to their store's data using seller_id or store_id. "
            f"For aggregate queries like 'best selling product', "
            f"join orders with products and group/order correctly."
        )

    history_block = (
        f"Conversation history:\n{state['history_text']}\n\n"
        if state.get("history_text") else ""
    )

    prompt = f"""You are a MySQL query generator for an e-commerce database.

Schema:
{schema}

{history_block}Context: {role_context}
Question: {state["question"]}

Instructions:
- Write ONE valid MySQL SELECT statement.
- Do NOT use DROP, DELETE, UPDATE, INSERT, ALTER, or TRUNCATE.
- Return ONLY the raw SQL. No markdown, no backticks, no explanation.
"""

    raw = llm_invoke(prompt)
    print(f"\n[SQL_WRITER] Q: {state['question']}")
    print(f"[SQL_WRITER] Raw:\n{raw}")

    # --- THE CRITICAL REGEX FIX ---
    # Replaced the destructive re.sub with safe string replacement
    raw = raw.replace("```sql", "").replace("```mysql", "").replace("```", "").strip()
    match = re.search(r"(SELECT\b.*)", raw, re.IGNORECASE | re.DOTALL)

    if not match:
        print("[SQL_WRITER] Failed to find SELECT statement.")
        return {"sql_query": "NONE", "error": "SQL generation failed"}

    sql = match.group(1).strip()
    if not sql.endswith(";"):
        sql += ";"
        
    print(f"[SQL_WRITER] SQL: {sql}\n")
    return {"sql_query": sql, "error": None}

def security_checker(state: AgentState) -> dict:
    sql = state.get("sql_query", "").lower()
    if sql == "none" or not sql:
        return {}

    forbidden = ["drop", "delete", "update", "insert", "alter", "truncate"]
    if any(word in sql for word in forbidden):
        return {"error": "SECURITY: Database modification blocked.", "sql_query": "NONE"}

    # GUEST users can only query public tables (products, categories)
    # Block any access to personal/transactional tables
    guest_blocked_tables = ["orders", "users", "shipments", "order_items", "payments", "stores"]
    if state["user_role"] == "GUEST" and any(t in sql for t in guest_blocked_tables):
        return {"error": "SECURITY: Guest access to personal data denied.", "sql_query": "NONE"}

    # INDIVIDUAL users cannot access system-level tables
    if state["user_role"] == "INDIVIDUAL" and any(t in sql for t in ["users", "stores"]):
        return {"error": "SECURITY: Access to system tables denied.", "sql_query": "NONE"}

    return {"error": None}

def db_executor(state: AgentState):
    if state["sql_query"] == "NONE" or state.get("error"):
        return {"db_results": {}}

    print(f"[DB_EXECUTOR] Running: {state['sql_query']}")
    result = run_query(state["sql_query"])
    print(f"[DB_EXECUTOR] Result: {result}\n")

    if "error" in result:
        return {"error": result["error"], "db_results": None}

    return {"db_results": result}

def summarizer(state: AgentState):
    error = state.get("error") or ""
    lang = detect_language(state["question"])

    if "SECURITY" in error:
        return {"final_answer": (
            "Üzgünüm, bu bilgilere erişme yetkiniz bulunmamaktadır."
            if lang == "tr" else
            "Sorry, you don't have permission to access this data."
        )}

    if "SQL generation failed" in error or state.get("sql_query") == "NONE":
        return {"final_answer": (
            "Sorunuzu anlayamadım veya veritabanında bir hata oluştu."
            if lang == "tr" else
            "I couldn't understand your question or a database error occurred."
        )}

    db_results = state.get("db_results")

    if not db_results or db_results == {}:
        return {"final_answer": (
            "Bu sorgu için veritabanında herhangi bir kayıt bulunamadı."
            if lang == "tr" else "No records were found for your query."
        )}

    if isinstance(db_results, dict) and db_results.get("message"):
        return {"final_answer": (
            "Bu kriterlere uyan kayıt bulunamadı."
            if lang == "tr" else "No matching records found."
        )}

    lang_instruction = (
        "Yanıtı kısa ve doğal Türkçe bir cümleyle ver."
        if lang == "tr" else
        "Respond in a short, natural English sentence."
    )

    prompt = f"""
Question: {state["question"]}
Result: {state["db_results"]}

Summarize the result as a helpful answer. {lang_instruction}
Do not mention SQL, databases, or technical terms.
"""

    return {"final_answer": llm_invoke(prompt)}

# ---------------------------------------------------------------------------
# GRAPH
# ---------------------------------------------------------------------------

workflow = StateGraph(AgentState)

workflow.add_node("intent", intent_classifier)
workflow.add_node("greet", greeting_handler)
workflow.add_node("sql", sql_writer)
workflow.add_node("security", security_checker)
workflow.add_node("db", db_executor)
workflow.add_node("sum", summarizer)

workflow.set_entry_point("intent")

workflow.add_conditional_edges(
    "intent",
    lambda s: "greet" if s["intent"] == "greeting" else "sql"
)

workflow.add_edge("greet", END)
workflow.add_edge("sql", "security")

def route_after_security(state: AgentState) -> str:
    return "sum" if state.get("error") else "db"

workflow.add_conditional_edges("security", route_after_security)
workflow.add_edge("db", "sum")
workflow.add_edge("sum", END)

ai_brain = workflow.compile()

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    user_role: str
    user_id: int

@app.post("/agent/ask")
async def ask_agent(request: ChatRequest):
    try:
        uid = request.user_id

        if uid not in user_memory_bank:
            user_memory_bank[uid] = []

        history = user_memory_bank[uid]
        history_str = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in history
        )

        state: AgentState = {
            "question": request.message,
            "user_role": request.user_role,
            "user_id": uid,
            "history_text": history_str,
            "intent": "db_query",
            "sql_query": "",
            "db_results": None,
            "error": None,
            "final_answer": "",
        }

        result = ai_brain.invoke(state, {"recursion_limit": 15})
        answer = result.get("final_answer", "Bir hata oluştu.")

        if result.get("intent") != "greeting":
            user_memory_bank[uid].append({"role": "user", "content": request.message})
            user_memory_bank[uid].append({"role": "ai", "content": answer})
            if len(user_memory_bank[uid]) > 6:
                user_memory_bank[uid] = user_memory_bank[uid][-6:]

        return {"reply": answer}

    except Exception as e:
        print("CRASH:", e)
        raise HTTPException(500, "Server error")

# ---------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)