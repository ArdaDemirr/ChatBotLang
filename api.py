import os
import re
import time
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
# REAL FALLBACK CHAIN
# ---------------------------------------------------------------------------

LLM_CHAIN = [
    {"name": "llama-3.3-70b", "model": "llama-3.3-70b-versatile", "provider": "groq"},
    {"name": "llama-3.1-8b", "model": "llama-3.1-8b-instant", "provider": "groq"},
    {"name": "gemini-2.5-flash", "model": "gemini-2.5-flash", "provider": "google"},
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
        return ChatGroq(model=entry["model"], groq_api_key=os.getenv("GROQ_API_KEY"), temperature=0)
    if provider == "google":
        return ChatGoogleGenerativeAI(model=entry["model"], google_api_key=os.getenv("GOOGLE_API_KEY"), temperature=0)

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
    lang = detect_language(state["question"])
    return {"final_answer": "Merhaba! Ne öğrenmek istersiniz?" if lang == "tr" else "Hello! What would you like to know?"}

# --- SQL_WRITER: LEFT UNTOUCHED AS REQUESTED ---
def sql_writer(state: AgentState):
    schema = get_schema()
    role = state["user_role"]
    user_id = state["user_id"]
    error = state.get("error")

    if role == "GUEST":
        role_context = (
            "You are talking to a GUEST (not logged in). "
            "They can ONLY ask about public products and categories. "
            "If they ask for orders, users, shipments, or payments, YOU MUST RETURN: UNAUTHORIZED_GUEST"
        )
    elif role == "INDIVIDUAL":
        role_context = (
            f"You are talking to a CUSTOMER with user_id = {user_id}. "
            f"the products and category of them with reviews made to them are public data can return queries like most expensive one or how many producs are there etc (but do not return their order or shipment values made by other users)"
            f"They can ONLY ask about their own orders, reviews, or shipments. "
            f"If they explicitly mention another user's name, email, or ID, YOU MUST RETURN: UNAUTHORIZED_USER\n"
            f"If they ask for global stats or admin data, YOU MUST RETURN: UNAUTHORIZED_ADMIN\n"
            f"If the request is allowed, you MUST filter by user_id = {user_id}."
        )
    elif role == "ADMIN":
        role_context = (
            "You are talking to an ADMIN. They have full access to everything. "
            "No restrictions. Do not force any filters unless asked."
        )
    else: # CORPORATE
        role_context = (
            f"You are talking to a STORE OWNER with user_id = {user_id}. "
            f"the products and category of them with reviews made to them are public data can return queries like most expensive one or how many producs are there etc (but do not return their order or shipment values made by other users)"
            f"If they ask for another store's private data, YOU MUST RETURN: UNAUTHORIZED_STORE\n"
            f"If they ask for global users or admin data, YOU MUST RETURN: UNAUTHORIZED_ADMIN\n"
            f"They can ONLY ask about their own stores orders, reviews, or shipments. "
            f"If allowed, you MUST filter by store_id or seller_id."
        )

    history_block = (
        f"Conversation history:\n{state['history_text']}\n\n"
        if state.get("history_text") else ""
    )

    if error and state.get("sql_query") and state["sql_query"] != "NONE":
        prompt = f"""You are a MySQL query generator.
Schema:
{schema}

Access Rules:
{role_context}

Question: {state["question"]}

WARNING! Your previous query failed with this error:
{error}
Previous Bad Query:
{state["sql_query"]}

Instructions:
1. FIRST, check the Access Rules. If the question is forbidden, return ONLY the exact UNAUTHORIZED string. DO NOT write SQL.
2. If permitted, fix the SQL so it works with the schema.
3. Return ONLY the raw SQL or the UNAUTHORIZED string. No markdown, no explanation.
"""
    else:
        prompt = f"""You are a MySQL query generator for an e-commerce database.

Schema:
{schema}

{history_block}Access Rules:
{role_context}

Question: {state["question"]}

Instructions:
1. CRITICAL: First, evaluate the Access Rules. If the user's question violates their permissions (like asking for another user's data), you MUST NOT write SQL. Instead, return ONLY the exact UNAUTHORIZED string mentioned in the rules.
2. If the request is permitted, write ONE valid MySQL SELECT statement.
3. Return ONLY the raw SQL (or the UNAUTHORIZED string). No markdown, no backticks, no explanation.
"""

    raw = llm_invoke(prompt)
    print(f"\n[SQL_WRITER] Q: {state['question']}")
    print(f"\n[SQL_WRITER] Raw:\n{raw}")

    # --- INTERCEPT REFUSALS BEFORE REGEX ---
    raw_upper = raw.upper()
    if "UNAUTHORIZED_USER" in raw_upper: return {"sql_query": "NONE", "error": "SECURITY_USER"}
    if "UNAUTHORIZED_STORE" in raw_upper: return {"sql_query": "NONE", "error": "SECURITY_STORE"}
    if "UNAUTHORIZED_ADMIN" in raw_upper: return {"sql_query": "NONE", "error": "SECURITY_ADMIN"}
    if "UNAUTHORIZED_GUEST" in raw_upper: return {"sql_query": "NONE", "error": "SECURITY_GUEST"}

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

# --- REFINED SECURITY_CHECKER ---
def security_checker(state: AgentState) -> dict:
    if state.get("error") and "SECURITY" in state["error"]:
        return {"error": state["error"], "sql_query": "NONE"}

    sql = state.get("sql_query", "").lower()
    if sql == "none" or not sql:
        return {}

    # 1. Block destructive commands
    forbidden = ["drop", "delete", "update", "insert", "alter", "truncate", "grant"]
    if any(word in sql for word in forbidden):
        return {"error": "SECURITY_ADMIN", "sql_query": "NONE"}

    role = state["user_role"]
    uid = str(state["user_id"])

    # 2. GUEST Restrictions
    if role == "GUEST":
        guest_blocked_tables = ["orders", "users", "shipments", "order_items", "payments", "stores"]
        if any(t in sql for t in guest_blocked_tables):
            return {"error": "SECURITY_GUEST", "sql_query": "NONE"}

    # 3. INDIVIDUAL Restrictions (Access to public data + own personal data)
    if role == "INDIVIDUAL":
        # Block access to other users or store management tables
        if any(t in sql for t in ["users", "stores"]):
            return {"error": "SECURITY_ADMIN", "sql_query": "NONE"}
        
        # Mandatory IDOR check for personal tables
        personal_tables = ["orders", "order_items", "shipments", "reviews", "payments"]
        if any(t in sql for t in personal_tables):
            if not re.search(r'\b' + uid + r'\b', sql):
                return {"error": "SECURITY_USER", "sql_query": "NONE"}

    # 4. CORPORATE Restrictions (Access to own store and products)
    if role == "CORPORATE":
        if "users" in sql:
             return {"error": "SECURITY_ADMIN", "sql_query": "NONE"}
        
        # Corporate users can query products (their own) and transactions (their store's)
        # We ensure their owner/seller ID is linked in the SQL
        transaction_tables = ["orders", "order_items", "shipments", "products"]
        if any(t in sql for t in transaction_tables):
            if not re.search(r'\b' + uid + r'\b', sql):
                return {"error": "SECURITY_STORE", "sql_query": "NONE"}

    return {"error": None}

def db_executor(state: AgentState):
    if state["sql_query"] == "NONE" or state.get("error"):
        return {"db_results": {}}
    print(f"[DB_EXECUTOR] Running: {state['sql_query']}")
    result = run_query(state["sql_query"])
    if "error" in result:
        return {"error": result["error"], "db_results": None}
    return {"db_results": result}

def summarizer(state: AgentState):
    error = state.get("error") or ""
    lang = detect_language(state["question"])

    # --- ROLE-BASED ERROR MESSAGES ---
    if "SECURITY_USER" in error:
        return {"final_answer": "Başka bir kullanıcının verisini görüntüleyemezsiniz." if lang == "tr" else "You cannot view another user's data."}
    if "SECURITY_STORE" in error:
        return {"final_answer": "Mağazanın sahibi siz olmadığınız için buna izniniz yok." if lang == "tr" else "You do not have permission for this store."}
    if "SECURITY_ADMIN" in error:
        return {"final_answer": "Yönetici değilsiniz, bu komutu sadece yöneticiler kullanabilir." if lang == "tr" else "This command is restricted to administrators."}
    if "SECURITY_GUEST" in error:
        return {"final_answer": "Bu işlemi yapmak için giriş yapmalısınız." if lang == "tr" else "Please log in to access this information."}
    
    if "SQL generation failed" in error or state.get("sql_query") == "NONE":
        return {"final_answer": "İsteğinizi gerçekleştiremedim." if lang == "tr" else "I couldn't fulfill your request."}

    db_results = state.get("db_results")
    if not db_results or db_results == {}:
        return {"final_answer": "Kayıt bulunamadı." if lang == "tr" else "No records found."}

    prompt = f"""Question: {state["question"]}\nResult: {db_results}\nSummarize clearly. { "Yanıtı Türkçe ver." if lang == "tr" else "Respond in English." }
If the data is aggregate (lists/counts), add a tag at the end: [CHART: pie] or [CHART: bar] or [CHART: donut]."""
    return {"final_answer": llm_invoke(prompt)}

# ---------------------------------------------------------------------------
# GRAPH & API
# ---------------------------------------------------------------------------

workflow = StateGraph(AgentState)
workflow.add_node("intent", intent_classifier)
workflow.add_node("greet", greeting_handler)
workflow.add_node("sql", sql_writer)
workflow.add_node("security", security_checker)
workflow.add_node("db", db_executor)
workflow.add_node("sum", summarizer)
workflow.set_entry_point("intent")
workflow.add_conditional_edges("intent", lambda s: "greet" if s["intent"] == "greeting" else "sql")
workflow.add_edge("greet", END)
workflow.add_edge("sql", "security")
workflow.add_conditional_edges("security", lambda s: "sum" if s.get("error") else "db")
workflow.add_conditional_edges("db", lambda s: "sql" if s.get("error") else "sum")
workflow.add_edge("sum", END)
ai_brain = workflow.compile()

class ChatRequest(BaseModel):
    message: str
    user_role: str
    user_id: int

@app.post("/agent/ask")
async def ask_agent(request: ChatRequest):
    try:
        uid = request.user_id
        if uid not in user_memory_bank: user_memory_bank[uid] = []
        history_str = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in user_memory_bank[uid])

        state: AgentState = {
            "question": request.message, "user_role": request.user_role, "user_id": uid,
            "history_text": history_str, "intent": "db_query", "sql_query": "",
            "db_results": None, "error": None, "final_answer": "",
        }

        result = ai_brain.invoke(state, {"recursion_limit": 15})
        answer = result.get("final_answer", "An error occurred.")

        if result.get("intent") != "greeting":
            clean_ans = re.sub(r"\[CHART:\s*.*?\]", "", answer).strip()
            user_memory_bank[uid].append({"role": "user", "content": request.message})
            user_memory_bank[uid].append({"role": "ai", "content": clean_ans})
            if len(user_memory_bank[uid]) > 6: user_memory_bank[uid] = user_memory_bank[uid][-6:]

        raw_db = result.get("db_results")
        has_chart = False
        if isinstance(raw_db, dict) and "columns" in raw_db and "data" in raw_db:
             if len(raw_db["columns"]) >= 2 and len(raw_db["data"]) > 0:
                 # Check if at least one column is numeric to avoid empty charts
                 # We check the first row of data
                 try:
                     has_chart = any(isinstance(v, (int, float)) for v in raw_db["data"][0])
                 except Exception:
                     has_chart = False

        return {"reply": answer, "hasChart": has_chart, "chartData": raw_db if has_chart else None}
    except Exception as e:
        import traceback
        print("\n❌ [CRASH] in ask_agent:")
        traceback.print_exc()
        raise HTTPException(500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)