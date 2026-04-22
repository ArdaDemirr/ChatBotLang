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
    {"name": "llama-3.3-70b", "model": "llama-3.3-70b-versatile", "provider": "groq"},
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
    lang = detect_language(state["question"])
    return {"final_answer": "Merhaba! Ne öğrenmek istersiniz?" if lang == "tr" else "Hello! What would you like to know?"}

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
            f"If they ask for another store's private data, YOU MUST RETURN: UNAUTHORIZED_STORE\n"
            f"If they ask for global users or admin data, YOU MUST RETURN: UNAUTHORIZED_ADMIN\n"
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

    # Safe regex fix
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
    # If the SQL Writer already flagged an unauthorized prompt, preserve the error and skip checks
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

    # 3. INDIVIDUAL Restrictions (Strict IDOR Regex Prevention)
    if role == "INDIVIDUAL":
        # Block system tables completely
        if any(t in sql for t in ["users", "stores"]):
            return {"error": "SECURITY_ADMIN", "sql_query": "NONE"}
        
        personal_tables = ["orders", "order_items", "shipments", "reviews", "payments"]
        if any(t in sql for t in personal_tables):
            # We use regex word boundaries (\b) so an ID of '5' doesn't accidentally pass just because 'LIMIT 50' is in the query.
            if not re.search(r'\b' + uid + r'\b', sql):
                return {"error": "SECURITY_USER", "sql_query": "NONE"}

    # 4. CORPORATE Restrictions
    if role == "CORPORATE":
        # Store owners shouldn't see all users
        if "users" in sql:
             return {"error": "SECURITY_ADMIN", "sql_query": "NONE"}
        
        transaction_tables = ["orders", "order_items", "shipments"]
        if any(t in sql for t in transaction_tables):
            if not re.search(r'\b' + uid + r'\b', sql):
                return {"error": "SECURITY_STORE", "sql_query": "NONE"}

    # Admin passes through with no restrictions
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

    # --- CUSTOM UI ERROR MESSAGES FOR THE PRESENTATION DEMO ---
    if "SECURITY_USER" in error:
        return {"final_answer": "Başka bir kullanıcının verisini görüntüleyemezsiniz." if lang == "tr" else "You cannot view another user's data."}
    if "SECURITY_STORE" in error:
        return {"final_answer": "Mağazanın sahibi siz olmadığınız için buna izniniz yok." if lang == "tr" else "You do not have permission as you do not own this store."}
    if "SECURITY_ADMIN" in error:
        return {"final_answer": "Yönetici değilsiniz, bu komutu sadece yöneticiler kullanabilir." if lang == "tr" else "You are not an admin. Only administrators can use this command."}
    if "SECURITY_GUEST" in error:
        return {"final_answer": "Bu işlemi yapmak için giriş yapmalısınız." if lang == "tr" else "You must log in to view this."}
    if "SECURITY" in error:
        return {"final_answer": "Üzgünüm, bu bilgilere erişme yetkiniz bulunmamaktadır." if lang == "tr" else "Sorry, you don't have permission to access this data."}

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

def route_after_db(state: AgentState) -> str:
    """If the DB throws an error, route back to the SQL writer to fix it."""
    if state.get("error"):
        print("\n🔄 [ROUTER] Database error detected, routing back to SQL Writer for correction...\n")
        return "sql"
    return "sum"

workflow.add_conditional_edges("db", route_after_db)
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

        # Recursion limit prevents infinite loops if it constantly fails
        result = ai_brain.invoke(state, {"recursion_limit": 15})
        answer = result.get("final_answer", "An error occurred.")

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