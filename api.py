import os
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from database_utils import get_schema, run_query

load_dotenv()
app = FastAPI()

# --- 1. THE MEMORY BANK (Fix for shared history) ---
# This dictionary will hold separate chat histories for every unique user_id
# Format: { 1: [{"role": "user", "content": "hi"}, ...], 2: [...] }
user_memory_bank = {}

# --- 2. STATE & LLM ---
class AgentState(TypedDict):
    question: str
    user_role: str
    user_id: int
    history_text: str  # <--- We now pass the user's specific history here
    sql_query: str
    db_results: dict
    error: str
    final_answer: str


# for google gemini use this command
# GOOGLE GEMINI KULLANMAK İÇİN BU KOMUTUN YORUM SATIRINI KALDIRIP BUNU KULLANIN
#llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

# Initializing your local Phi-3 model! Temperature 0.0 keeps it strictly logical.
llm = ChatOllama(model="phi3", temperature=0.0)


# --- NODE 1: SQL Writer (Role-Aware & Regex Enhanced) ---
def sql_writer(state: AgentState):
    schema = get_schema()
    role = state['user_role']
    user_id = state['user_id']
    
    role_rules = ""
    if role == "INDIVIDUAL":
        role_rules = f"CRITICAL: User is an INDIVIDUAL (ID: {user_id}). They can ONLY query products. NO access to users or stores."
    elif role == "CORPORATE":
        role_rules = f"CRITICAL: User is a CORPORATE seller (ID: {user_id}). Filter queries by store_id."

    prompt = f"""
    You are a Data Analyst AI.
    
    Previous Conversation Context:
    {state['history_text']}
    
    Write a MySQL query for this new question: '{state['question']}'
    
    {role_rules}
    Schema: {schema}
    
    Return ONLY a valid SELECT statement ending with a semicolon (;). Do not explain anything.
    """
        
    response = llm.invoke(prompt)
    raw_text = response.content
    
    # --- REGEX SNIPER: The fix for Phi-3 being too chatty ---
    # This hunts specifically for "SELECT ... ;" and ignores all conversational junk
    sql_match = re.search(r"(SELECT.*?;)", raw_text, re.IGNORECASE | re.DOTALL)
    
    if sql_match:
        clean_sql = sql_match.group(1).strip()
    else:
        # Fallback if it forgot the semicolon
        clean_sql = raw_text.replace("```sql", "").replace("```", "").strip()
        
    return {"sql_query": clean_sql, "error": None}


# --- NODE 2: Security Checker ---
def security_checker(state: AgentState):
    sql = state['sql_query'].lower()
    
    forbidden_words = ["drop", "delete", "update", "insert", "alter", "truncate"]
    if any(word in sql for word in forbidden_words):
        return {"error": "SECURITY VIOLATION: Database modification blocked.", "sql_query": ""}
    
    if state['user_role'] == "INDIVIDUAL" and ("users" in sql or "stores" in sql):
        return {"error": "SECURITY VIOLATION: Permission denied for system tables.", "sql_query": ""}

    return {"error": None}


# --- NODE 3: Database Executor ---
def db_executor(state: AgentState):
    if state.get("error"):
        return {"db_results": None}
        
    result = run_query(state['sql_query'])
    if "error" in result:
        return {"error": result['error'], "db_results": None}
    return {"db_results": result, "error": None}


# --- NODE 4: Summarizer ---
def summarizer(state: AgentState):
    if state.get("error") and "SECURITY VIOLATION" in state["error"]:
        return {"final_answer": "I'm sorry, you do not have permission to view this data."}
        
    prompt = f"""
    User Question: {state['question']}
    Data Results: {state['db_results']}
    
    Write a short, natural response answering the user's question based strictly on the data provided. Do not show the SQL query.
    """
    response = llm.invoke(prompt)
    return {"final_answer": response.content}


# --- EDGE ROUTING ---
def route_after_security(state: AgentState):
    if state.get("error"):
        return "summarizer" 
    return "db_executor"

def route_after_execution(state: AgentState):
    if state.get("error"):
        return "sql_writer" 
    return "summarizer"


# --- BUILD GRAPH ---
workflow = StateGraph(AgentState)
workflow.add_node("sql_writer", sql_writer)
workflow.add_node("security_checker", security_checker)
workflow.add_node("db_executor", db_executor)
workflow.add_node("summarizer", summarizer)

workflow.set_entry_point("sql_writer")
workflow.add_edge("sql_writer", "security_checker")
workflow.add_conditional_edges("security_checker", route_after_security)
workflow.add_conditional_edges("db_executor", route_after_execution)
workflow.add_edge("summarizer", END)

ai_brain = workflow.compile()


# --- FASTAPI ENDPOINT & MEMORY MANAGEMENT ---
class ChatRequest(BaseModel):
    message: str
    user_role: str
    user_id: int

@app.post("/agent/ask")
async def ask_agent(request: ChatRequest):
    try:
        user_id = request.user_id
        
        # 1. Retrieve THIS specific user's history
        if user_id not in user_memory_bank:
            user_memory_bank[user_id] = []
            
        user_history = user_memory_bank[user_id]
        
        # Format it into a readable string for Phi-3
        history_str = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in user_history])
        
        # 2. Run the LangGraph
        inputs = {
            "question": request.message, 
            "user_role": request.user_role,
            "user_id": user_id,
            "history_text": history_str
        }
        
        for output in ai_brain.stream(inputs):
            pass 
        
        final_answer = output.get('summarizer', {}).get('final_answer', "Sorry, an error occurred.")
        
        # 3. Save the new interaction back to THIS user's memory
        user_memory_bank[user_id].append({"role": "user", "content": request.message})
        user_memory_bank[user_id].append({"role": "ai", "content": final_answer})
        
        # Optional: Keep memory short (last 6 messages) so Phi-3 doesn't get confused
        if len(user_memory_bank[user_id]) > 6:
            user_memory_bank[user_id] = user_memory_bank[user_id][-6:]
            
        return {"reply": final_answer}
        
    except Exception as e:
        print(f"❌ ERROR CRASH: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)