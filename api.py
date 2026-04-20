import os
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from database_utils import get_schema, run_query

load_dotenv()
app = FastAPI()

# 1. State now includes user_role and user_id for security context
class AgentState(TypedDict):
    question: str
    user_role: str
    user_id: int
    sql_query: str
    db_results: dict
    error: str
    final_answer: str

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

# --- NODE 1: SQL Writer (Role-Aware) ---
def sql_writer(state: AgentState):
    schema = get_schema()
    role = state['user_role']
    user_id = state['user_id']
    
    # Give the AI strict rules based on the user's role
    role_rules = ""
    if role == "INDIVIDUAL":
        role_rules = f"CRITICAL: The user is an INDIVIDUAL customer (ID: {user_id}). They can ONLY query products. They CANNOT see other users' orders or system stats."
    elif role == "CORPORATE":
        role_rules = f"CRITICAL: The user is a CORPORATE seller (ID: {user_id}). They can ONLY query their own store's products and orders. Make sure to filter by store_id."
    elif role == "ADMIN":
        role_rules = "The user is an ADMIN. They have full read access to all metrics."

    prompt = f"""
    You are a Data Analyst AI. Write a MySQL query for this question: '{state['question']}'
    
    {role_rules}
    
    Schema: {schema}
    Return ONLY raw SQL code. Must be a SELECT statement.
    """
        
    response = llm.invoke(prompt)
    clean_sql = response.content.replace("```sql", "").replace("```", "").strip()
    return {"sql_query": clean_sql, "error": None}

# --- NODE 2: THE SECURITY GUARDRAIL ---
def security_checker(state: AgentState):
    sql = state['sql_query'].lower()
    
    # 1. Block malicious SQL commands
    forbidden_words = ["drop", "delete", "update", "insert", "alter", "truncate", "grant", "revoke"]
    if any(word in sql for word in forbidden_words):
        return {"error": "SECURITY VIOLATION: AI attempted to modify the database.", "sql_query": ""}
    
    # 2. Block Individuals from seeing sensitive tables
    if state['user_role'] == "INDIVIDUAL":
        if "users" in sql or "stores" in sql:
             return {"error": "SECURITY VIOLATION: Individual users cannot access system tables.", "sql_query": ""}

    return {"error": None} # Passed security checks!

# --- NODE 3: Database Executor ---
def db_executor(state: AgentState):
    # If the security checker flagged an error, DO NOT run the query!
    if state.get("error"):
        return {"db_results": None}
        
    result = run_query(state['sql_query'])
    if "error" in result:
        return {"error": result['error'], "db_results": None}
    return {"db_results": result, "error": None}

# --- NODE 4: Summarizer ---
def summarizer(state: AgentState):
    if state.get("error") and "SECURITY VIOLATION" in state["error"]:
        return {"final_answer": "I'm sorry, but you do not have permission to access that information."}
        
    prompt = f"User Question: {state['question']}\nData: {state['db_results']}\nWrite a helpful, professional response answering the user."
    response = llm.invoke(prompt)
    return {"final_answer": response.content}

# --- EDGE ROUTING ---
def route_after_security(state: AgentState):
    if state.get("error"):
        return "summarizer" # If security violation, go straight to the end and say "No permission"
    return "db_executor"

def route_after_execution(state: AgentState):
    if state.get("error"):
        return "sql_writer" # If it's just a syntax error, try writing it again
    return "summarizer"

# --- Build Graph ---
workflow = StateGraph(AgentState)
workflow.add_node("sql_writer", sql_writer)
workflow.add_node("security_checker", security_checker) # Add the new node!
workflow.add_node("db_executor", db_executor)
workflow.add_node("summarizer", summarizer)

workflow.set_entry_point("sql_writer")
workflow.add_edge("sql_writer", "security_checker")
workflow.add_conditional_edges("security_checker", route_after_security)
workflow.add_conditional_edges("db_executor", route_after_execution)
workflow.add_edge("summarizer", END)

ai_brain = workflow.compile()

# --- FASTAPI ENDPOINT ---
class ChatRequest(BaseModel):
    message: str
    user_role: str
    user_id: int

@app.post("/agent/ask")
async def ask_agent(request: ChatRequest):
    try:
        inputs = {
            "question": request.message, 
            "user_role": request.user_role,
            "user_id": request.user_id
        }
        for output in ai_brain.stream(inputs):
            pass 
        
        final_answer = output.get('summarizer', {}).get('final_answer', "Sorry, an error occurred.")
        return {"reply": final_answer}
    except Exception as e:
        print(f"❌ ERROR CRASH: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)