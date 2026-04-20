import os
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from database_utils import get_schema, run_query

load_dotenv()

# 1. Define the "Clipboard" that travels between agents
class AgentState(TypedDict):
    question: str
    sql_query: str
    db_results: dict
    error: str
    final_answer: str

# 2. Initialize Gemini 1.5 Flash (Fast and great at code)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

# --- AGENT 1: The SQL Writer ---
def sql_writer(state: AgentState):
    print("🤖 Agent: Writing SQL...")
    schema = get_schema()
    
    # If we are coming from an error, tell the AI to fix it!
    if state.get("error"):
        prompt = f"""
        You wrote a MySQL query that failed. 
        Question: {state['question']}
        Bad Query: {state['sql_query']}
        Error: {state['error']}
        Schema: {schema}
        
        Fix the query. Return ONLY the raw SQL code. No markdown formatting (no ```sql).
        """
    else:
        prompt = f"""
        You are an expert MySQL database analyst. 
        Schema: {schema}
        Question: "{state['question']}"
        
        Write a MySQL query to answer the question. 
        Return ONLY the raw SQL code. No markdown formatting (no ```sql).
        """
        
    response = llm.invoke(prompt)
    clean_sql = response.content.replace("```sql", "").replace("```", "").strip()
    return {"sql_query": clean_sql, "error": None}

# --- AGENT 2: The Database Executor ---
def db_executor(state: AgentState):
    print(f"⚙️ Agent: Executing SQL -> {state['sql_query']}")
    result = run_query(state['sql_query'])
    
    if "error" in result:
        print(f"⚠️ Error caught: {result['error']}")
        return {"error": result['error'], "db_results": None}
    
    print("✅ SQL Execution Successful!")
    return {"db_results": result, "error": None}

# --- AGENT 3: The Summarizer ---
def summarizer(state: AgentState):
    print("📝 Agent: Summarizing results for the user...")
    prompt = f"""
    User Question: {state['question']}
    Data from Database: {state['db_results']}
    
    Write a helpful, natural language response answering the user's question based on the data.
    """
    response = llm.invoke(prompt)
    return {"final_answer": response.content}

# --- EDGE ROUTING LOGIC ---
def route_after_execution(state: AgentState):
    """Decides where to go after the database runs the query."""
    if state.get("error"):
        return "sql_writer" # Send it back to be fixed
    return "summarizer"     # Send it to the final step

# 3. Build the Graph Workflow
workflow = StateGraph(AgentState)

workflow.add_node("sql_writer", sql_writer)
workflow.add_node("db_executor", db_executor)
workflow.add_node("summarizer", summarizer)

workflow.set_entry_point("sql_writer")
workflow.add_edge("sql_writer", "db_executor")

# This is the self-correction loop!
workflow.add_conditional_edges("db_executor", route_after_execution)

workflow.add_edge("summarizer", END)

# Compile the brain
app = workflow.compile()

# --- TEST IT IN THE TERMINAL ---
if __name__ == "__main__":
    print("--- E-Commerce AI Agent Started ---")
    test_question = "How many products do we have in the database?"
    print(f"User: {test_question}\n")
    
    inputs = {"question": test_question}
    
    # Run the graph and stream the outputs
    for output in app.stream(inputs):
        pass # The agents have print statements inside them to show progress
        
    final_state = output.get('summarizer', {})
    print(f"\n🤖 AI Final Answer: {final_state.get('final_answer')}")