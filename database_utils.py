import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text

# Load secrets from your .env file
load_dotenv()

# Build the connection string for MySQL
db_user = os.getenv("DB_USER", "root")
db_password = os.getenv("DB_PASSWORD", "password") # Uses 'password' if .env is missing
db_host = os.getenv("DB_HOST", "localhost")
db_name = os.getenv("DB_NAME", "projectspring")

db_url = f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}"
engine = create_engine(db_url)

def get_schema():
    """Reads your MySQL database and returns a text summary of all tables and columns."""
    try:
        inspector = inspect(engine)
        schema_info = "Database Schema:\n"
        for table_name in inspector.get_table_names():
            schema_info += f"\nTable: {table_name}\nColumns:\n"
            for column in inspector.get_columns(table_name):
                schema_info += f"  - {column['name']} ({column['type']})\n"
        return schema_info
    except Exception as e:
        return f"Error connecting to DB: {e}"

def run_query(query: str):
    """Executes a SQL query safely and returns the rows or an error."""
    try:
        with engine.connect() as connection:
            result = connection.execute(text(query))
            
            # If it's an UPDATE/DELETE/INSERT, it won't have rows to fetch
            if not result.returns_rows:
                return {"message": "Query executed successfully. No rows returned."}
                
            rows = result.fetchall()
            if not rows:
                return {"message": "Query ran successfully, but returned 0 rows."}
                
            return {"columns": list(result.keys()), "data": [list(row) for row in rows]}
    except Exception as e:
        # We capture the error so the AI can read it and fix its own code!
        return {"error": str(e)}