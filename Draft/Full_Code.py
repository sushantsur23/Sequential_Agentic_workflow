# api_design_doc.py
import os, sys
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing_extensions import TypedDict

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

# Load API Key
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="gemma2-9b-it")

class State(TypedDict):
    idea: str
    draft_design: str
    improved_design: str
    final_doc: str

# Nodes
def generate_api_design(state: State):
    msg = llm.invoke(f"Generate a REST API design for: {state['idea']}. Include endpoints, request/response formats.")
    return {"draft_design": msg.content}

def peer_review_design(state: State):
    msg = llm.invoke(f"Check if this API design follows REST best practices:\n{state['draft_design']}\nReturn 'Pass' or 'Fail'.")
    return "Pass" if "pass" in msg.content.lower() else "Fail"

def improve_design(state: State):
    msg = llm.invoke(f"Improve this API design by adding authentication, error handling, pagination if needed:\n{state['draft_design']}")
    return {"improved_design": msg.content}

def manager_review(state: State):
    msg = llm.invoke(f"Convert this API design into a clean OpenAPI-style documentation with clear formatting:\n{state['improved_design']}")
    return {"final_doc": msg.content}

# Workflow
workflow = StateGraph(State)
workflow.add_node("generate_api_design", generate_api_design)
workflow.add_node("improve_design", improve_design)
workflow.add_node("manager_review", manager_review)

workflow.add_edge(START, "generate_api_design")
workflow.add_conditional_edges("generate_api_design", peer_review_design, {"Fail": "improve_design", "Pass": END})
workflow.add_edge("improve_design", "manager_review")
workflow.add_edge("manager_review", END)
chain = workflow.compile()

# FastAPI App
app = FastAPI(title="API Design & Documentation Generator")

class APIDesignResponse(BaseModel):
    idea: str
    draft_design: str
    improved_design: str | None = None
    final_doc: str | None = None

@app.post("/design", response_model=APIDesignResponse)
def design_api(idea: str = Query(..., description="Describe your API idea")):
    state = chain.invoke({"idea": idea})
    return APIDesignResponse(**state)


#pip install streamlit langchain_groq langgraph python-dotenv
