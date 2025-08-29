
#pip install fastapi uvicorn langchain_groq langgraph python-dotenv
#uvicorn main:app --reload


import os
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from pydantic import BaseModel

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict


# --------------------------
# Load Environment Variables
# --------------------------
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


# --------------------------
# LangGraph State Definition
# --------------------------
class State(TypedDict):
    topic: str
    code: str
    improved_code: str
    final_code: str


# Initialize LLM
model_name = "gemma2-9b-it"
llm = ChatGroq(model=f"{model_name}")


# --------------------------
# Helper Function
# --------------------------
def clean_python_code(raw_code: str) -> str:
    """Remove markdown formatting and keep only code"""
    return raw_code.strip("`").replace("python", "").strip()


# --------------------------
# Workflow Nodes
# --------------------------
def generate_code(state: State):
    msg = llm.invoke(f"Generate a python code with proper indentation about {state['topic']}")
    return {"code": msg.content}


def peer_review(state: State):
    review_prompt = f"""
    You are a code reviewer. The user wrote the following Python code:

    {state['code']}

    Please do the following:
    1. Verify if the code is syntactically correct and runs without errors.
    2. Test it on edge cases and typical test cases.
    3. Check if it handles invalid inputs gracefully (if applicable).
    4. Finally, return only one word: "Pass" if the code is correct and robust, otherwise "Fail".
    """
    msg = llm.invoke(review_prompt)
    review_result = msg.content.strip().lower()
    return "Pass" if "pass" in review_result else "Fail"


def improve_code(state: State):
    prompt = f"""
    Here is the current Python code:

    {state['code']}

    Improve this code by:
    1. Checking if there are more efficient Data Structures and Algorithms that can be applied.
    2. Improving time and space complexity if possible.
    3. Keeping readability and maintainability in mind.
    4. Returning only the improved Python code with proper indentation and docstrings.
    """
    msg = llm.invoke(prompt)
    return {"improved_code": msg.content}


def manager_review(state: State):
    prompt = f"""
    You are acting as a senior engineering manager reviewing this Python code:

    {state['improved_code']}

    Please do the following:
    1. Verify correctness and robustness of the code.
    2. Ensure it passes edge cases and potential failure scenarios.
    3. Check whether it uses efficient Data Structures / Algorithms where relevant.
    4. Ensure code readability and Pythonic style.
    """
    msg = llm.invoke(prompt)
    final_code = msg.content.strip()
    return {"final_code": final_code}


# --------------------------
# Build Workflow
# --------------------------
workflow = StateGraph(State)
workflow.add_node("generate_code", generate_code)
workflow.add_node("improve_code", improve_code)
workflow.add_node("manager_review", manager_review)

workflow.add_edge(START, "generate_code")
workflow.add_conditional_edges("generate_code", peer_review, {"Fail": "improve_code", "Pass": END})
workflow.add_edge("improve_code", "manager_review")
workflow.add_edge("manager_review", END)

chain = workflow.compile()


# --------------------------
# FastAPI App
# --------------------------
app = FastAPI(title="Python Code Generator & Reviewer API",
              description="Generate, improve, and review Python code using Groq LLM + LangGraph",
              version="1.0")


class CodeResponse(BaseModel):
    topic: str
    raw_code: str
    clean_code: str
    improved_code: str | None = None
    final_code: str | None = None


@app.get("/generate", response_model=CodeResponse)
def generate_code_api(topic: str = Query(..., description="Topic for Python code generation")):
    """Generate Python code for a given topic"""
    state = chain.invoke({"topic": topic})

    raw_code = state.get("code", "")
    clean_code = clean_python_code(raw_code)
    improved_code = state.get("improved_code", None)
    final_code = state.get("final_code", None)

    return CodeResponse(
        topic=topic,
        raw_code=raw_code,
        clean_code=clean_code,
        improved_code=improved_code,
        final_code=final_code
    )
