import os
from dotenv import load_dotenv
import streamlit as st
import sys

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from IPython.display import Image, display


# --------------------------
# Load Environment Variables
# --------------------------
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

st.title("My First Project")

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
def colorize_python_code(code: str) -> str:
    """Apply ANSI colors: code = green, comments/docstrings = cyan."""
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    RESET = "\033[0m"

    colored_lines = []
    in_docstring = False

    for line in code.splitlines():
        stripped = line.strip()
        if (stripped.startswith('"""') or stripped.startswith("'''")):
            if not in_docstring:
                colored_lines.append(f"{CYAN}{line}{RESET}")
                in_docstring = True
            else:
                colored_lines.append(f"{CYAN}{line}{RESET}")
                in_docstring = False
        elif in_docstring:
            colored_lines.append(f"{CYAN}{line}{RESET}")
        elif stripped.startswith("#"):
            colored_lines.append(f"{CYAN}{line}{RESET}")
        else:
            colored_lines.append(f"{GREEN}{line}{RESET}")

    return "\n".join(colored_lines)


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
    colored_code = colorize_python_code(final_code)
    return {"final_code": colored_code}


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
# Streamlit UI
# --------------------------
st.title("🧑‍💻 Python Code Generator & Reviewer")
st.write("This app generates Python code from a topic, improves it, and reviews it for quality.")

topic = st.text_input("Enter your topic:", placeholder="e.g., Take input string and replace recurring characters with blank")

if st.button("Generate Code"):
    if topic.strip():
        with st.spinner("Generating and reviewing code..."):
            state = chain.invoke({"topic": topic})
            raw_code = state["code"]
            clean_code = raw_code.strip("`").replace("python", "").strip()

        st.subheader("✅ Clean Extracted Python Code")
        st.code(clean_code, language="python")

    else:
        st.warning("⚠️ Please enter a topic before generating code.")
