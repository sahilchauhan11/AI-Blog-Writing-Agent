from __future__ import annotations

import operator
from pathlib import Path
from typing import TypedDict, List, Annotated, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq


# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()


# -----------------------------
# Models
# -----------------------------

class Task(BaseModel):
    id: int
    title: str

    goal: str = Field(
        ...,
        description="One sentence describing what the reader should be able to do/understand after this section.",
    )

    bullets: List[str] = Field(
        ...,
        min_length=3,
        max_length=5,
        description="3–5 concrete, non-overlapping subpoints to cover in this section.",
    )

    target_words: int = Field(
        ...,
        description="Target word count for this section (120–450).",
    )

    section_type: Literal[
        "intro", "core", "examples", "checklist", "common_mistakes", "conclusion"
    ] = Field(
        ...,
        description="Use 'common_mistakes' exactly once in the plan.",
    )


class Plan(BaseModel):
    blog_title: str
    audience: str = Field(..., description="Who this blog is for.")
    tone: str = Field(..., description="Writing tone (e.g., practical, crisp).")
    tasks: List[Task]


class State(TypedDict, total=False):
    topic: str
    plan: Plan
    sections: Annotated[List[str], operator.add]
    final: str


# -----------------------------
# LLM (Groq)
# -----------------------------

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.7,
)


# -----------------------------
# Orchestrator (Planner)
# -----------------------------

def orchestrator(state: State) -> dict:
    planner = llm.with_structured_output(Plan)

    plan = planner.invoke(
        [
            SystemMessage(
                content=(
                    "You are a senior technical writer and developer advocate. Your job is to produce a "
                    "highly actionable outline for a technical blog post.\n\n"
                    "Hard requirements:\n"
                    "- Create 5–7 sections (tasks).\n"
                    "- Each section must include:\n"
                    "  1) goal (1 sentence outcome)\n"
                    "  2) 3–5 concrete bullets\n"
                    "  3) target word count (120–450)\n"
                    "- Include EXACTLY ONE section with section_type='common_mistakes'.\n\n"
                    "Make it technical and implementation-focused.\n"
                    "Bullets must be actionable and testable.\n"
                    "Output must strictly match the Plan schema."
                )
            ),
            HumanMessage(content=f"Topic: {state['topic']}"),
        ]
    )

    return {"plan": plan}


# -----------------------------
# Fanout
# -----------------------------

def fanout(state: State):
    return [
        Send(
            "worker",
            {"task": task, "topic": state["topic"], "plan": state["plan"]},
        )
        for task in state["plan"].tasks
    ]


# -----------------------------
# Worker (Section Writer)
# -----------------------------

def worker(payload: dict) -> dict:
    task = payload["task"]
    topic = payload["topic"]
    plan = payload["plan"]

    bullets_text = "\n- " + "\n- ".join(task.bullets)

    section_md = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a senior technical writer. Write ONE section in Markdown.\n\n"
                    "- Follow Goal and ALL Bullets.\n"
                    "- Stay near Target words (±15%).\n"
                    "- Output ONLY section content.\n"
                    "- Start with '## <Section Title>'.\n"
                    "- Include code snippets or examples if relevant.\n"
                    "- Mention trade-offs and edge cases briefly.\n"
                )
            ),
            HumanMessage(
                content=(
                    f"Blog: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Topic: {topic}\n\n"
                    f"Section: {task.title}\n"
                    f"Section type: {task.section_type}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"Bullets:{bullets_text}\n"
                )
            ),
        ]
    ).content.strip()

    return {"sections": [section_md]}


# -----------------------------
# Reducer
# -----------------------------

def reducer(state: State) -> dict:
    title = state["plan"].blog_title
    body = "\n\n".join(state["sections"]).strip()

    final_md = f"# {title}\n\n{body}\n"

    filename = "".join(c if c.isalnum() or c in (" ", "_", "-") else "" for c in title)
    filename = filename.strip().lower().replace(" ", "_") + ".md"

    Path(filename).write_text(final_md, encoding="utf-8")

    return {"final": final_md}


# -----------------------------
# Graph
# -----------------------------

g = StateGraph(State)

g.add_node("orchestrator", orchestrator)
g.add_node("worker", worker)
g.add_node("reducer", reducer)

g.add_edge(START, "orchestrator")
g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()


# -----------------------------
# Run
# -----------------------------

def run():
    out = app.invoke(
        {"topic": "Write a blog on Self Attention", "sections": []}
    )
    print(out["final"])


if __name__ == "__main__":
    run()