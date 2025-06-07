from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START

from tools import *
from state import *

builder = StateGraph(State)
builder.add_node("solver", code_solver)
builder.add_node("evaluate", evaluate)
builder.add_node("retrieve_examples", retrieve_examples)

builder.add_edge(START, "solver")

def control_from_solver(state: State):
    if 'examples' in state and state['examples'] is not None:
        return "evaluate"
    else:
        return "retrieve_examples"

def control_end(state: State):
    if state.get("status") == "success":
        return END
    return "solver"

builder.add_conditional_edge("solver", control_from_solver, {"evaluate": "evaluate", "retrieve_examples": "retrieve_examples"})
builder.add_conditional_edge("evaluate", control_end, {END: END, "solver": "solver"})

checkpoint = MemorySaver()
graph = builder.compile(checkpoint=checkpoint)