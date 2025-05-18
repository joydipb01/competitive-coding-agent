from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph.message import AnyMessage, add_messages
from langgraph.graph.state import MessagesState


class TestCase(TypedDict):
    inputs: str
    outputs: str


class State(MessagesState):
    candidate: AIMessage
    examples: str
    test_cases: list[TestCase]
    runtime_limit: int
    status: str