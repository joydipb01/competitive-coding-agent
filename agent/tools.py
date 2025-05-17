from pydantic import BaseModel, Field
from state import GraphState
from llm_manager import LLMManager

class writeCpp(BaseModel):

    reasoning: str = Field(..., description="The reasoning behind the code generation.")
    pseudo_code: str = Field(..., description="The pseudo code generated for the task in English.")
    code: str = Field(..., description="Valid C++17 code to solve the task.")

def code_solver(state: GraphState)