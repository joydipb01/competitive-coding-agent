from pydantic import BaseModel, Field
from state import GraphState
from llm_manager import LLMManager
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

llm_manager = LLMManager()

class writeCpp(BaseModel):

    reasoning: str = Field(..., description="The reasoning behind the code generation.")
    pseudo_code: str = Field(..., description="The pseudo code generated for the task in English.")
    code: str = Field(..., description="Valid C++17 code to solve the task.")

def code_solver(state: GraphState):
    question = state['question']

    prompt = PromptTemplate.from_file('../prompts/prompt_code_solver.txt')
    llm = llm_manager.llm

    runnable = prompt | llm.bind_tools([writeCpp])
    code = runnable.invoke({"question": question})
    return {"code": code, "messages": [AIMessage(code)]}