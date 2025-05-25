from pydantic import BaseModel, Field
from state import GraphState
from llm_manager import LLMManager
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from test_eval_utils import check_correctness

llm_manager = LLMManager()

class writeCpp(BaseModel):

    reasoning: str = Field(..., description="The reasoning behind the code generation.")
    pseudo_code: str = Field(..., description="The pseudo code generated for the task in English.")
    code: str = Field(..., description="Valid C++14 code to solve the task.")

def code_solver(state: GraphState):
    inputs = {"messages": state['messages']}
    has_examples = 'examples' in state and state['examples'] is not None
    output_key = "messages" if has_examples else "candidate"

    if has_examples:
        inputs['examples'] = state['examples']

    prompt = PromptTemplate.from_file('../prompts/prompt_code_solver.txt')
    llm = llm_manager.llm

    runnable = prompt | llm.bind_tools([writeCpp])
    output = runnable.invoke({"messages": inputs})

    if not output.content:
        return {"messages": [AIMessage(content="Hmmm, I will need to think about this step by step.")]}

    return {output_key: output}

def format_tool_message(response: str, ai_message: AIMessage):
    return ToolMessage(
        content=response + "\nMake all fixes using the writeCpp tool.",
        tool_call_id=ai_message.tool_calls[0].id,
    )

def evaluate(state: GraphState):
    test_cases = state['test_cases']
    ai_message = state['messages'][-1]
    if not ai_message.tool_calls:
        return {
            "messages": [
                HumanMessage(
                    content="No code submitted. Please try again using the correct C++14 code."
                )
            ]
        }
    
    try:
        code = ai_message.tool_calls[0]['args']['code']
    except Exception as e:
        return {
            "messages": [
                format_tool_message(repr(e), ai_message)
            ]
        }
    
    num_test_cases = len(test_cases)
    success_count = 0
    test_results = []

    for test_case in test_cases:
        input_data = test_case['input']
        expected_output = test_case['output']
        test_result = check_correctness(code, input_data, expected_output)
        test_results.append(test_result)
        if test_result:
            success_count += 1
    success_rate = success_count / num_test_cases if num_test_cases else "N/A"
    if success_rate == 1.0:
        return {"status": "success"}
    
    responses = "\n".join([f"<test id={i}>\n{r}\n</test>" for i, r in enumerate(test_results)])
    response = f"Incorrect submission. Please respond with updated code.\nPass rate: {success_rate}\nResults:\n{responses}"
    formatted_message = format_tool_message(response, ai_message)
    return {"messages": [formatted_message]}