from langchain_core.prompts import PromptTemplate
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, GenerationConfig, pipeline, AutoTokenizer
import torch
from langchain_community import BM25Retriever

class LLMManager:

    MODEL_NAME = "anonymous4chan/llama-2-7b"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        use_fast=True
    )
    generation_config = GenerationConfig.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )
    generation_config.temperature = 0.0001
    generation_config.top_p = 0.95
    generation_config.top_k = 50
    generation_config.max_new_tokens = 1024

    code_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        generation_config=generation_config,
    )

    def __init__(self, train_df):
        self.llm = HuggingFacePipeline(
            pipeline=self.code_pipeline
        )

        self.retriever = BM25Retriever.from_texts(
            [f"""<problem>
            {row["description"]}
            </problem>
            <solution>
            {row["solution"]}
            </solution>""" for _, row in train_df.iterrows()]
        )
