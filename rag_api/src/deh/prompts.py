# Pydantic classes and prompts for LLM interaction

from pydantic import BaseModel, Field
from enum import Enum

# Based on https://smith.langchain.com/hub/rlm/rag-prompt-llama
rag_prompt_llama_text = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

Question: {question}
Context: {context}

Answer:
"""

qa_eval_prompt_with_context_text = """
You are a teacher evaluating a test.
You are provided with a question along with an answer for the question written by a student.
Evaluate the question-answer pair using the provided context and provide feedback.
Only mark the answer as correct if it agress with the provided context.

{format_instructions}
Context : {context}
Question : {question}
Answer : {answer}
"""


class gradeEnum(str, Enum):
    correct = "correct"
    incorrect = "incorrect"


class LLMEvalResult(BaseModel):
    grade: gradeEnum = Field(
        description="Final grade label. Accepted labels : Correct, Incorrect"
    )
    description: str = Field(
        description="Explanation of why the specific grade was assigned. Must be concise. Not more than 2 sentences"
    )


hyde_prompt = """
Imagine you are an expert writing a detailed explanation on the topic: '{question}'
    Your response should be comprehensive and include all key points that would be found in the top search result.
"""
