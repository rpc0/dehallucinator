# Get the absolute path to the `deh/src` directory
from langchain.prompts import PromptTemplate
import prompts


query_prompts = [
    PromptTemplate(
        template=prompts.rag_text_prompts[2],
        input_variables=["question"]
    ),
    # PromptTemplate(
    #     template=prompts.rag_text_prompts[1],
    #     input_variables = ["context", "question"]
    # ),

    PromptTemplate(
        template="""
    You are an assistant for question-answering tasks.
    Please only use the following pieces of retrieved context to answer the question.
    Use ten words maximum and keep the answer concise.

    Question: {question}
    Context: {context}

    Answer:
    """,
        input_variables=["context", "question"]
    ),

    PromptTemplate(
        template=("""
                You are an assistant for question-answering tasks.
                Use the following pieces of retrieved context to answer the question.
                If you don't know the answer, just return 'DONT KNOW'. 
                If you know the answer, keep it as short and concise as possible,
                i.e. to a maximum of a couple of words.

                Question: {question}
                Context: {context}

                Answer:
                """
        ),
        input_variables=["context", "question"]
    ),
    PromptTemplate(
        template=prompts.hyde_prompts[1],
        input_variables=["question"]
    )
]

