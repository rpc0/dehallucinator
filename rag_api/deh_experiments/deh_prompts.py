# Get the absolute path to the `deh/src` directory
from langchain.prompts import PromptTemplate
import prompts

NO_RAG_PROMPT_TEMPLATE = PromptTemplate(
    template=prompts.rag_text_prompts[2],
    input_variables=["question"]
)

BASIC_RAG_PROMPT_TEMPLATE = PromptTemplate(
    template="""
    You are an assistant for question-answering tasks.
    Please only use the following pieces of retrieved context 
    to answer the question. Use ten words maximum and keep
    the answer concise.

    Question: {question}
    Context: {context}

    Answer:
    """,
    input_variables=["context", "question"]
)

BASIC_RAG_DONT_LIE_PROMPT_TEMPLATE = PromptTemplate(
    template="""
        You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just return the text 'DONT KNOW'. Don't add
        any other text to that, just 'DONT KNOW' on its own line. 
        If you know the answer, keep it as short and concise as possible,
        i.e. to a maximum of a couple of words.

        Question: {question}
        Context: {context}

        Answer:
        """,
    input_variables=["context", "question"]
)

BASIC_RAG_HYDE_PROMPT_TEMPLATE = PromptTemplate(
    template=prompts.hyde_prompts[1],
    input_variables=["question"]
)

BASIC_RAG_SUPPRESS_ANSWERS_PROMPT_TEMPLATE_1 = PromptTemplate(
    template="""
        You are an assistant for validating answers to questions.
        You are given the following question and context:

        Question: {question}
        Context: {context}

        Furthermore, you are given a potential answer to the question:

        Answer: {answer}

        Please provide a score between 0 and 1, where 0 means that the answer is completely wrong
        and 1 means that the answer is completely correct. Please use the given context to
        determine the score.

    """,
    input_variables=["context", "question", "answer"]
)

llm_as_judge_prompt = """

You are an impartial and analytical judge tasked with evaluating the correctness of an answer based 
on a given question and context. You first need to assess if given only the context, an answer to the
question is possible at all. If you answer this with 'Yes', then your next task is to evaluate 
the correctness of the answer based on the given question and context. Your judgment should be logical,
well-reasoned, and clear.

Here is the process you should follow:

1.) Carefully read and understand the Question, Context, and Answer provided.

2.) Based on the Context, determine if an answer is possible at all, based on the Context. 
If you believe that an answser is possible, please determine if the provided Answer
addresses the Question accurately and answers it correctly.

3.) Please conclude with providing exactly two responses based on the descriptions provided 
under a.) and b.) below:

a.) Return 'YES' if you believe that an answer is possible and correct, 'NO' else. By all means, 
do not provide any additional text or a written explanation, not even within parantheses.
Please just answer with 'YES' or 'NO'.

b.) If you said 'YES' in the previous step:
    please provide a single score between 0.0 and 1.0, where 0.0 means that the Answer is completely wrong
    and 1.0 means that the Answer is completely correct. 
    
    If you said 'NO' in the previous step, please provide a score of 0. 

Please only provide a single number for your score and nothing else! By all means, do not 
provide any additional text or a written explanation, not even within parantheses. All I need is a
single, standalone score!

For the answers to 3a.) and 3b.), please provide your answers in the following format: (Yes/No, Score)

Following are some examples of the format of how you should answer:

Example 1:
----------

(Yes, 0.8)

Example 2:
----------

(No, 0.0)

Example 3:
----------

(Yes, 1.0)

Example 4:
----------

(Yes, 0.3)


----------------------------------------------------------------------
Question: {question}

Context: {context}

Answer: {answer}

Your Judgment: (Yes/No, Score)
----------------------------------------------------------------------


"""

# BASIC_RAG_SUPPRESS_ANSWERS_PROMPT_TEMPLATE_2 = PromptTemplate(
#     template=llm_as_judge_prompt_1,
#     input_variables=["context", "question", "answer"]
# )

BASIC_RAG_SUPPRESS_ANSWERS_PROMPT_TEMPLATE = PromptTemplate(
    template=llm_as_judge_prompt,
    input_variables=["context", "question", "answer"]
)


query_prompts = [
    NO_RAG_PROMPT_TEMPLATE,
    BASIC_RAG_PROMPT_TEMPLATE,
    BASIC_RAG_DONT_LIE_PROMPT_TEMPLATE,
    BASIC_RAG_HYDE_PROMPT_TEMPLATE,
    BASIC_RAG_SUPPRESS_ANSWERS_PROMPT_TEMPLATE
]
