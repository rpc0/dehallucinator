import gradio as gr
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.prompts.prompts import SimpleInputPrompt
from llama_index.llms.ollama import Ollama
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.legacy.embeddings.langchain import LangchainEmbedding
from llama_index.core.settings import Settings

data_folder_path = "./data"
loaded_model = None
loaded_data_folder = None
query_engine = None


# Function to load documents and set up the model
def setup_rag(data_folder, model_name):
    global loaded_model, loaded_data_folder, query_engine

    print(loaded_model, model_name, loaded_data_folder, data_folder)

    if loaded_model == model_name and loaded_data_folder == data_folder:
        print("came inside the if clause")
        return query_engine

    documents = SimpleDirectoryReader(data_folder).load_data()
    print(documents)

    system_prompt = """
    You are a Q&A assistant. Your goal is to answer questions as accurately and concisely 
    as possible based on the instructions and context provided. When you provide an answer, 
    cite your source document names.
    """

    query_wrapper_prompt = SimpleInputPrompt("{query_str}")

    llm = Ollama(
        model=model_name, 
        base_url="http://localhost:11434", 
        context_window=4096,
        system_prompt=system_prompt, 
        query_wrapper_prompt=query_wrapper_prompt, 
        temperature=0
    )

    embed_model = LangchainEmbedding(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    )

    Settings.llm = llm
    Settings.embed_model = embed_model

    index = VectorStoreIndex.from_documents(documents, chunk_size=1024)
    query_engine = index.as_query_engine()

    loaded_model = model_name
    loaded_data_folder = data_folder

    return query_engine

def ask_question(query_engine, question):
    response = query_engine.query(question)
    return response.response


# List of available models
models = ["mistral", "gemini", "llama3.1"]  # Add your available models here

# Gradio interface
def gradio_queries(data_folder, question, model_name):
    qe = setup_rag(data_folder, model_name)
    return ask_question(qe, question)

data_folder = gr.Textbox(label="Documents folder")
question_input = gr.Textbox(label="Question")
model_dropdown = gr.Dropdown(choices=models, label="Choose Model")
response_output = gr.Textbox(label="Response")

interface = gr.Interface(
    fn = gradio_queries,
    inputs = [data_folder, question_input, model_dropdown],
    outputs = "text",
    title = "Text Search In Folder Using LLM",
    description = "Choose document folder and model of choice, type query, and ask"
)

interface.launch()
