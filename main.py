import os


import dotenv
import fastapi


from back_end.server.api_model import PromptInput
from back_end.llm.rag_pipeline import Chain


dotenv.load_dotenv()


model_id = 'microsoft/Phi-3-mini-128k-instruct'
# model_id = 'mistralai/Mistral-7B-v0.1'
api_key = {
    'tavily': os.getenv('TAVILY_API_KEY'),
    'hf_k': os.getenv('HF_API_KEY'),
    'wv_k': os.getenv('WEAVIATE_API_KEY'),
    'wv_url': os.getenv('WEAVIATE_URL')
}


rag_chain = Chain(model_id=model_id, api_key=api_key)


app = fastapi.FastAPI()


@app.get('/')
def home():
    return "hello world"


@app.post('/prompt')
def prompt_llm(prompt: PromptInput):
    return rag_chain.prompt_chain(prompt.input)

# import os
# import dotenv
# import fastapi
# from back_end.server.api_model import PromptInput
# from back_end.llm.rag_pipeline import Chain

# # Load environment variables
# dotenv.load_dotenv()

# # Use a smaller and optimized model
# model_id = 'meta-llama/Llama-3.2-3B' 
# api_key = {
#     'tavily': os.getenv('TAVILY_API_KEY'),
#     'hf_k': os.getenv('HF_API_KEY'),
#     'wv_k': os.getenv('WEAVIATE_API_KEY'),
#     'wv_url': os.getenv('WEAVIATE_URL')
# }

# # Initialize the Chain with hybrid-optimized LLM
# rag_chain = Chain(model_id=model_id, api_key=api_key, device="cuda")  # Ensure the Chain uses quantized LLM

# # FastAPI setup
# app = fastapi.FastAPI()

# @app.get('/')
# def home():
#     return "hello world"

# @app.post('/prompt')
# def prompt_llm(prompt: PromptInput):
#     return rag_chain.prompt_chain(prompt.input)
