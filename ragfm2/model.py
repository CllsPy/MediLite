from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import CTransformers
import chainlit as cl
from llama_cpp import Llama

DB_FAISS_PATH = "vectorstore/db_faiss"
MODEL_PATH = "/workspaces/lfm2MedicalBot/ragfm2/LFM2-1.2B-F16.gguf"

custom_prompt = """Leia o PDF anexado, avalie como médico e responda de forma objetiva, 
em no máximo 200 caracteres, diretamente à pergunta: <COLOQUE AQUI A PERGUNTA>
"""

def load_llm():
    llm = Llama(
        model_path= MODEL_PATH, 
        temperature=0.7, 
        n_ctx=2040, 
        max_tokens=128
    )

    return llm
