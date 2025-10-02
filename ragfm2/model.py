from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain.chains import RetrievalQA
import chainlit as cl
from langchain_community.llms import LlamaCpp


DB_FAISS_PATH = "vectorstores/db_faiss"
MODEL_PATH = "/workspaces/lfm2MedicalBot/ragfm2/LFM2-1.2B-F16.gguf"

custom_prompt = """Use as informações disponíveis, avalie como médico e responda de forma objetiva, 
em no máximo 200 caracteres, diretamente à pergunta.

Context: {context}
Question: {question}

Resposta:
"""

def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt, input_variables=["context", "question"])
    return prompt


def load_llm():
    llm = LlamaCpp(
        model_path= MODEL_PATH, 
        temperature=0.3, 
        # n_ctx=2040, 
        max_tokens=100
    )

    return llm



def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm = llm, 
        chain_type="stuff",
         retriever = db.as_retriever(search_kwargs={"k":2}),
          return_source_documents = True, 
          chain_type_kwargs = {"prompt": prompt})

    return qa_chain


def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2", 
    model_kwargs={"device": "cpu"})

    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

def final_result(query):
    qa_result = qa_bot()
    response = qa_result.invoke({"query": query})

    return response["result"]

@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Pergunte algo sobre medicina..")
    await msg.send()
    msg.content = "Bem-vindo ao LFM2HEALTH"
    await msg.update()
    
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, 
        answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    
    cb.answer_reached = True
    
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        answer += f"\nSources: " + str(sources)
    else:
        answer += f"\nNo Sources"
    
    await cl.Message(content=answer).send()