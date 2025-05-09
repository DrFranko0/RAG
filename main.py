import os
import asyncio
import streamlit as st
import requests

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

@st.cache_resource
def build_vector_store():
    loader = DirectoryLoader("data/docs/", glob="**/*.txt")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64, length_function=len)
    chunks = splitter.split_documents(docs)
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embedder)

vector_store = build_vector_store()

provider = OpenAIProvider(base_url="https://api.together.xyz/v1", api_key=TOGETHER_API_KEY)
llm = OpenAIModel("meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", provider=provider)
agent = Agent(model=llm, max_retries=2)

@agent.tool_plain
def calculator_fn(expression: str) -> str:
    try:
        return str(eval(expression, {}, {}))
    except:
        return "Calculation error"

@agent.tool_plain
def dictionary_fn(word: str) -> str:
    try:
        res = requests.get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}").json()
        return res[0]["meanings"][0]["definitions"][0]["definition"]
    except:
        return "Definition not found"

async def rag_and_generate(query: str):
    docs_and_scores = vector_store.similarity_search_with_score(query, k=3)
    context = "\n\n".join([doc.page_content for doc, _ in docs_and_scores])
    prompt = f"Context:\n{context}\n\nQ: {query}\nA:"
    resp = await agent.run(prompt)
    return resp.output, context

async def async_answer(query: str):
    q_low = query.lower()
    if "calculate" in q_low:
        expr = q_low.replace("calculate", "").strip()
        return calculator_fn(expr), "calculator", None
    if "define" in q_low:
        word = q_low.split("define", 1)[1].strip().split()[0]
        return dictionary_fn(word), "dictionary", None
    ans, ctx = await rag_and_generate(query)
    return ans, "RAG", ctx

def answer(query: str):
    try:
        return asyncio.run(async_answer(query))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(async_answer(query))
        loop.close()
        return result

st.title("Knowledge Assistant")
query = st.text_input("Enter your query:")

if query:
    answer_text, branch, context = answer(query)
    st.write(f"Branch used: {branch}")
    if branch == "RAG" and context:
        st.write("Retrieved Context:")
        st.write(context)
    st.write("Answer:")
    st.write(answer_text)
