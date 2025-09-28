# rag_setup.py
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

def setup_rag(rag_docs, index_name="stocks-index-3072"):
    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=3072,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    vectorstore = PineconeVectorStore.from_texts(
        texts=rag_docs,
        embedding=embeddings,
        index_name=index_name
    )

    retriever = vectorstore.as_retriever()

    # Setup LLM + prompt
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

    system_prompt = """
    You are a financial assistant.
    - Use RAG retrieval for stock-related queries.
    - If user asks about min, max, avg, or sum prices, generate a JSON command like:
      {{ "action": "query_segment_tree", "stock": "AAPL", "operation": "avg", "start": "2025-01-01", "end": "2025-01-15" }}
    - Use the retrieved context (historical stock docs) to help answer questions.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Context: {context}\nQuestion: {input}")
    ])

    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)

    return rag_chain
