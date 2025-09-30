import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

def setup_rag(rag_docs, index_name="stocks-index-3072", batch_size=50, sleep_time=10.0):
    """
    Setup RAG system with Pinecone and Gemini embeddings.

    :param rag_docs: List of document strings.
    :param index_name: Name of the Pinecone index.
    :param batch_size: Number of documents per batch for insertion.
    :param sleep_time: Seconds to wait between batches to avoid rate limits.
    """
    # Load API keys
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")

    if not pinecone_api_key or not google_api_key:
        raise ValueError("Missing API keys in .env")

    # Initialize Pinecone
    pc = Pinecone(api_key=pinecone_api_key)
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=3072,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    # Initialize embeddings with API key
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=google_api_key
    )

 
    # Batch insertion into Pinecone with time delay

    print(f"Inserting {len(rag_docs)} docs in batches of {batch_size}...")
    for i in range(0, len(rag_docs), batch_size):
        batch = rag_docs[i:i + batch_size]
        PineconeVectorStore.from_texts(
            texts=batch,
            embedding=embeddings,
            index_name=index_name
        )
        print(f"Inserted batch {i // batch_size + 1} ({len(batch)} docs)")
        if i + batch_size < len(rag_docs):
            print(f"Sleeping for {sleep_time} seconds to avoid rate limits...")
            time.sleep(sleep_time)

    # Create retriever from the full index
    vectorstore = PineconeVectorStore(
        embedding=embeddings,
        index_name=index_name
    )
    retriever = vectorstore.as_retriever()

    
    # LLM + prompt
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=google_api_key
    )

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
