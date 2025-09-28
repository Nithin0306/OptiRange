import os
import yfinance as yf
import pandas as pd
from dotenv import load_dotenv
import json

# Pinecone + LangChain
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()

# ----------------------------
# Enhanced Segment Tree
# ----------------------------
class SegmentTree:
    def __init__(self, data):
        self.n = len(data)
        self.tree = [0] * (4 * self.n)
        self.build(data, 0, 0, self.n - 1)

    def build(self, data, node, l, r):
        if l == r:
            self.tree[node] = (data[l], data[l], data[l], 1)  # (min, max, sum, count)
            return
        mid = (l + r) // 2
        self.build(data, 2 * node + 1, l, mid)
        self.build(data, 2 * node + 2, mid + 1, r)
        self.tree[node] = self._merge(self.tree[2 * node + 1], self.tree[2 * node + 2])

    def _merge(self, left, right):
        return (
            min(left[0], right[0]),
            max(left[1], right[1]),
            left[2] + right[2],
            left[3] + right[3]
        )

    def query(self, ql, qr, node=0, l=0, r=None):
        if r is None:
            r = self.n - 1
        if qr < l or ql > r:
            return (float('inf'), float('-inf'), 0, 0)
        if ql <= l and r <= qr:
            return self.tree[node]
        mid = (l + r) // 2
        left = self.query(ql, qr, 2 * node + 1, l, mid)
        right = self.query(ql, qr, 2 * node + 2, mid + 1, r)
        return self._merge(left, right)

    def range_min(self, l, r): return self.query(l, r)[0]
    def range_max(self, l, r): return self.query(l, r)[1]
    def range_sum(self, l, r): return self.query(l, r)[2]
    def range_avg(self, l, r):
        q = self.query(l, r)
        return q[2] / q[3] if q[3] else None


# ----------------------------
# Fetch stock data
# ----------------------------
tickers = ["AAPL", "MSFT", "TSLA"]
data = yf.download(tickers, start="2025-01-01", end="2025-02-01")["Close"]

trees = {}
rag_docs = []
dates = data.index.strftime("%Y-%m-%d").tolist()

for ticker in tickers:
    prices = data[ticker].dropna().tolist()
    # Build segment tree
    trees[ticker] = SegmentTree(prices)
    # Build docs for RAG
    rag_docs.extend([
        f"On {dates[i]}, {ticker} closed at ${prices[i]}."
        for i in range(len(prices))
    ])

# ----------------------------
# Setup Pinecone Vector DB (v4 API)
# ----------------------------
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "stocks-index-3072"


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

# ----------------------------
# Setup Gemini LLM + RAG chain
# ----------------------------
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
    ("human", "Context: {context}\nQuestion: {input}"),
])


qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)

# ----------------------------
# Helper: date → indices
# ----------------------------
def date_range_to_indices(start_date, end_date, dates):
    try:
        start_idx = dates.index(start_date)
    except ValueError:
        start_idx = 0
    try:
        end_idx = dates.index(end_date)
    except ValueError:
        end_idx = len(dates) - 1
    return start_idx, end_idx

# ----------------------------
# Execute command from Gemini
# ----------------------------
def execute_command(command_json, trees, dates):
    """
    command_json: dict from LLM
    trees: dict of SegmentTree objects per stock
    dates: list of date strings corresponding to tree indices
    """
    stock = command_json["stock"]
    operation = command_json["operation"]
    start_date = command_json["start"]
    end_date = command_json["end"]

    # Map dates to indices
    try:
        start_idx = dates.index(start_date)
        end_idx = dates.index(end_date)
    except ValueError:
        return f"Date out of range: {start_date} to {end_date}"

    tree = trees.get(stock)
    if not tree:
        return f"Unknown stock: {stock}"

    # Perform the operation
    if operation == "min":
        return tree.range_min(start_idx, end_idx)
    elif operation == "max":
        return tree.range_max(start_idx, end_idx)
    elif operation == "sum":
        return tree.range_sum(start_idx, end_idx)
    elif operation == "avg":
        return tree.range_avg(start_idx, end_idx)
    else:
        return f"Unknown operation: {operation}"

# ----------------------------
# Chatbot interface
# ----------------------------
def chatbot_query(user_input):
    response = rag_chain.invoke({"input": user_input})
    answer = response["answer"]

    import json
    try:
        command = json.loads(answer)
        if "action" in command and command["action"] == "query_segment_tree":
            numeric_answer = execute_command(command, trees, dates)
            return f"Computed from Segment Tree: {numeric_answer}"
    except:
        pass

    return answer

# ----------------------------
# Example run
# ----------------------------
if __name__ == "__main__":
    while True:
        q = input("\nUser: ")
        if q.lower() in ["exit", "quit"]:
            break
        print("Bot:", chatbot_query(q))
