import yfinance as yf
import pandas as pd


# Enhanced Segment Tree

class SegmentTree:
    def __init__(self, data):
        self.n = len(data)
        self.tree = [0] * (4 * self.n)
        self.build(data, 0, 0, self.n - 1)

    def build(self, data, node, l, r):
        if l == r:
            self.tree[node] = (data[l], data[l], data[l], 1)  # min, max, sum, count
            return
        mid = (l + r) // 2
        self.build(data, 2*node+1, l, mid)
        self.build(data, 2*node+2, mid+1, r)
        self.tree[node] = self._merge(self.tree[2*node+1], self.tree[2*node+2])

    def _merge(self, left, right):
        return (
            min(left[0], right[0]),
            max(left[1], right[1]),
            left[2] + right[2],
            left[3] + right[3]
        )

    def query(self, ql, qr, node=0, l=0, r=None):
        if r is None: r = self.n-1
        if qr < l or ql > r: return (float('inf'), float('-inf'), 0, 0)
        if ql <= l and r <= qr: return self.tree[node]
        mid = (l + r) // 2
        left = self.query(ql, qr, 2*node+1, l, mid)
        right = self.query(ql, qr, 2*node+2, mid+1, r)
        return self._merge(left, right)

    def range_min(self, l, r): return self.query(l, r)[0]
    def range_max(self, l, r): return self.query(l, r)[1]
    def range_sum(self, l, r): return self.query(l, r)[2]
    def range_avg(self, l, r): 
        q = self.query(l, r)
        return q[2]/q[3] if q[3] else None


# Fetch stock data and build RAG docs

stocks = ["AAPL", "MSFT", "TSLA"]
data = yf.download(stocks, start="2025-01-01", end="2025-02-01")["Close"]

trees = {}
rag_docs = {}

for stock in stocks:
    prices = data[stock].dropna().tolist()
    dates = data.index.strftime("%Y-%m-%d").tolist()
    
    trees[stock] = SegmentTree(prices)
    rag_docs[stock] = [
        {"text": f"On {dates[i]}, {stock} closed at ${prices[i]}.", "price": prices[i], "date": dates[i]}
        for i in range(len(prices))
    ]


# Resilient date mapping

def date_range_to_indices(start_date, end_date, dates):
    """Find nearest available indices for start_date and end_date"""
    start_idx = next((i for i, d in enumerate(dates) if d >= start_date), 0)
    end_idx = next((i for i, d in reversed(list(enumerate(dates))) if d <= end_date), len(dates)-1)
    return start_idx, end_idx


# Execute numeric command

def execute_command(command, trees, data):
    op = command["operation"]
    stocks = command["stocks"]
    start_date = command["start_date"]
    end_date = command["end_date"]

    results = {}
    for stock in stocks:
        prices = data[stock].loc[start_date:end_date].dropna().tolist()
        if not prices:
            continue
        tree = SegmentTree(prices)
        if op == "min":
            results[stock] = tree.range_min(0, len(prices)-1)
        elif op == "max":
            results[stock] = tree.range_max(0, len(prices)-1)
        elif op == "sum":
            results[stock] = tree.range_sum(0, len(prices)-1)
        elif op == "avg":
            results[stock] = tree.range_avg(0, len(prices)-1)

    if op in ["min", "max", "avg"]:
        best_stock = max(results, key=results.get) if op in ["max", "avg"] else min(results, key=results.get)
        return f"{best_stock} has the {op} value of {results[best_stock]} among {stocks}"
    return results


# chatbot (simulated)

def chatbot_query(user_query):
  

    user_query_lower = user_query.lower()
    if "highest average" in user_query_lower:
        command = {"operation": "avg", "stocks": stocks, "start_date": "2025-01-01", "end_date": "2025-01-31"}
    elif "lowest price" in user_query_lower:
        command = {"operation": "min", "stocks": stocks, "start_date": "2025-01-01", "end_date": "2025-01-31"}
    elif "highest price" in user_query_lower:
        command = {"operation": "max", "stocks": stocks, "start_date": "2025-01-01", "end_date": "2025-01-31"}
    elif "total sum" in user_query_lower:
        command = {"operation": "sum", "stocks": stocks, "start_date": "2025-01-01", "end_date": "2025-01-31"}
    else:
        return "Sorry, I currently support min, max, sum, avg queries only."

    numeric_answer = execute_command(command, trees, data)

    
    stock_docs = []
    for stock in command["stocks"]:
        stock_docs += rag_docs[stock][:3]  # top 3 docs for demo
    doc_text = "\n".join([d["text"] for d in stock_docs])

    return f"{numeric_answer}\n"

# Example chatbot queries

queries = [
    "Which stock had the highest average price in January 2025?",
    "Which stock had the lowest price in January 2025?",
    "Total sum of all stocks in January 2025?"
]

for q in queries:
    print("\nUser:", q)
    print("Bot:", chatbot_query(q))
