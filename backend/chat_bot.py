import yfinance as yf
from segment_tree import SegmentTree
from rag_setup import setup_rag
import re
import json


# Fetch stock data

tickers = ["AAPL", "MSFT", "TSLA"]
data = yf.download(tickers, start="2025-01-01", end="2025-01-15")["Close"]

trees = {}
rag_docs = []
dates = data.index.strftime("%Y-%m-%d").tolist()

for ticker in tickers:
    prices = data[ticker].dropna().tolist()
    trees[ticker] = SegmentTree(prices)
    rag_docs.extend([f"On {dates[i]}, {ticker} closed at ${prices[i]}." for i in range(len(prices))])


# Setup RAG

rag_chain = setup_rag(rag_docs)


# Helper functions

def date_range_to_indices(start_date, end_date, dates):
    try:
        start_idx = dates.index(start_date)
    except ValueError:
        start_idx = 0
    try:
        end_idx = dates.index(end_date)
    except ValueError:
        end_idx = len(dates)-1
    return start_idx, end_idx

 
def execute_command(command_json, trees, dates):
    stock = command_json["stock"]
    operation = command_json["operation"]
    start_date = command_json["start"]
    end_date = command_json["end"]

    start_idx, end_idx = date_range_to_indices(start_date, end_date, dates)
    tree = trees.get(stock)
    if not tree:
        return f"Unknown stock: {stock}"

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

def extract_json(text):
    """
    Extracts all {...} JSON objects from string.
    Returns a list of dicts.
    """
    matches = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
    commands = []
    for m in matches:
        try:
            commands.append(json.loads(m))
        except:
            continue
    return commands

def chatbot_query(user_input):
    response = rag_chain.invoke({"input": user_input})
    answer = response["answer"]

    commands = extract_json(answer)

    if commands:
        results = []
        for cmd in commands:
            if cmd.get("action") == "query_segment_tree":
                numeric_answer = execute_command(cmd, trees, dates)
                cmd["numeric_answer"] = numeric_answer
                results.append(cmd)

        if len(results) > 1:
            
            op = results[0]["operation"]
            if op in ["avg", "min", "max", "sum"]:
                # Find stock with highest/lowest value
                if op in ["avg", "max", "sum"]:
                    best = max(results, key=lambda x: x["numeric_answer"])
                else:  # min
                    best = min(results, key=lambda x: x["numeric_answer"])

                return (f"The {op} closing price comparison shows that {best['stock']} "
                        f"had the {op} value of ${best['numeric_answer']:.2f} "
                        f"from {best['start']} to {best['end']}.")

        # Single stock
        cmd = results[0]
        op_text = {"min":"lowest","max":"highest","avg":"average","sum":"total"}.get(cmd["operation"], cmd["operation"])
        return f"The {op_text} closing price of {cmd['stock']} from {cmd['start']} to {cmd['end']} was ${cmd['numeric_answer']:.2f}."

    return answer



# Chatbot loop

if __name__ == "__main__":
    print("Financial Chatbot (type 'exit' to quit)")
    while True:
        q = input("\nUser: ")
        if q.lower() in ["exit", "quit"]:
            break
        print("Bot:", chatbot_query(q))
