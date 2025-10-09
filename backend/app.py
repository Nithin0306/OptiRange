from flask import Flask, request, jsonify
from flask_cors import CORS


try:
    from chat_bot import chatbot_query  
except ImportError:
  
    def chatbot_query(user_input: str) -> str:
        return "Backend not fully initialized. Please wire chatbot_query."

app = Flask(__name__)


CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    supports_credentials=False,
    allow_headers=["Content-Type", "Authorization"],
    methods=["GET", "POST", "OPTIONS"],
)

@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200

@app.post("/chat")
def chat():
    try:
        data = request.get_json(silent=True) or {}
        user_message = (data.get("message") or "").strip()
        if not user_message:
            return jsonify({"error": "Message is required"}), 400

        answer = chatbot_query(user_message)
        return jsonify({"answer": answer}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    
    app.run(host="0.0.0.0", port=5000, debug=True)
