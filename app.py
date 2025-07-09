from flask import Flask, render_template, request, send_from_directory
from rag.retriever import get_esg_answer
import os

app = Flask(__name__)
chat_history = []

@app.route("/", methods=["GET", "POST"])
def index():
    global chat_history

    if request.method == "POST":
        user_input = request.form["query"].strip()
        if user_input:
            chat_history.append({"role": "user", "text": user_input})

            try:
                response = get_esg_answer(user_input)

                answer_text = response.get("text", "")
                chart = response.get("chart", False)

                chat_history.append({"role": "bot", "text": answer_text, "chart": chart})
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                chat_history.append({"role": "bot", "text": error_msg, "chart": False})

    return render_template("index.html", chat_history=chat_history)

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)

if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    app.run(debug=True)

