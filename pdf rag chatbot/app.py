from flask import Flask, render_template, request, jsonify
from rag_engine import process_pdf, ask_question
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        if 'pdf' in request.files:
            pdf_file = request.files['pdf']
            if pdf_file.filename != '':
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], pdf_file.filename)
                pdf_file.save(filepath)
                num_chunks = process_pdf(filepath)
                return jsonify({"status": "success", "chunks": num_chunks})

        elif 'question' in request.form:
            question = request.form['question']
            result = ask_question(question)
            return jsonify(result)

    return render_template("index.html")

if __name__ == "__main__":
    print("🚀 Starting PDF RAG Chat...")
    app.run(debug=True)