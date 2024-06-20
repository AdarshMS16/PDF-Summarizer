from flask import Flask, request, render_template, redirect, url_for
import os
import PyPDF2
from transformers import T5ForConditionalGeneration, T5Tokenizer

app = Flask(__name__)

# Directory configuration
pdf_directory = 'D:/Projects/PPTsummarizer/pdf'
text_directory = 'D:/Projects/PPTsummarizer/text'
summary_directory = 'D:/Projects/PPTsummarizer/summaries'

# Ensure directories exist
os.makedirs(pdf_directory, exist_ok=True)
os.makedirs(text_directory, exist_ok=True)
os.makedirs(summary_directory, exist_ok=True)

# Initialize model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("t5-base")
tokenizer = T5Tokenizer.from_pretrained("t5-base")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file and file.filename.endswith('.pdf'):
        file_path = os.path.join(pdf_directory, file.filename)
        file.save(file_path)

        # Process the PDF
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text()

            # Save the extracted text as a text file
            text_file_name = file.filename.replace('.pdf', '.txt') if file and hasattr(file, 'filename') else 'default.txt'
            text_file_path = os.path.join(text_directory, text_file_name)
            with open(text_file_path, 'w') as text_file:
                text_file.write(text)

            # Generate summary
            inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1000, truncation=True)
            outputs = model.generate(inputs, max_length=1000, min_length=100, length_penalty=2.0, num_beams=4, early_stopping=True)
            summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Save summary
            summary_file_path = os.path.join(summary_directory, text_file_name)
            with open(summary_file_path, 'w') as summary_file:
                summary_file.write(summary)

            return redirect(url_for('summary', filename=text_file_name))

    return redirect(request.url)

@app.route('/summary/<filename>')
def summary(filename):
    summary_file_path = os.path.join(summary_directory, filename)
    with open(summary_file_path, 'r') as file:
        summary = file.read()
    return render_template('summary.html', summary=summary, filename=filename)

if __name__ == '__main__':
    app.run(debug=True)
