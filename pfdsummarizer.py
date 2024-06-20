import os
import PyPDF2
from PIL import Image
import pytesseract

# Directory for storing PDF resumes and job applications
pdf_directory = 'D:\Projects\PPTsummarizer'

# Directory for storing extracted text from PDFs
text_directory = 'D:\Projects\PPTsummarizer'

# OCR output directory for scanned PDFs
ocr_directory = 'D:\Projects\PPTsummarizer'

# Directory for storing summaries
summary_directory = 'D:\Projects\PPTsummarizer'

# Create directories if they don't exist
os.makedirs(pdf_directory, exist_ok=True)
os.makedirs(text_directory, exist_ok=True)
os.makedirs(ocr_directory, exist_ok=True)
os.makedirs(summary_directory, exist_ok=True)



for file_name in os.listdir(pdf_directory):
    if file_name.endswith('.pdf'):
        # Open the PDF file
        with open(os.path.join(pdf_directory, file_name), 'rb') as file:
            # Create a PDF reader object
            reader = PyPDF2.PdfReader(file)

            # Extract text from each page
            text = ''
            for page in reader.pages:
                text += page.extract_text()

            # Save the extracted text as a text file
            text_file_name = file_name.replace('.pdf', '.txt')
            text_file_path = os.path.join(text_directory, text_file_name)
            with open(text_file_path, 'w') as text_file:
                text_file.write(text)



# Optional Step
for file_name in os.listdir(pdf_directory):
    if file_name.endswith('.pdf'):
        # Open the PDF file
        with Image.open(os.path.join(pdf_directory, file_name)) as img:
            # Perform OCR using pytesseract
            ocr_text = pytesseract.image_to_string(img, lang='eng')

            # Save the OCR output as a text file
            ocr_file_name = file_name.replace('.pdf', '.txt')
            ocr_file_path = os.path.join(ocr_directory, ocr_file_name)
            with open(ocr_file_path, 'w') as ocr_file:
                ocr_file.write(ocr_text)


pdf_directory = 'D:\presentations'

resume_files = []
for file_name in os.listdir(pdf_directory):
    if file_name.endswith('.pdf'):
        resume_files.append(os.path.join(pdf_directory, file_name))

resume_summaries = []  # To store the generated summaries

# Loop through each resume file
for resume_file in resume_files:
    with open(resume_file, 'rb') as file:
        # Create a PDF reader object
        reader = PyPDF2.PdfReader(file)

        # Extract text from each page
        text = ''
        for page in reader.pages:
            text += page.extract_text()


# Continuing the loop from the previous step
        from transformers import T5ForConditionalGeneration,T5Tokenizer

        # Initialize the model and tokenizer
        model = T5ForConditionalGeneration.from_pretrained("t5-base")
        tokenizer = T5Tokenizer.from_pretrained("t5-base")

        # Encode the text
        inputs = tokenizer.encode("summarize: " + text, 
        return_tensors="pt", max_length=1000, 
        truncation=True)

        # Generate the summary
        outputs = model.generate(inputs, 
        max_length=100, min_length=10, 
        length_penalty=2.0, num_beams=4, 
        early_stopping=True)

        # Decode the summary
        summary = tokenizer.decode(outputs[0])

        resume_summaries.append(summary)

# Print the generated summaries for each resume
for i, summary in enumerate(resume_summaries):
    print(f"Summary for {i+1}:")
    print(summary)
    print()