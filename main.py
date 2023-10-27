from flask import Flask, render_template, request, jsonify
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from PyPDF2 import PdfReader
from docx import Document

app = Flask(__name__)

chat_history = []

# Function to split text into smaller chunks that don't exceed the token limit
def split_text_into_chunks(text, max_tokens=4096):
    chunks = []
    current_chunk = ""
    
    for line in text.split("\n"):
        if len(current_chunk) + len(line) < max_tokens:
            current_chunk += line + "\n"
        else:
            chunks.append(current_chunk)
            current_chunk = line + "\n"
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def get_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_from_docx(docx_path):
    doc = Document(docx_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def get_text_from_doc(doc_path):
    # Implement your logic to extract text from DOC files
    # You can use libraries like python-docx or other suitable tools
    pass

@app.route('/')
def index():
    return render_template('index.html', chat_history=chat_history)  # Pass chat_history to the template

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.form.get('user_input')
    if user_input:
        # Specify the folder containing different file types
        files_folder = 'data'

        text = ""
        for filename in os.listdir(files_folder):
            file_path = os.path.join(files_folder, filename)
            if filename.endswith('.pdf'):
                text += get_text_from_pdf(file_path)
            elif filename.endswith('.docx'):
                text += get_text_from_docx(file_path)
            elif filename.endswith('.doc'):
                text += get_text_from_doc(file_path)
                

        # text_splitter = CharacterTextSplitter(
        #     #separator="\n",
        #     #chunk_size=1000,
        #     #chunk_overlap=200,
        #     #length_function=len
        # )
        # chunks = text_splitter.split_text(text)
        # Split the text into smaller chunks to avoid exceeding token limit
        text_chunks = split_text_into_chunks(text)

        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(text_chunks, embeddings)

        docs = knowledge_base.similarity_search(user_input)

        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")

        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=user_input)
        
        response = {'response': response}

        # Append user input and AI response to the chat history
        chat_history.append({'user': user_input, 'assistant': response['response']})

        return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
