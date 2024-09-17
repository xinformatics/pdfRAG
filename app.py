from flask import Flask, request, jsonify, render_template
from langchain_ollama.llms import OllamaLLM
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders.pdf import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
import os

app = Flask(__name__)

db_path = "db"

model_name = ['phi3.5','gemma2:2b','gemma2:2b-instruct-q5_K_M','gemma2:9b-instruct-q2_K','qwen2:1.5b-instruct-q5_K_M']
model_index= 3

cached_llm = OllamaLLM(model=model_name[model_index])

emb = OllamaEmbeddings(model=model_name[model_index])

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, 
    chunk_overlap=80, 
    length_function=len, 
    is_separator_regex=False)

raw_prompt = PromptTemplate.from_template(
    """ 
    <s>[INST] You are a technical assistant good at searching docuemnts. If you do not have an answer from the provided information say so. [/INST] </s>
    [INST] {input}
           Context: {context}
           Answer:
    [/INST]
"""
)

# Route for the home page where users can input a query and see a response
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling AI queries
@app.route("/ai", methods=["POST"])
def aiPost():
    query = request.form.get("query")  # Get query from form submission
    response = cached_llm.invoke(query)
    return render_template('index.html', query=query, answer=response)  # Display result on the same page

# Route for handling PDF uploads
@app.route("/upload_pdf", methods=["POST"])
def pdfPost():
    file = request.files["file"]
    file_name = file.filename
    save_file = os.path.join("pdf", file_name)
    file.save(save_file)

    loader = PDFPlumberLoader(save_file)
    docs = loader.load_and_split()

    chunks = text_splitter.split_documents(docs)
    
    vector_store = Chroma.from_documents(chunks, emb, persist_directory=db_path)

    response = {
        "status": "Successfully Uploaded", 
        "filename": file_name,
        "doc_len": len(docs),
        "chunks": len(chunks)
    }

    return render_template('index.html', upload_response=response)

# Route for asking questions based on a PDF
@app.route("/ask_pdf", methods=["POST"])
def askPDFPost():
    query = request.form.get("query")

    vector_store = Chroma(persist_directory=db_path, embedding_function=emb)

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 20, "score_threshold": 0.1}
    )

    document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
    chain = create_retrieval_chain(retriever, document_chain)

    result = chain.invoke({"input": query})

    sources = [{"source": doc.metadata["source"], "page_content": doc.page_content} for doc in result["context"]]

    response_answer = {"answer": result["answer"], "sources": sources}

    return render_template('index.html', query=query, answer=result["answer"], sources=sources)

def start_app():
    app.run(host="0.0.0.0", port=8080, debug=True)

if __name__ == "__main__":
    start_app()