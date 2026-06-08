from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma

# ================== CONFIG ==================
embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = OllamaLLM(model="llama3.1", temperature=0.4)

print("✅ Using Ollama Embeddings + Llama 3.1")

vectorstore = None

def process_pdf(pdf_path: str):
    global vectorstore
    
    print("📄 Loading PDF...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print("✂️ Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=850,
        chunk_overlap=180
    )
    chunks = text_splitter.split_documents(documents)

    print("🔍 Creating vector database...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print(f"✅ Success! {len(chunks)} chunks created.")
    return len(chunks)


def ask_question(question: str):
    if vectorstore is None:
        return {"answer": "Please upload a PDF first.", "sources": []}

    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    relevant_chunks = retriever.invoke(question)

    context = "\n\n".join([doc.page_content for doc in relevant_chunks])

    prompt = f"""
You are a helpful and accurate assistant. Answer the question based on the provided context from the PDF.

Context:
{context}

Question: {question}

Provide a clear, detailed, and well-structured answer.
If the question is about any Project, explain its Objective, Requirements, and Constraints properly.

Answer:
"""

    response = llm.invoke(prompt)

    sources = []
    for i, doc in enumerate(relevant_chunks, 1):
        page = doc.metadata.get('page', 'Unknown')
        sources.append(f"Page {page + 1}")

    return {
        "answer": response.strip(),
        "sources": sources
    }