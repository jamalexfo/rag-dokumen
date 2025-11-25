import os
import sys
from dotenv import load_dotenv
import google.generativeai as genai

# LangChain imports
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()

def initialize_rag_chain():
    # Check for API Key
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")

    print("Initializing RAG System with Google AI (LLM) and HuggingFace (Embeddings)...")

    # 1. Load Documents
    print("Loading documents...")
    
    # Load Text Files
    txt_loader = DirectoryLoader('./documents', glob="**/*.txt", loader_cls=TextLoader)
    txt_docs = txt_loader.load()
    
    # Load PDF Files
    pdf_loader = DirectoryLoader('./documents', glob="**/*.pdf", loader_cls=PyPDFLoader)
    pdf_docs = pdf_loader.load()
    
    # Combine
    docs = txt_docs + pdf_docs
    print(f"Loaded {len(docs)} documents ({len(txt_docs)} txt, {len(pdf_docs)} pdf).")

    if not docs:
        print("Warning: No documents found in ./documents/")
        # Return None or handle empty case, but for now let's proceed or raise
        # return None

    # 2. Split Text
    print("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(f"Created {len(splits)} chunks.")

    # 3. Embed & Store
    print("Creating vector store (using local embeddings)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Persist directory is optional but good for restart. For now, keeping it in-memory/ephemeral for simplicity unless requested.
    # Actually Chroma is ephemeral by default if persist_directory not set.
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # 4. Create Chain
    print("Setting up retrieval chain...")
    
    # Configure Gemini
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    
    # Use GoogleGenerativeAI with correct model name from available models
    llm = GoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        temperature=0.3,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    # Simple prompt template for RetrievalQA
    prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer or the answer is not in the context, just say 'NOT_FOUND' and nothing else.

Context: {context}

Question: {question}

Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # Use RetrievalQA chain (simpler and more stable)
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return rag_chain
