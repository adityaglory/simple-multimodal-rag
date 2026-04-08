import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

def ingest_pdfs(data_dir: str, db_dir: str):
    """
    Scans the data directory for PDFs, chunks their text, and saves them to ChromaDB.
    """
    print(f"Scanning for PDFs in {data_dir}...")
    
    # Find all PDF files in the data folder
    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
    
    if not pdf_files:
        print("No PDFs found! Please add some .pdf files to your data/ folder.")
        return

    # Load all pages from all PDFs
    all_docs = []
    for pdf in pdf_files:
        print(f"Loading {os.path.basename(pdf)}...")
        loader = PyPDFLoader(pdf)
        docs = loader.load()
        all_docs.extend(docs)

    print(f"Loaded {len(all_docs)} pages. Now chunking text...")
    
    # ---------------------------------------------------------
    # THE CHUNKER: This is the most important part of RAG!
    # We split text into chunks of 1000 characters.
    # The 'overlap' of 200 ensures we don't cut a sentence in half.
    # ---------------------------------------------------------
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(all_docs)
    print(f"Split into {len(chunks)} manageable chunks.")

    # Initialize our exact same ChromaDB setup
    print("Loading embedding model (nomic-embed-text)...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma(persist_directory=db_dir, embedding_function=embeddings)
    
    # Save the chunks to the database
    print("Saving chunks to ChromaDB...")
    db.add_documents(chunks)
    print("Success! Your database has been updated with real PDF data.")

if __name__ == "__main__":
    # Define our absolute paths based on the project structure
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_directory = os.path.join(base_dir, "data")
    db_directory = os.path.join(base_dir, "local_rag_db")
    
    ingest_pdfs(data_directory, db_directory)