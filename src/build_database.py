import os
# --- UPDATED IMPORTS HERE ---
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

def get_database():
    """
    Initializes and returns our local Chroma database.
    It uses 'nomic-embed-text' to convert our text into searchable vectors.
    """
    print("Loading embedding model...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    db_directory = os.path.join(os.path.dirname(__file__), "..", "local_rag_db")
    
    # Create or load the database
    db = Chroma(persist_directory=db_directory, embedding_function=embeddings)
    return db

def add_documents_to_db(db, texts: list, metadatas: list):
    """Takes a list of strings, converts them to Document objects, and adds them to ChromaDB."""
    print(f"Adding {len(texts)} documents to the database...")
    docs = [Document(page_content=text, metadata=meta) for text, meta in zip(texts, metadatas)]
    db.add_documents(docs)
    print("Documents successfully added and saved!")

def search_database(db, query: str):
    """Searches the database for the pieces of text most similar to the user's query."""
    print(f"\nSearching database for: '{query}'")
    results = db.similarity_search(query, k=2)
    
    print("\n--- Search Results ---")
    for i, res in enumerate(results):
        print(f"Result {i+1} (Source: {res.metadata['source']}):")
        print(f"{res.page_content}\n")

if __name__ == "__main__":
    my_db = get_database()
    
    sample_texts = [
        "The server rack overheated on Tuesday because the cooling fan failed.",
        "The new Multimodal RAG system uses LLaVA for vision and Chroma for storage.",
        "Company policy dictates that all passwords must be rotated every 90 days."
    ]
    
    sample_metadata = [
        {"source": "IT_Log_Tuesday.txt"},
        {"source": "Project_Architecture.pdf"},
        {"source": "Security_Manual.pdf"}
    ]
    
    # We can keep adding them, Chroma is smart enough to handle it!
    add_documents_to_db(my_db, sample_texts, sample_metadata)
    
    search_database(my_db, "Why was the server room hot?")
    search_database(my_db, "What models are we using for our new project?")