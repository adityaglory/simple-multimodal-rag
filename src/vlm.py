import os
import time
import base64
import mlflow
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma

def encode_image(image_path: str) -> str:
    """Encodes an image to base64 for the VLM."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_retriever():
    """Loads our Chroma database and returns a search engine (retriever)."""
    db_directory = os.path.join(os.path.dirname(__file__), "..", "local_rag_db")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    db = Chroma(persist_directory=db_directory, embedding_function=embeddings)
    return db.as_retriever(search_kwargs={"k": 1})

def run_multimodal_rag(question: str, image_path: str = None, chat_history: list = None, model_name: str = "llava"):
    """
    The core LLMOps pipeline: Retrieve -> Combine -> Generate -> Log.
    Now equipped with Sliding Window Conversational Memory!
    """
    mlflow.set_experiment("Multimodal_RAG_Pipeline")
    
    with mlflow.start_run(run_name="rag_memory_test"):
        print(f"\n--- STEP 1: Fetching context for '{question}' ---")
        retriever = get_retriever()
        
        retrieval_start = time.time()
        docs = retriever.invoke(question)
        retrieval_time = time.time() - retrieval_start
        
        context = "\n".join([doc.page_content for doc in docs]) if docs else "No relevant documents found."
        mlflow.log_metric("retrieval_latency", retrieval_time)
        
        print("\n--- STEP 2: Preparing the Prompt with Memory ---")
        
        # --- NEW: Format the Sliding Window Memory ---
        history_text = ""
        if chat_history:
            history_text = "Previous Conversation History:\n"
            for msg in chat_history:
                role = "User" if msg["role"] == "user" else "AI"
                history_text += f"{role}: {msg['content']}\n"
        
        # We inject the history right above the new question
        augmented_prompt = f"""You are an intelligent IT and Systems assistant.
        Use the following retrieved context to help answer the user's question. 
        
        Retrieved Context:
        {context}
        
        {history_text}
        
        Current User Question: {question}
        """
        
        llm = OllamaLLM(model=model_name, temperature=0.1)
        generation_start = time.time()
        
        if image_path:
            augmented_prompt += "\nPlease also consider the provided image in your response."
            base64_image = encode_image(image_path)
            response = llm.bind(images=[base64_image]).invoke(augmented_prompt)
        else:
            response = llm.invoke(augmented_prompt)
            
        generation_time = time.time() - generation_start
        
        mlflow.log_metric("generation_latency", generation_time)
        mlflow.log_param("model", model_name)
        
        print(f"\n--- Final AI Response (Took {generation_time:.2f}s) ---")
        return response

if __name__ == "__main__":
    print("==================================================")
    print("   Welcome to your Multimodal RAG CLI Engine!     ")
    print("==================================================")
    print("Type 'quit' or 'exit' at any time to stop.")
    
    while True:
        print("-" * 50)
        # 1. Get the text prompt
        user_question = input("\nEnter your question: ").strip()
        
        # Check if the user wants to exit
        if user_question.lower() in ['quit', 'exit', 'q']:
            print("Shutting down the engine. Great coding today!")
            break
            
        if not user_question:
            print("Question cannot be empty. Please try again.")
            continue
            
        # 2. Get the optional image
        user_image = input("Enter image path (or press Enter to skip for text-only): ").strip()
        
        # 3. Validate the image path
        if user_image == "":
            user_image = None # Text-only mode
        elif not os.path.exists(user_image):
            print(f"⚠️ Warning: Could not find an image at '{user_image}'. Proceeding with text-only.")
            user_image = None
            
        # 4. Run the pipeline dynamically!
        try:
            run_multimodal_rag(question=user_question, image_path=user_image)
        except Exception as e:
            print(f"❌ An error occurred during generation: {e}")