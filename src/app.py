import streamlit as st
import os
import tempfile
from vlm import run_multimodal_rag

# 1. Page Configuration
st.set_page_config(page_title="Multimodal RAG Engine", page_icon="🧠", layout="wide")
st.title("🧠 Multimodal RAG Assistant")

# Initialize chat history in Streamlit's session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. Sidebar for Image Uploads and Controls
with st.sidebar:
    st.header("🖼️ Image Input (Optional)")
    st.write("Drag and drop an image here if your question is about a picture.")
    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_image:
        st.image(uploaded_image, caption="Ready for AI Analysis", use_container_width=True)
        
    st.divider()
    # Add a button to clear the chat history
    if st.button("Clear Chat History", type="secondary"):
        st.session_state.messages = []
        st.rerun()

# 3. Display existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Main Chat Input
# st.chat_input pins the text box to the bottom of the screen!
if prompt := st.chat_input("Ask a question about your documents or image..."):
    
    # Immediately show the user's message in the chat
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Add the user's message to our saved history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Show the AI thinking, then display its response
    with st.chat_message("assistant"):
        with st.spinner("Searching database and analyzing..."):
    # ... inside the with st.chat_message("assistant"): block ...
            # ... inside the with st.chat_message("assistant"): block ...
                try:
                    temp_image_path = None
                    
                    if uploaded_image:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                            tmp_file.write(uploaded_image.getvalue())
                            temp_image_path = tmp_file.name

                    # --- THE FIX: Grab history, but EXCLUDE the brand new question we just added! ---
                    # We take the last 5 messages, but stop before the very last one ([:-1])
                    recent_history = st.session_state.messages[-5:-1] if len(st.session_state.messages) > 1 else None

                    # --- Run Your Core AI Engine with Memory ---
                    response = run_multimodal_rag(
                        question=prompt, 
                        image_path=temp_image_path,
                        chat_history=recent_history 
                    )
                    
                    # Safety net: If the AI returns a blank string, tell the user!
                    if not response or response.isspace():
                        response = "⚠️ The model returned an empty response. Please try asking your question slightly differently."
                    
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

                except Exception as e:
                    st.error(f"❌ An error occurred in the backend: {e}")
                
                finally:
                    # Clean up the temporary file so we don't leak memory/storage
                    if temp_image_path and os.path.exists(temp_image_path):
                        os.remove(temp_image_path)