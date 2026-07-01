"""
Web UI for Local RAG System using Streamlit
Provides an interactive interface for document ingestion and Q&A
"""

import streamlit as st
from pathlib import Path
import sys
from rag_system_simple import SimpleRAGSystem
import logging

# Configure Streamlit
st.set_page_config(
    page_title="Local RAG Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = SimpleRAGSystem(
        vector_store_path="vector_store",
        embedding_model="nomic-embed-text",
        query_model="llama2"
    )

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    st.title("📚 Local RAG-Based Document Q&A")
    st.markdown("*Powered by Ollama, Chromadb, and nomic-embed-text*")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Collection stats
        stats = st.session_state.rag_system.get_collection_stats()
        st.metric("Indexed Chunks", stats.get('total_chunks', 0))
        
        st.markdown("---")
        st.subheader("📤 Document Upload")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_uploader"
        )
        
        if uploaded_files:
            if st.button("Process Documents", key="process_btn"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {uploaded_file.name}...")
                    
                    # Save temporarily
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Ingest
                    success = st.session_state.rag_system.ingest_pdf(
                        temp_path,
                        uploaded_file.name.replace('.pdf', '')
                    )
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                    
                    # Cleanup
                    Path(temp_path).unlink()
                
                status_text.success("✓ Documents processed successfully!")
                st.balloons()
        
        # Folder loading
        st.markdown("---")
        st.subheader("📁 Load from Folder")
        
        documents_folder = st.text_input("Documents folder path:", value="documents")
        
        if st.button("Load Folder", key="load_folder_btn"):
            if Path(documents_folder).exists():
                results = st.session_state.rag_system.load_documents_from_folder(documents_folder)
                
                with st.expander("Loading Results"):
                    for filename, success in results.items():
                        status = "✓" if success else "✗"
                        st.write(f"{status} {filename}")
                
                st.success("Folder loaded!")
            else:
                st.error(f"Folder not found: {documents_folder}")
        
        # Model configuration
        st.markdown("---")
        st.subheader("🤖 Model Settings")
        
        embedding_model = st.selectbox(
            "Embedding Model",
            ["nomic-embed-text"],
            help="Model for generating embeddings"
        )
        
        query_model = st.selectbox(
            "Query Model",
            ["llama2", "llama3", "mistral", "neural-chat"],
            help="Model for generating answers"
        )
        
        num_context = st.slider(
            "Context Chunks",
            min_value=1,
            max_value=10,
            value=3,
            help="Number of relevant chunks to retrieve"
        )
        
        st.session_state.num_context = num_context
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Ask a Question")
        
        query = st.text_area(
            "Enter your question:",
            placeholder="What would you like to know about your documents?",
            height=100,
            key="query_input"
        )
        
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            ask_button = st.button("🔍 Get Answer", use_container_width=True)
        
        with col_btn2:
            clear_button = st.button("🗑️ Clear History", use_container_width=True)
        
        if clear_button:
            st.session_state.chat_history = []
            st.rerun()
        
        if ask_button and query:
            with st.spinner("Retrieving relevant documents..."):
                answer = st.session_state.rag_system.answer_query(
                    query,
                    st.session_state.num_context
                )
            
            st.session_state.chat_history.append({
                'query': query,
                'answer': answer
            })
    
    with col2:
        st.subheader("📊 Statistics")
        stats = st.session_state.rag_system.get_collection_stats()
        
        st.info(
            f"""
            **Collection Info:**
            - Total Chunks: {stats.get('total_chunks', 0)}
            - Embedding Model: {stats.get('embedding_model', 'N/A')}
            - Query Model: {stats.get('query_model', 'N/A')}
            """
        )
    
    # Chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("📜 Conversation History")
        
        for i, exchange in enumerate(st.session_state.chat_history):
            with st.expander(f"Q{i+1}: {exchange['query'][:50]}..."):
                st.write(f"**Question:** {exchange['query']}")
                st.write(f"**Answer:** {exchange['answer']}")


if __name__ == "__main__":
    main()
