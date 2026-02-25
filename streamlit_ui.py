"""
RAG Pipeline with Streamlit UI
Upload documents and ask questions using Ollama LLM with Qdrant vector storage
"""

import streamlit as st
from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag_pipeline import RAGPipeline
import config

# Page configuration
st.set_page_config(
    page_title="RAG Pipeline - Document Q&A",
    layout="wide"
)

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = VectorStore()
    st.session_state.rag_pipeline = RAGPipeline(st.session_state.vector_store)
    st.session_state.document_processor = DocumentProcessor()
    st.session_state.uploaded_files = []
    st.session_state.chat_history = []

# Title and description
st.title("RAG Pipeline - Document Q&A System")
st.markdown("""
Upload your documents and ask questions! The system uses **Ollama** for AI responses 
and **Qdrant** for intelligent document retrieval.
""")

# Sidebar for configuration and stats
with st.sidebar:
    st.header("Configuration")
    st.info(f"**Model:** {config.OLLAMA_MODEL}")
    st.info(f"**Embedding:** {config.EMBEDDING_MODEL}")
    
    st.header("Statistics")
    collection_info = st.session_state.vector_store.get_collection_info()
    if 'error' not in collection_info:
        st.metric("Documents in Database", collection_info.get('points_count', 0))
    else:
        st.warning("Unable to fetch collection info")
    
    st.header("Management")
    if st.button("Clear All Documents", type="secondary"):
        st.session_state.vector_store.delete_collection()
        st.session_state.uploaded_files = []
        st.session_state.chat_history = []
        st.success("All documents cleared!")
        st.rerun()

# Create tabs
tab1, tab2, tab3 = st.tabs(["Upload Documents", "Ask Questions", "Uploaded Files"])

# Tab 1: Upload Documents
with tab1:
    st.header("Upload Your Documents")
    st.markdown("Supported formats: **PDF**, **DOCX**, **TXT**")
    
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    if uploaded_files:
        if st.button("Process and Upload", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_chunks = 0
            for idx, uploaded_file in enumerate(uploaded_files):
                try:
                    status_text.text(f"Processing {uploaded_file.name}...")
                    
                    # Process document
                    chunks = st.session_state.document_processor.process_document(
                        uploaded_file, 
                        uploaded_file.name
                    )
                    
                    # Add to vector store
                    num_chunks = st.session_state.vector_store.add_documents(chunks)
                    total_chunks += num_chunks
                    
                    # Track uploaded file
                    if uploaded_file.name not in st.session_state.uploaded_files:
                        st.session_state.uploaded_files.append(uploaded_file.name)
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                    
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            
            status_text.empty()
            progress_bar.empty()
            st.success(f"Successfully processed {len(uploaded_files)} file(s) into {total_chunks} chunks!")
            st.rerun()

# Tab 2: Ask Questions
with tab2:
    st.header("Ask Questions About Your Documents")
    
    # Check if documents are uploaded
    if not st.session_state.uploaded_files:
        st.warning("Please upload documents first in the 'Upload Documents' tab.")
    else:
        # Display chat history
        st.subheader("Chat History")
        chat_container = st.container()
        
        with chat_container:
            for chat in st.session_state.chat_history:
                with st.chat_message("user"):
                    st.write(chat['question'])
                with st.chat_message("assistant"):
                    st.write(chat['answer'])
                    if chat.get('show_sources'):
                        with st.expander("View Sources"):
                            for i, ctx in enumerate(chat.get('context', []), 1):
                                st.markdown(f"**Source {i}:** {ctx['metadata'].get('filename', 'Unknown')}")
                                st.text(ctx['text'][:200] + "..." if len(ctx['text']) > 200 else ctx['text'])
                                st.markdown(f"*Relevance Score: {ctx['score']:.3f}*")
                                st.divider()
        
        # Question input
        st.divider()
        question = st.text_input(
            "Enter your question:",
            placeholder="What is this document about?",
            key="question_input"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            ask_button = st.button("Ask", type="primary", use_container_width=True)
        with col2:
            show_sources = st.checkbox("Show sources", value=True)
        
        if ask_button and question:
            with st.spinner("Thinking..."):
                # Get response from RAG pipeline
                result = st.session_state.rag_pipeline.query(question)
                
                if result['success']:
                    # Add to chat history
                    st.session_state.chat_history.append({
                        'question': question,
                        'answer': result['answer'],
                        'context': result['context'],
                        'show_sources': show_sources
                    })
                    st.rerun()
                else:
                    st.error(f"Error: {result.get('error', 'Unknown error')}")

# Tab 3: Uploaded Files
with tab3:
    st.header("Uploaded Files")
    
    if st.session_state.uploaded_files:
        st.markdown(f"**Total files uploaded:** {len(st.session_state.uploaded_files)}")
        
        for idx, filename in enumerate(st.session_state.uploaded_files, 1):
            st.markdown(f"{idx}. {filename}")
    else:
        st.info("No files uploaded yet. Go to the 'Upload Documents' tab to get started!")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
    Powered by Ollama | Qdrant | Streamlit
</div>
""", unsafe_allow_html=True)
