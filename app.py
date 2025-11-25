import streamlit as st
import os
from rag_core import initialize_rag_chain
from langchain_google_genai import GoogleGenerativeAI

# Page config
st.set_page_config(
    page_title="Edubia RAG System",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
    st.session_state.initialized = False

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    
    if not st.session_state.initialized:
        if st.button("Initialize RAG System", type="primary"):
            with st.spinner("Loading documents and initializing..."):
                try:
                    st.session_state.rag_chain = initialize_rag_chain()
                    st.session_state.initialized = True
                    st.success("✅ RAG System initialized!")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    else:
        st.success("✅ System Ready")
        if st.button("Reset Chat"):
            st.session_state.messages = []
            st.rerun()
    
    st.divider()
    st.markdown("### About")
    st.info("RAG System menggunakan dokumen lokal untuk menjawab pertanyaan. Jika jawaban tidak ditemukan, sistem akan menggunakan base model.")

# Main content
st.title("🤖 Edubia RAG System")
st.markdown("Tanyakan apapun tentang dokumen yang tersedia!")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 Sources"):
                for source in message["sources"]:
                    st.text(f"• {source}")

# Chat input
if prompt := st.chat_input("Ketik pertanyaan Anda..."):
    if not st.session_state.initialized:
        st.warning("⚠️ Please initialize the RAG system first from the sidebar!")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.rag_chain.invoke({"query": prompt})
                    answer = response.get('result', '')
                    sources = []
                    
                    # Check if NOT_FOUND, use base model
                    if "NOT_FOUND" in answer:
                        st.info("💡 Answer not found in documents, using base model...")
                        base_llm = GoogleGenerativeAI(
                            model="models/gemini-2.5-flash",
                            temperature=0.7,
                            google_api_key=os.getenv("GOOGLE_API_KEY")
                        )
                        answer = base_llm.invoke(prompt)
                        st.markdown(answer)
                    else:
                        st.markdown(answer)
                        
                        # Extract sources
                        if 'source_documents' in response:
                            for doc in response['source_documents']:
                                source = doc.metadata.get('source', 'Unknown')
                                sources.append(source)
                            sources = list(set(sources))
                            
                            if sources:
                                with st.expander("📚 Sources"):
                                    for source in sources:
                                        st.text(f"• {source}")
                    
                    # Add assistant message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                    
                except Exception as e:
                    st.error(f"❌ Error: {e}")
