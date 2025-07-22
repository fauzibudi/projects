import streamlit as st
import os
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # Import yang diperbarui
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

# Konfigurasi untuk mengurangi beban startup
@st.cache_resource(ttl=3600, show_spinner="Memuat model...") 
def load_components():
    """Muat semua komponen sekaligus untuk efisiensi"""
    # Model embedding
    embedding_model = HuggingFaceEmbeddings(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # LLM - Gunakan config untuk timeout
    llm = ChatOpenAI(
        api_key=st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY")),
        base_url="https://api.groq.com/openai/v1",
        model="llama-3.3-70b-versatile",
        max_tokens=1024,
        temperature=0.3,
        request_timeout=60  # Tambah timeout
    )
    
    # Vector store
    vector_store = FAISS.load_local(
        "history_faiss_index",
        embedding_model,
        allow_dangerous_deserialization=True
    )
    
    return embedding_model, llm, vector_store

def main():
    st.set_page_config(
        page_title="RAG Sejarah Indonesia",
        page_icon="📜",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Tampilkan loading state awal
    with st.spinner("Menyiapkan sistem sejarah Indonesia..."):
        try:
            embedding_model, llm, vector_store = load_components()
            
            # Setup QA system setelah komponen siap
            prompt_template = PromptTemplate(
                input_variables=["context", "question"],
                template="""..."""  # Template sama seperti sebelumnya
            )
            
            qa_system = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(search_kwargs={"k": 3}),  # Kurangi jumlah dokumen
                chain_type_kwargs={"prompt": prompt_template},
                return_source_documents=True
            )
            
        except Exception as e:
            st.error(f"Gagal memuat sistem: {str(e)}")
            st.stop()
    
    # ... (bagian antarmuka tetap sama)

if __name__ == "__main__":
    # Konfigurasi untuk handling timeout
    import signal
    signal.signal(signal.SIGALRM, lambda signum, frame: st.error("Timeout saat memuat komponen"))
    signal.alarm(120)  # Timeout 2 menit
    
    main()
    signal.alarm(0)  # Matikan alarm
