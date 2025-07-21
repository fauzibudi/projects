import streamlit as st
import os
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

# Konfigurasi aplikasi
st.set_page_config(
    page_title="RAG Sejarah Indonesia",
    page_icon="📜",
    layout="wide"
)

# Fungsi untuk inisialisasi model dengan caching
@st.cache_resource(show_spinner=False)
def load_models():
    """Muat model embedding dan LLM"""
    with st.spinner("Memuat model AI..."):
        # Model embedding
        embedding_model = HuggingFaceEmbeddings(
            model_name="Qwen/Qwen3-Embedding-0.6B",
            model_kwargs={'device': 'cuda' if st.secrets["USE_GPU"] else 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # LLM
        llm = ChatOpenAI(
            api_key=st.secrets["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1",
            model="llama-3.3-70b-versatile",
            max_tokens=1024,
            temperature=0.3
        )
        
        return embedding_model, llm

# Fungsi untuk memuat vector store
@st.cache_resource(show_spinner=False)
def load_vector_store():
    """Muat basis pengetahuan vektor"""
    with st.spinner("Memuat basis pengetahuan sejarah..."):
        embedding_model, _ = load_models()
        vector_store = FAISS.load_local(
            "history_faiss_index",
            embedding_model,
            allow_dangerous_deserialization=True
        )
        return vector_store

# Fungsi untuk inisialisasi sistem QA
@st.cache_resource(show_spinner=False)
def initialize_qa_system():
    """Inisialisasi sistem tanya jawab"""
    embedding_model, llm = load_models()
    vector_store = load_vector_store()
    
    # Template prompt
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        Anda adalah asisten ahli sejarah Indonesia. Jawablah pertanyaan HANYA berdasarkan konteks di bawah ini.
        Jika tidak ada informasi yang relevan, katakan 'Data tidak ditemukan'.
        
        Konteks:
        {context}
        
        Pertanyaan: {question}
        
        Jawaban:
        """
    )
    
    # Retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    # QA System
    qa_system = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=True
    )
    
    return qa_system

# Fungsi untuk menampilkan sumber referensi
def display_sources(source_docs):
    """Tampilkan sumber referensi dengan gaya yang baik"""
    if not source_docs:
        return
        
    with st.expander("🔍 **Sumber Referensi**"):
        for i, doc in enumerate(source_docs):
            col1, col2 = st.columns([0.9, 0.1])
            with col1:
                st.markdown(f"**{i+1}. {doc.metadata['source']}**")
                st.caption(f"URL: {doc.metadata.get('url', 'Tidak tersedia')}")
                st.caption(f"Rentang Tahun: {doc.metadata.get('min_year', '')}-{doc.metadata.get('max_year', '')}")
                
                # Tampilkan preview teks dengan highlight
                preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                st.markdown(f"<div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px;'>{preview}</div>", 
                            unsafe_allow_html=True)
            
            with col2:
                if doc.metadata.get('url'):
                    st.link_button("🌐", doc.metadata['url'], use_container_width=True)

# Antarmuka Utama
def main():
    # Sidebar
    with st.sidebar:
        st.title("📚 RAG Sejarah Indonesia")
        st.markdown("""
        **Sistem Tanya Jawab Sejarah Indonesia**  
        Didukung oleh:
        - Model: Llama-3-70B
        - Embedding: Qwen-0.6B
        - Basis Pengetahuan: Wikipedia Indonesia
        """)
        
        st.divider()
        st.markdown("### Contoh Pertanyaan:")
        st.code("Apa peran Soekarno dalam kemerdekaan?")
        st.code("Bagaimana Pertempuran Surabaya terjadi?")
        st.code("Apa itu Bandung Lautan Api?")
        st.code("Jelaskan Konferensi Meja Bundar")
        
        st.divider()
        st.info("""
        ⚠️ **Catatan:**  
        Jawaban bersumber dari basis pengetahuan sejarah Indonesia. 
        Untuk informasi terkini, selalu verifikasi dengan sumber terpercaya.
        """)
    
    # Header utama
    st.header("🔍 Sistem Tanya Jawab Sejarah Indonesia")
    st.markdown("Tanyakan apapun tentang sejarah Indonesia.")
    
    # Inisialisasi sistem
    qa_system = initialize_qa_system()
    
    # Form input pengguna
    with st.form("question_form"):
        question = st.text_area(
            "Masukkan pertanyaan Anda tentang sejarah Indonesia:",
            placeholder="Contoh: Apa peran Soekarno dalam kemerdekaan Indonesia?",
            height=150
        )
        
        col1, col2 = st.columns([0.1, 0.9])
        with col1:
            submit_btn = st.form_submit_button("🔎 Cari Jawaban", use_container_width=True)
        with col2:
            st.markdown("<div style='height: 36px;'></div>", unsafe_allow_html=True)
            st.caption("Sistem mungkin membutuhkan waktu 10-20 detik untuk merespons")
    
    # Proses pertanyaan
    if submit_btn and question:
        with st.spinner("Mencari informasi dalam basis pengetahuan..."):
            start_time = time.time()
            response = qa_system({"query": question})
            processing_time = time.time() - start_time
            
        # Tampilkan jawaban
        st.subheader("📝 Jawaban")
        st.markdown(f"<div style='background-color: #e6f7ff; padding: 20px; border-radius: 10px;'>{response['result']}</div>", 
                    unsafe_allow_html=True)
        
        # Tampilkan waktu pemrosesan
        st.caption(f"⏱️ Waktu pemrosesan: {processing_time:.2f} detik | {len(response['source_documents'])} dokumen referensi")
        
        # Tampilkan sumber referensi
        if response['source_documents']:
            display_sources(response['source_documents'])
        else:
            st.warning("Tidak ditemukan dokumen referensi yang relevan")
    
    elif submit_btn and not question:
        st.warning("Silakan masukkan pertanyaan terlebih dahulu")

if __name__ == "__main__":
    main()