import streamlit as st
import os
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

# Configuration with improved error handling
def setup_rag_system():
    """Initialize all components with proper error handling"""
    try:
        # 1. Load embedding model
        with st.spinner("🔄 Memuat model embedding..."):
            embedding_model = HuggingFaceEmbeddings(
                model_name="Qwen/Qwen3-Embedding-0.6B",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        day-123-CNN/best_butterfly_classification_model_transfer_learning.h5
        # 2. Load vector store
        with st.spinner("📂 Memuat basis pengetahuan..."):
            vector_store = FAISS.load_local(
                "day-124-RAGSystem/history_faiss_index",
                embedding_model,
                allow_dangerous_deserialization=True
            )
        
        # 3. Initialize LLM
        with st.spinner("🧠 Memuat model bahasa..."):
            llm = ChatOpenAI(
                api_key=st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY")),
                base_url="https://api.groq.com/openai/v1",
                model="llama-3.3-70b-versatile",
                max_tokens=1024,
                temperature=0.3,
                request_timeout=30
            )
        
        # 4. Setup QA system
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
        
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": prompt_template},
            return_source_documents=True
        )
    
    except Exception as e:
        st.error(f"❌ Gagal memuat sistem: {str(e)}")
        st.stop()

def main():
    st.set_page_config(
        page_title="RAG Sejarah Indonesia",
        page_icon="📜",
        layout="wide"
    )
    
    # Initialize with progress tracking
    with st.status("🚀 Memulai sistem...", expanded=True) as status:
        st.write("Menginisialisasi komponen AI...")
        qa_system = setup_rag_system()
        status.update(label="✅ Sistem siap!", state="complete")
    
    # UI Components
    st.header("🔍 Sistem Tanya Jawab Sejarah Indonesia")
    
    with st.form("qa_form"):
        question = st.text_area(
            "Pertanyaan Anda:",
            placeholder="Contoh: Apa peran Soekarno dalam kemerdekaan?",
            height=150
        )
        
        if st.form_submit_button("🔎 Cari Jawaban", type="primary"):
            if not question.strip():
                st.warning("Mohon masukkan pertanyaan")
            else:
                with st.spinner("🔍 Mencari jawaban..."):
                    try:
                        start_time = time.time()
                        response = qa_system({"query": question})
                        elapsed = time.time() - start_time
                        
                        # Display results
                        st.subheader("📝 Jawaban")
                        st.markdown(f"""<div style='
                            background-color: #f8f9fa;
                            padding: 1.5rem;
                            border-radius: 0.5rem;
                            border-left: 4px solid #4e79a7;
                            margin-bottom: 1rem;
                        '>{response['result']}</div>""", unsafe_allow_html=True)
                        
                        # Show sources
                        if response['source_documents']:
                            with st.expander(f"📚 Sumber Referensi ({len(response['source_documents'])})"):
                                for doc in response['source_documents']:
                                    st.markdown(f"""
                                    **{doc.metadata['source']}**  
                                    *{doc.metadata.get('min_year', '')}-{doc.metadata.get('max_year', '')}*  
                                    `{doc.page_content[:200]}...`  
                                    """)
                        
                        st.caption(f"⏱️ Diproses dalam {elapsed:.2f} detik")
                    
                    except Exception as e:
                        st.error(f"Error saat memproses: {str(e)}")

if __name__ == "__main__":
    # Streamlit Cloud compatible entry point
    main()
