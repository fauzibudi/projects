import os
import asyncio
import streamlit as st
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import time

os.environ['STREAMLIT_SERVER_ENABLE_FILE_WATCHER'] = 'false'

def ensure_event_loop():
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

@st.cache_resource(show_spinner=False)
def load_rag_system():
    """Muat semua komponen RAG dengan error handling"""
    try:
        base_path = Path(__file__).parent
        faiss_path = base_path / "history_faiss_index"
        
        if not (faiss_path / "index.faiss").exists():
            st.error(f"File indeks tidak ditemukan di: {faiss_path}")
            st.stop()

        with st.spinner("Memuat model embedding..."):
            embedding_model = HuggingFaceEmbeddings(
                model_name="Qwen/Qwen3-Embedding-0.6B",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )

        with st.spinner("Memuat basis pengetahuan..."):
            vector_store = FAISS.load_local(
                str(faiss_path),
                embedding_model,
                allow_dangerous_deserialization=True
            )

        with st.spinner("Menyiapkan model bahasa..."):
            llm = ChatOpenAI(
                api_key=st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY")),
                base_url="https://api.groq.com/openai/v1",
                model="llama-3.3-70b-versatile",
                max_tokens=1024,
                temperature=0.3,
                request_timeout=30  
            )

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
        st.error(f"Gagal memuat sistem: {str(e)}")
        st.stop()

def main():
    st.set_page_config(
        page_title="RAG Sejarah Indonesia",
        page_icon="📜",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    with st.sidebar:
        st.title("📚 RAG Sejarah Indonesia")
        st.markdown("""
        **Sistem Tanya Jawab Otomatis**  
        Didukung oleh:
        - Llama-3-70B
        - Qwen Embeddings
        - FAISS Vector Store
        """)
        st.divider()
        st.markdown("### Contoh Pertanyaan:")
        st.code("Apa peran Soekarno dalam kemerdekaan?")
        st.code("Jelaskan Pertempuran Surabaya 1945")
        st.code("Apa itu Konferensi Meja Bundar?")
        st.divider()
        st.info("ℹ️ Untuk informasi lebih lanjut, hubungi: fbwmalang123@gmail.com")

    st.header("🔍 Sistem Tanya Jawab Sejarah Indonesia")
    st.caption("Tanyakan apapun tentang sejarah Indonesia!")

    qa_system = load_rag_system()

    with st.form("qa_form"):
        question = st.text_area(
            "Masukkan pertanyaan Anda:",
            placeholder="Contoh: Siapa yang memproklamasikan kemerdekaan Indonesia?",
            height=150
        )
        
        submitted = st.form_submit_button("🔎 Cari Jawaban", type="primary")

    if submitted and question:
        with st.spinner("Mencari jawaban..."):
            try:
                start_time = time.time()
                response = qa_system({"query": question})
                elapsed = time.time() - start_time

                st.subheader("📝 Jawaban")
                st.markdown(f"""
                <div style='
                    padding: 1.5rem;
                    border-radius: 0.5rem;
                    border-left: 4px solid #4e79a7;
                    margin-bottom: 1rem;
                '>
                {response['result']}
                </div>
                """, unsafe_allow_html=True)
                
                st.caption(f"⏱️ Diproses dalam {elapsed:.2f} detik")

            except Exception as e:
                st.error(f"Terjadi kesalahan: {str(e)}")

if __name__ == "__main__":
    loop = ensure_event_loop()
    
    try:
        main()
    finally:
        loop.close()
