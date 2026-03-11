import streamlit as st
import os
from fpdf import FPDF

from src.ingestion.loader import PDFLoader
from src.processing.chunking import ChunkProcessor
from src.embeddings.embedding_manager import EmbeddingManager
from src.vector_store.faiss_store import FAISSVectorStore
from src.retrieval.retriever import Retriever
from src.llm.llm_handler import GroqLLM

st.set_page_config(layout="wide")

# CHANGE: Added CSS to handle the "Notes" toggle visibility and fixed positioning
st.markdown("""
    <style>
    .main .block-container {
        padding-bottom: 100px;
    }
    /* Simple styling to make the notes area stand out when active */
    .notes-box {
        background-color: #f9f9f9;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------
# SESSION STATE
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome buddy! Let's dive into your research."}
    ]

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "notes" not in st.session_state:
    st.session_state.notes = ""

# CHANGE: Track whether notes are visible
if "show_notes" not in st.session_state:
    st.session_state.show_notes = False

# -----------------------------
# TOP NAVIGATION / HEADER
# -----------------------------
# CHANGE: Added a top bar with a "Notes" button on the right
col1, col2 = st.columns([9, 1])
with col1:
    st.title("Research Copilot")
with col2:
    if st.button("📝 Notes"):
        st.session_state.show_notes = not st.session_state.show_notes

# -----------------------------
# LEFT SIDEBAR (RESOURCES)
# -----------------------------
# CHANGE: Moved file upload and processing status to the left sidebar
with st.sidebar:
    st.header("Resources")
    uploaded_files = st.file_uploader(
        "Upload Research Papers",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files and st.session_state.retriever is None:
        os.makedirs("data/papers", exist_ok=True)
        for file in uploaded_files:
            path = os.path.join("data/papers", file.name)
            with open(path, "wb") as f:
                f.write(file.getbuffer())

        with st.status("Processing papers..."):
            loader = PDFLoader("data/papers")
            documents = loader.load_pdfs()
            chunker = ChunkProcessor()
            chunks = chunker.split_documents(documents)
            embedding_manager = EmbeddingManager()
            embeddings = embedding_manager.generate_embeddings(chunks)
            vector_store = FAISSVectorStore(len(embeddings[0]))
            vector_store.add_embeddings(embeddings, chunks)
            st.session_state.retriever = Retriever(vector_store, embedding_manager.model)
            st.success("Analysis Complete!")

# -----------------------------
# MAIN LAYOUT (CHAT & OPTIONAL NOTES)
# -----------------------------
# CHANGE: Logic to dynamically adjust column widths based on notes visibility
if st.session_state.show_notes:
    chat_col, notes_col = st.columns([2, 1])
else:
    chat_col = st.container()

with chat_col:
    chat_holder = st.container()
    with chat_holder:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

# CHANGE: The notes panel now only renders if 'show_notes' is True
if st.session_state.show_notes:
    with notes_col:
        st.subheader("Research Notes")
        st.session_state.notes = st.text_area(
            "Write insights here",
            value=st.session_state.notes,
            height=500
        )
        if st.button("Download PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            for line in st.session_state.notes.split("\n"):
                pdf.cell(0, 10, txt=line, ln=True)
            pdf.output("research_notes.pdf")
            with open("research_notes.pdf", "rb") as f:
                st.download_button("Download PDF", f, file_name="research_notes.pdf")

# -----------------------------
# CHAT INPUT
# -----------------------------
prompt = st.chat_input("Ask anything about your research...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with chat_holder:
        with st.chat_message("user"):
            st.write(prompt)

        if st.session_state.retriever is None:
            answer = "Upload papers in the left sidebar first!"
        else:
            docs = st.session_state.retriever.retrieve(prompt)
            context = "\n\n".join([doc.page_content for doc in docs])
            llm = GroqLLM()
            answer = llm.generate_answer(prompt, context)

        with st.chat_message("assistant"):
            st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})