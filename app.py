import os
import tempfile

import chromadb
import ollama
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder, SentenceTransformer
from streamlit.runtime.uploaded_file_manager import UploadedFile

# ---------------------------------------------------------------------------
# Configuration — read from .env with sensible defaults
# ---------------------------------------------------------------------------
load_dotenv()

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large:latest")
LLM_MODEL: str = os.getenv("LLM_MODEL", "llama3.2:3b")
USE_OLLAMA_EMBEDDING: bool = os.getenv("USE_OLLAMA_EMBEDDING", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Disable ChromaDB telemetry to suppress PostHog warnings
# ---------------------------------------------------------------------------
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# ---------------------------------------------------------------------------
# Cached model loaders (only loaded once per session)
# ---------------------------------------------------------------------------

@st.cache_resource
def get_sentence_transformer() -> SentenceTransformer:
    """Fallback embedding model (only used if Ollama is unavailable)."""
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def get_cross_encoder() -> CrossEncoder:
    """Cached cross-encoder for re-ranking."""
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# ---------------------------------------------------------------------------
# Embedding function: Ollama first, local sentence-transformer fallback
# ---------------------------------------------------------------------------

class AppEmbeddingFunction:
    """Fast local embeddings by default. Set USE_OLLAMA_EMBEDDING=true in .env to use Ollama instead."""

    def __call__(self, input: list[str]) -> list[list[float]]:
        if USE_OLLAMA_EMBEDDING:
            try:
                response = ollama.embed(model=EMBEDDING_MODEL, input=input)
                return response["embeddings"]
            except Exception as exc:
                st.warning(
                    f"⚠️ Ollama embedding failed ({exc}). Falling back to local model."
                )
        # Fast local path (primary when USE_OLLAMA_EMBEDDING=false)
        return get_sentence_transformer().encode(input).tolist()


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

system_prompt = """
You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

context will be passed as "Context:"
user question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

Format your response as follows:
1. Use clear, concise language.
2. Organize your answer into paragraphs for readability.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.
5. Ensure proper grammar, punctuation, and spelling throughout your answer.

Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
"""


# ---------------------------------------------------------------------------
# Document processing
# ---------------------------------------------------------------------------

def process_document(uploaded_file: UploadedFile) -> list[Document]:
    """Saves the uploaded PDF to a temp file, loads it, and returns text chunks."""
    temp_file = tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False)
    try:
        temp_file.write(uploaded_file.read())
        temp_file.close()
        loader = PyMuPDFLoader(temp_file.name)
        docs = loader.load()
    finally:
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )
    return text_splitter.split_documents(docs)


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

def get_vector_collection() -> chromadb.Collection:
    """Returns (or creates) the ChromaDB collection with telemetry disabled."""
    settings = chromadb.config.Settings(anonymized_telemetry=False)
    chroma_client = chromadb.PersistentClient(
        path="./demo-rag-chroma", settings=settings
    )
    return chroma_client.get_or_create_collection(
        name="rag_app",
        embedding_function=AppEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"},
    )


def add_to_vector_collection(all_splits: list[Document], file_name: str) -> None:
    """Adds non-empty document chunks to ChromaDB."""
    if not all_splits:
        st.error("No content to add to the vector store!")
        return

    collection = get_vector_collection()
    documents, metadatas, ids = [], [], []

    for idx, split in enumerate(all_splits):
        if not split.page_content.strip():
            continue
        documents.append(split.page_content)
        metadatas.append(split.metadata)
        ids.append(f"{file_name}_{idx}")

    if not documents:
        st.error("No valid text found in the document chunks.")
        return

    try:
        collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
        st.success("✅ Data added to the vector store!")
    except Exception as e:
        st.error(f"Error adding to vector store: {e}")


def query_collection(prompt: str, n_results: int = 5) -> dict:
    """Queries the vector collection and returns matching documents."""
    collection = get_vector_collection()
    return collection.query(query_texts=[prompt], n_results=n_results)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_llm(context: str, prompt: str):
    """Streams a response from the Ollama LLM."""
    response = ollama.chat(
        model=LLM_MODEL,
        stream=True,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context: {context}, Question: {prompt}"},
        ],
    )
    for chunk in response:
        if chunk["done"] is False:
            yield chunk["message"]["content"]
        else:
            break


# ---------------------------------------------------------------------------
# Re-ranking
# ---------------------------------------------------------------------------

def re_rank_cross_encoders(documents: list[str]) -> tuple[str, list[int]]:
    """Re-ranks retrieved documents with a cross-encoder and returns the top-3."""
    relevant_text = ""
    relevant_text_ids = []
    encoder_model = get_cross_encoder()
    ranks = encoder_model.rank(prompt, documents, top_k=3)
    for rank in ranks:
        relevant_text += documents[rank["corpus_id"]]
        relevant_text_ids.append(rank["corpus_id"])
    return relevant_text, relevant_text_ids


# ---------------------------------------------------------------------------
# Startup check
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def check_ollama() -> None:
    """Verifies Ollama is reachable and that the LLM model is installed."""
    required_models = {LLM_MODEL}
    if USE_OLLAMA_EMBEDDING:
        required_models.add(EMBEDDING_MODEL)
    try:
        installed = {m["name"] for m in ollama.list()["models"]}
        missing = required_models - installed
        if missing:
            for model in sorted(missing):
                st.error(
                    f"❌ Model **{model}** is not installed in Ollama. "
                    f"Fix it by running:\n```\nollama pull {model}\n```"
                )
    except Exception as exc:
        st.error(
            f"❌ Cannot reach Ollama at `{OLLAMA_BASE_URL}`. "
            f"Make sure Ollama is running. Error: {exc}"
        )


# ---------------------------------------------------------------------------
# Main Streamlit app
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    st.set_page_config(page_title="RAG Question Answer")
    check_ollama()

    # Sidebar — document upload
    with st.sidebar:
        uploaded_file = st.file_uploader(
            "**📑 Upload PDF files for QnA**", type=["pdf"], accept_multiple_files=False
        )
        process = st.button("⚡️ Process")
        if uploaded_file and process:
            normalize_uploaded_file_name = uploaded_file.name.translate(
                str.maketrans({"-": "_", ".": "_", " ": "_"})
            )
            all_splits = process_document(uploaded_file)
            add_to_vector_collection(all_splits, normalize_uploaded_file_name)

    # Main area — Q&A
    st.header("🗣️ RAG Question Answer")
    prompt = st.text_area("**Ask a question related to your document:**")
    ask = st.button("🔥 Ask")

    if ask and prompt:
        results = query_collection(prompt)
        context = results.get("documents")[0]
        relevant_text, relevant_text_ids = re_rank_cross_encoders(context)
        response = call_llm(context=relevant_text, prompt=prompt)
        st.write_stream(response)

        with st.expander("See retrieved documents"):
            st.write(results)

        with st.expander("See most relevant document ids"):
            st.write(relevant_text_ids)
            st.write(relevant_text)
