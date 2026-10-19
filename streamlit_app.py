import streamlit as st
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.llm import get_llm
from src.prompts import NOTARY_PROMPT, RETRIEVER_K, format_local_source
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
from src.paragraph_extractor import extract_paragraph
from backend.online_rag import get_online_legal_sources

CHROMA_FOLDER = "chroma_db"


st.set_page_config(layout="wide")
with open("styles.css", "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div class="hero-title">⚖️ Assistant juridique notarial allemand</div>
    <div class="hero-subtitle">
        Réponses en français à partir de sources juridiques allemandes officielles.
    </div>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_rag():
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma(
        persist_directory=CHROMA_FOLDER,
        embedding_function=embeddings,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})

    llm = get_llm()

    return retriever, llm, vectorstore

def add_uploaded_pdf(uploaded_file, vectorstore):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    loader = PyPDFLoader(temp_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
    )

    chunks = splitter.split_documents(documents)

    for chunk in chunks:
        chunk.metadata["law"] = "Uploaded Document"
        chunk.metadata["language"] = "unknown"
        chunk.metadata["source_file"] = uploaded_file.name
        chunk.metadata["source_type"] = "user_upload"

    vectorstore.add_documents(chunks)

retriever, llm, vectorstore = load_rag()

prompt = NOTARY_PROMPT


if "messages" not in st.session_state:
    st.session_state.messages = []

if "sources" not in st.session_state:
    st.session_state.sources = []

if "online_sources" not in st.session_state:
    st.session_state.online_sources = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


question = st.chat_input("Posez votre question juridique en français...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.write(question)
    simple_greetings = ["hi", "hello", "hey", "salut", "bonjour", "salam"]

    if question.lower().strip() in simple_greetings:
        answer = "Bonjour 👋 Comment puis-je vous aider avec une question juridique notariale allemande ?"

        with st.chat_message("assistant"):
            st.write(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.stop()
    docs = retriever.invoke(question)

    local_context = "\n\n".join(format_local_source(doc) for doc in docs)

    with st.spinner("Recherche de sources officielles en ligne..."):
        online_sources = get_online_legal_sources(question, max_results=3)

    online_context = "\n\n".join(
        [
            f"Type: Source officielle en ligne\n"
            f"Titre: {source['title']}\n"
            f"URL: {source['url']}\n"
            f"Texte: {source['text'][:2000]}"
            for source in online_sources
        ]
    )

    context = local_context + "\n\n" + online_context

    history = "\n".join(
        [
            f"{m['role']}: {m['content']}"
            for m in st.session_state.messages[-6:]
        ]
    )

    messages = prompt.format_messages(
        history=history,
        context=context,
        question=question,
    )

    with st.chat_message("assistant"):
        with st.spinner("Recherche dans les lois et génération de la réponse..."):
            response = llm.invoke(messages)
            answer = response.content
            st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

    st.session_state.sources = docs
    st.session_state.online_sources = online_sources


with st.sidebar:
    uploaded_file = st.file_uploader(
        "Télécharger un document PDF",
        type=["pdf"]
    )

    if uploaded_file:
        add_uploaded_pdf(uploaded_file, vectorstore)
        st.success("Document ajouté à la base RAG.")

    st.header("Sources utilisées")

    if st.session_state.sources:
        for i, doc in enumerate(st.session_state.sources, start=1):
            st.markdown(f"**Source {i}**")

            if doc.metadata.get("source_type") == "official_xml":
                st.write("Loi:", doc.metadata.get("law"))
                st.write("Section:", doc.metadata.get("section"))
                st.markdown(f"[{doc.metadata.get('url')}]({doc.metadata.get('url')})")
            else:
                st.write("Loi:", doc.metadata.get("law"))
                st.write("Fichier:", doc.metadata.get("source_file"))
                st.write("Paragraphe:", extract_paragraph(doc.page_content))

            with st.expander("Voir le texte juridique complet"):
                st.write(doc.page_content)

            st.divider()
    else:
        st.write("Aucune source encore.")

    st.header("Sources officielles en ligne")

    if st.session_state.online_sources:
        for i, source in enumerate(st.session_state.online_sources, start=1):
            st.markdown(f"**Source {i}**")
            st.write(source["title"])
            st.markdown(f"[{source['url']}]({source['url']})")

            with st.expander("Voir le texte complet"):
                st.write(source["text"])

            st.divider()
    else:
        st.write("Aucune source en ligne encore.")
