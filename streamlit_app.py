import streamlit as st
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
from src.paragraph_extractor import extract_paragraph

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

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatOllama(
        model="phi3",
        temperature=0,
    )

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

    vectorstore.add_documents(chunks)

retriever, llm, vectorstore = load_rag()

prompt = ChatPromptTemplate.from_template("""
Tu es un assistant juridique spécialisé dans le droit notarial allemand.

Réponds toujours en français.
Utilise uniquement les sources fournies.
Ne donne pas de conseil juridique personnalisé.
Cite toujours la loi, la page et le fichier source si disponibles.

Ajoute toujours cette phrase à la fin :
"Cette réponse est fournie à titre informatif uniquement et ne constitue pas un conseil juridique officiel."

Si les sources ne permettent pas de répondre avec certitude, dis :
"Les sources fournies ne permettent pas de répondre avec certitude."

Historique:
{history}

Sources:
{context}

Question:
{question}
""")


if "messages" not in st.session_state:
    st.session_state.messages = []

if "sources" not in st.session_state:
    st.session_state.sources = []


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

    context = "\n\n".join(
        [
            f"Loi: {doc.metadata.get('law')}\n"
            f"Langue: {doc.metadata.get('language')}\n"
            f"Fichier: {doc.metadata.get('source_file')}\n"
            f"Page: {doc.metadata.get('page')}\n"
            f"Texte: {doc.page_content[:1500]}"
            for doc in docs
        ]
    )

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
            st.write("Loi:", doc.metadata.get("law"))
            st.write("Fichier:", doc.metadata.get("source_file"))
            st.write("Paragraphe:", extract_paragraph(doc.page_content))

            with st.expander("Voir le texte juridique complet"):
                st.write(doc.page_content)

            st.divider()
    else:
        st.write("Aucune source encore.")
