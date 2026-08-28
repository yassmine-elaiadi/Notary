from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from backend.online_rag import get_online_legal_sources

CHROMA_FOLDER = "chroma_db"


embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore = Chroma(
    persist_directory=CHROMA_FOLDER,
    embedding_function=embeddings,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

llm = ChatOllama(
    model="mistral",
    mirostat=2,
    mirostat_tau=5,
    mirostat_eta=0.1,
    num_predict=600,
    stop=["\nQuestion", "\nType: Source", "\nSources:"],
)

prompt = ChatPromptTemplate.from_template("""
Tu es un assistant juridique spécialisé dans le droit notarial allemand.

Réponds toujours en français.
Utilise uniquement les sources fournies.
Ne donne pas de conseil juridique personnalisé.
Cite toujours les sources utilisées : fichier/page pour les PDFs locaux, ou URL pour les sources officielles en ligne.
Si les sources ne permettent pas de répondre avec certitude, dis :
"Les sources fournies ne permettent pas de répondre avec certitude."

Sources:
{context}

Question (réponds uniquement en français, même si les sources sont en allemand ou en anglais):
{question}
""")

chat_history = []


def format_local_source(doc):
    if doc.metadata.get("source_type") == "official_xml":
        return (
            f"Type: Loi officielle allemande\n"
            f"Loi: {doc.metadata.get('law')}\n"
            f"Section: {doc.metadata.get('section')}\n"
            f"URL: {doc.metadata.get('url')}\n"
            f"Texte: {doc.page_content}"
        )

    return (
        f"Type: Document local (PDF)\n"
        f"Loi: {doc.metadata.get('law')}\n"
        f"Langue: {doc.metadata.get('language')}\n"
        f"Fichier: {doc.metadata.get('source_file')}\n"
        f"Page: {doc.metadata.get('page')}\n"
        f"Texte: {doc.page_content}"
    )


def ask(question):

    docs = retriever.invoke(question)

    local_context = "\n\n".join(format_local_source(doc) for doc in docs)

    online_sources = get_online_legal_sources(question, max_results=3)

    online_context = "\n\n".join(
        [
            f"Type: Source officielle en ligne\n"
            f"Titre: {source['title']}\n"
            f"URL: {source['url']}\n"
            f"Texte: {source['text']}"
            for source in online_sources
        ]
    )

    context = local_context + "\n\n" + online_context

    history_text = "\n".join(chat_history[-6:])

    messages = prompt.format_messages(
        context=context,
        question=f"""
Historique de conversation:
{history_text}

Nouvelle question:
{question}
"""
    )

    response = llm.invoke(messages)

    chat_history.append(f"Utilisateur: {question}")
    chat_history.append(f"Assistant: {response.content}")

    print("\n=== RÉPONSE ===\n")
    print(response.content)

    print("\n=== SOURCES LOCALES UTILISÉES ===\n")
    for doc in docs:
        if doc.metadata.get("source_type") == "official_xml":
            print(f"- {doc.metadata.get('law')} {doc.metadata.get('section')} | {doc.metadata.get('url')}")
        else:
            print(
                f"- {doc.metadata.get('law')} | "
                f"{doc.metadata.get('source_file')} | "
                f"page {doc.metadata.get('page')}"
            )

    print("\n=== SOURCES EN LIGNE UTILISÉES ===\n")
    for source in online_sources:
        print(f"- {source['title']} | {source['url']}")

while True:
    question = input("\nPosez votre question juridique en français : ")

    if question.lower() in ["quit", "exit", "q"]:
        break

    ask(question)