from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


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
    temperature=0,
)

prompt = ChatPromptTemplate.from_template("""
Tu es un assistant juridique spécialisé dans le droit notarial allemand.

Réponds toujours en français.
Utilise uniquement les sources fournies.
Ne donne pas de conseil juridique personnalisé.
Cite toujours la loi, la page et le fichier source si disponibles.
Si les sources ne permettent pas de répondre avec certitude, dis :
"Les sources fournies ne permettent pas de répondre avec certitude."

Sources:
{context}

Question:
{question}
""")

chat_history = []


def ask(question):
    docs = retriever.invoke(question)

    context = "\n\n".join(
        [
            f"Loi: {doc.metadata.get('law')}\n"
            f"Langue: {doc.metadata.get('language')}\n"
            f"Fichier: {doc.metadata.get('source_file')}\n"
            f"Page: {doc.metadata.get('page')}\n"
            f"Texte: {doc.page_content}"
            for doc in docs
        ]
    )

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

    print("\n=== SOURCES UTILISÉES ===\n")
    for doc in docs:
        print(
            f"- {doc.metadata.get('law')} | "
            f"{doc.metadata.get('source_file')} | "
            f"page {doc.metadata.get('page')}"
        )


while True:
    question = input("\nPosez votre question juridique en français : ")

    if question.lower() in ["quit", "exit", "q"]:
        break

    ask(question)