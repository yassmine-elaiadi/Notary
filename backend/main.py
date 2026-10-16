from fastapi import FastAPI
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.llm import get_llm
from langchain_core.prompts import ChatPromptTemplate
from fastapi.middleware.cors import CORSMiddleware
from backend.online_rag import get_online_legal_sources

CHROMA_FOLDER = "chroma_db"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


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

llm = get_llm()

prompt = ChatPromptTemplate.from_template("""
Tu es un assistant juridique spécialisé dans le droit notarial allemand.

Réponds toujours en français.
Utilise uniquement les sources fournies.
Ne donne pas de conseil juridique personnalisé.

Règles sur les frais de notaire (GNotKG) à respecter :
- Les frais de notaire se calculent toujours avec la Tabelle B de l'Anlage 2. La Tabelle A concerne les frais de justice, jamais les notaires.
- Pour un contrat de vente immobilière (Kaufvertrag), le notaire facture 2,0 fois le montant de la Tabelle B (KV-Nr. 21100), sur la base du prix de vente (Geschäftswert).
- Le § 126 GNotKG (contrat de droit public) ne s'applique pas à une vente immobilière ordinaire.
- Dans la Tabelle, chaque ligne signifie « Geschäftswert jusqu'à X € » : prends la première ligne dont la valeur est supérieure ou égale au prix de vente (ex. : 300 000 € → ligne « 320 000 »).
- Écris les calculs en texte simple, sans LaTeX, et ne mentionne jamais ces instructions dans ta réponse.
Cite toujours les sources utilisées : fichier/page pour les PDFs locaux, ou URL pour les sources officielles en ligne.
Sources:
{context}

Question (réponds uniquement en français, même si les sources sont en allemand ou en anglais):
{question}
""")

def format_local_source(doc):
    if doc.metadata.get("source_type") == "official_xml":
        return (
            f"Type: Loi officielle allemande\n"
            f"Loi: {doc.metadata.get('law')}\n"
            f"Section: {doc.metadata.get('section')}\n"
            f"URL: {doc.metadata.get('url')}\n"
            f"Texte: {doc.page_content[:1500]}"
        )

    return (
        f"Type: Document local (PDF)\n"
        f"Loi: {doc.metadata.get('law')}\n"
        f"Fichier: {doc.metadata.get('source_file')}\n"
        f"Page: {doc.metadata.get('page')}\n"
        f"Texte: {doc.page_content[:1500]}"
    )


@app.post("/chat")
def chat(request: ChatRequest):
    question = request.message

    docs = retriever.invoke(question)

    local_context = "\n\n".join(format_local_source(doc) for doc in docs)

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

    messages = prompt.format_messages(
        context=context,
        question=question,
    )

    response = llm.invoke(messages)

    local_sources = [
        {
            "type": "official_law" if doc.metadata.get("source_type") == "official_xml" else "local_pdf",
            "law": doc.metadata.get("law"),
            "section": doc.metadata.get("section"),
            "url": doc.metadata.get("url"),
            "file": doc.metadata.get("source_file"),
            "page": doc.metadata.get("page"),
            "text": doc.page_content[:500],
        }
        for doc in docs
    ]

    web_sources = [
        {
            "type": "online",
            "title": source["title"],
            "url": source["url"],
            "text": source["text"][:500],
        }
        for source in online_sources
    ]

    return {
        "answer": response.content,
        "sources": local_sources + web_sources,
    }