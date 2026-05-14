from fastapi import FastAPI
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from fastapi.middleware.cors import CORSMiddleware


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

llm = ChatOllama(
    model="phi3",
    temperature=0,
)

prompt = ChatPromptTemplate.from_template("""
Tu es un assistant juridique spécialisé dans le droit notarial allemand.

Réponds toujours en français.
Utilise uniquement les sources fournies.
Ne donne pas de conseil juridique personnalisé.
Cite toujours la loi, la page et le fichier source si disponibles.

Sources:
{context}

Question:
{question}
""")


@app.post("/chat")
def chat(request: ChatRequest):
    question = request.message

    docs = retriever.invoke(question)

    context = "\n\n".join(
        [
            f"Loi: {doc.metadata.get('law')}\n"
            f"Fichier: {doc.metadata.get('source_file')}\n"
            f"Page: {doc.metadata.get('page')}\n"
            f"Texte: {doc.page_content[:1500]}"
            for doc in docs
        ]
    )

    messages = prompt.format_messages(
        context=context,
        question=question,
    )

    response = llm.invoke(messages)

    sources = [
        {
            "law": doc.metadata.get("law"),
            "file": doc.metadata.get("source_file"),
            "page": doc.metadata.get("page"),
            "text": doc.page_content[:500],
        }
        for doc in docs
    ]

    return {
        "answer": response.content,
        "sources": sources,
    }