import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


PDF_FOLDER = "data/laws"
CHROMA_FOLDER = "chroma_db"


def detect_law(source):
    source = source.lower()

    if "bgb" in source and "egbgb" not in source:
        return "BGB"
    if "egbgb" in source or "bgbeg" in source:
        return "EGBGB"
    if "famfg" in source:
        return "FamFG"
    if "gbo" in source:
        return "GBO"
    if "weg" in source or "woeg" in source:
        return "WEG"
    if "beurkg" in source:
        return "BeurkG"
    if "bnoto" in source:
        return "BNotO"
    if "gnotkg" in source:
        return "GNotKG"

    return "Unknown"


def detect_language(source):
    source = source.lower()

    if "english" in source or "englisch" in source:
        return "en"

    return "de"


def main():
    print("Loading PDFs...")

    loader = PyPDFDirectoryLoader(PDF_FOLDER)
    documents = loader.load()

    print(f"Loaded {len(documents)} pages.")

    for doc in documents:
        source = doc.metadata.get("source", "")

        doc.metadata["law"] = detect_law(source)
        doc.metadata["language"] = detect_language(source)
        doc.metadata["source_file"] = os.path.basename(source)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n§", "\nArticle", "\n\n", "\n", ".", " "],
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    print("Creating embeddings...")

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("Saving to ChromaDB...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_FOLDER,
    )

    print("Done.")
    print(f"Vector database saved in: {CHROMA_FOLDER}")


if __name__ == "__main__":
    main()