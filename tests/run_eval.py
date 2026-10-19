"""Run the eval questions through the same pipeline as streamlit_app.py and score them.

Usage (from the project root):
    .venv/Scripts/python.exe tests/run_eval.py              # all questions, local laws only
    .venv/Scripts/python.exe tests/run_eval.py --online     # also use the online search, like the app
    .venv/Scripts/python.exe tests/run_eval.py --only fees_180k pflichtteil
"""
import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from src.llm import get_llm
from src.prompts import NOTARY_PROMPT, RETRIEVER_K, format_local_source

QUESTIONS_FILE = ROOT / "tests" / "eval_questions.json"
RESULTS_DIR = ROOT / "tests" / "results"


def normalize(text):
    # Models write thousands with narrow/non-breaking spaces or dots: "1 270", "1.270"
    text = text.replace(" ", " ").replace("\xa0", " ")
    return re.sub(r"(?<=\d)\.(?=\d{3}\b)", " ", text)


def retrieval_hit(expected, docs):
    if not expected:
        return None
    return any(
        doc.metadata.get("law") == law and str(doc.metadata.get("section", "")).startswith(section)
        for law, section in expected
        for doc in docs
    )


def build_context(question, docs, online):
    context = "\n\n".join(format_local_source(doc) for doc in docs)
    if online:
        from backend.online_rag import get_online_legal_sources

        context += "\n\n" + "\n\n".join(
            f"Type: Source officielle en ligne\nTitre: {s['title']}\nURL: {s['url']}\nTexte: {s['text'][:2000]}"
            for s in get_online_legal_sources(question, max_results=3)
        )
    return context


def ask(llm, messages, attempts=3):
    # Groq's free tier rate-limits per minute; wait and retry instead of failing the question
    for attempt in range(attempts):
        try:
            return llm.invoke(messages).content
        except Exception as e:
            if attempt == attempts - 1 or "rate" not in str(e).lower():
                raise
            time.sleep(30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--online", action="store_true", help="add online search results to the context")
    parser.add_argument("--only", nargs="+", help="run only these question ids")
    args = parser.parse_args()

    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    if args.only:
        questions = [q for q in questions if q["id"] in args.only]

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    retriever = Chroma(
        persist_directory=str(ROOT / "chroma_db"), embedding_function=embeddings
    ).as_retriever(search_kwargs={"k": RETRIEVER_K})
    llm = get_llm()

    results = []
    for q in questions:
        docs = retriever.invoke(q["question"])
        messages = NOTARY_PROMPT.format_messages(
            history="", context=build_context(q["question"], docs, args.online), question=q["question"]
        )
        try:
            answer = ask(llm, messages)
            error = None
        except Exception as e:
            answer, error = "", str(e)

        checks = {p: bool(re.search(p, normalize(answer), re.I)) for p in q["must_match"]}
        result = {
            "id": q["id"],
            "topic": q["topic"],
            "question": q["question"],
            "retrieved": [f"{d.metadata.get('law')} {d.metadata.get('section')}" for d in docs],
            "retrieval_ok": retrieval_hit(q["sources"], docs),
            "checks": checks,
            "answer_ok": bool(answer) and all(checks.values()),
            "error": error,
            "note": q.get("note", ""),
            "answer": answer,
        }
        results.append(result)

        status = "PASS" if result["answer_ok"] else "FAIL"
        retrieval = {True: "ok", False: "MISSED", None: "-"}[result["retrieval_ok"]]
        failed = [p for p, ok in checks.items() if not ok]
        print(f"{status}  {q['id']:<22} retrieval {retrieval:<7} {'missing: ' + ', '.join(failed) if failed else ''}{error or ''}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    passed = sum(r["answer_ok"] for r in results)
    scored = [r for r in results if r["retrieval_ok"] is not None]
    print(f"\nAnswers correct: {passed}/{len(results)}")
    print(f"Retrieval found the right section: {sum(r['retrieval_ok'] for r in scored)}/{len(scored)}")
    print(f"Full answers saved to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
