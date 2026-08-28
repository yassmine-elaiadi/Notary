"""
Downloads the official German law texts from gesetze-im-internet.de as XML
(the format the site recommends for automated processing) and indexes them
into ChromaDB, one chunk per legal section (Section/§). This replaces the
old PDF-based ingest.py: PDFs force us to guess where a section starts and
ends from plain text, which breaks whenever a section spans a page boundary.
XML gives each section as one structural element, so chunking is exact.

Run it directly to build/refresh the index:

    .venv/Scripts/python.exe update_laws.py

Safe to re-run: each law's XML is hashed, and a law is only re-parsed and
re-indexed when its official text actually changed since the last run.
"""

import hashlib
import io
import re
import zipfile
from datetime import date
from pathlib import Path
from xml.etree import ElementTree

import requests
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHROMA_FOLDER = "chroma_db"
XML_CACHE_FOLDER = Path("data/laws_xml")
SOURCE_TYPE = "official_xml"

# A handful of provisions (mainly fee-table annexes like GNotKG's
# Kostenverzeichnis) run to tens of thousands of characters. Left as one
# chunk, they'd get silently truncated by the embedding model or blow out
# the LLM's whole context if ever retrieved - split those into pieces.
MAX_SECTION_CHARS = 3000
section_splitter = RecursiveCharacterTextSplitter(
    chunk_size=MAX_SECTION_CHARS,
    chunk_overlap=0,
    separators=["\n\n", "\n"],
)

# Fixed list of laws relevant to German notarial practice, mapped to their
# slug on gesetze-im-internet.de (used both for the download URL and for
# building per-section citation links).
LAWS = {
    "BGB": "bgb",
    "EGBGB": "bgbeg",
    "FamFG": "famfg",
    "GBO": "gbo",
    "WEG": "woeigg",
    "BeurkG": "beurkg",
    "BNotO": "bnoto",
    "GNotKG": "gnotkg",
}


def download_law_xml(slug):
    url = f"https://www.gesetze-im-internet.de/{slug}/xml.zip"
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        xml_names = [n for n in archive.namelist() if n.lower().endswith(".xml")]
        if not xml_names:
            raise ValueError(f"No XML file found in {url}")
        return archive.read(xml_names[0])


def has_changed(vectorstore, law, slug, xml_bytes):
    cached_path = XML_CACHE_FOLDER / f"{slug}.xml"

    if not cached_path.exists():
        return True

    # The cache file alone isn't proof the DB has this law's data: a run
    # can be interrupted after the Chroma write returns but before it's
    # durably persisted, leaving the cache file written but the DB empty
    # for that law. Trust the cache hash only if the DB backs it up.
    existing = vectorstore.get(where={"$and": [{"law": law}, {"source_type": SOURCE_TYPE}]})
    if not existing["ids"]:
        return True

    new_hash = hashlib.sha256(xml_bytes).hexdigest()
    old_hash = hashlib.sha256(cached_path.read_bytes()).hexdigest()
    return new_hash != old_hash


def section_url(slug, enbez):
    match = re.search(r"§\s*([0-9]+[a-z]?)", enbez)
    if not match:
        return f"https://www.gesetze-im-internet.de/{slug}/"
    return f"https://www.gesetze-im-internet.de/{slug}/__{match.group(1)}.html"


def table_to_text(table_el):
    lines = []
    for row in table_el.iter("row"):
        cells = ["".join(entry.itertext()).strip() for entry in row.findall("entry")]
        line = " | ".join(c for c in cells if c)
        if line:
            lines.append(line)
    return "\n".join(lines)


def parse_sections(law, slug, xml_bytes):
    root = ElementTree.fromstring(xml_bytes)

    sections = []
    breadcrumb = ""

    for norm in root.findall("norm"):
        metadaten = norm.find("metadaten")
        if metadaten is None:
            continue

        enbez = metadaten.findtext("enbez")
        if enbez == "Inhaltsübersicht":
            # Table of contents, not actual provision text.
            continue

        content = norm.find(".//textdaten/text/Content")
        paragraphs = (
            ["".join(p.itertext()).strip() for p in content.findall("P")]
            if content is not None
            else []
        )
        # Some provisions (e.g. GNotKG's fee schedules) carry their real
        # content as a <table> rather than <P> paragraphs - without this,
        # that content is silently dropped.
        tables = (
            [table_to_text(t) for t in content.findall("table")]
            if content is not None
            else []
        )
        text = "\n".join(p for p in paragraphs if p)
        table_text = "\n".join(t for t in tables if t)
        if table_text:
            text = f"{text}\n\n{table_text}".strip() if text else table_text

        gliederung = metadaten.find("gliederungseinheit")
        titel = metadaten.findtext("titel", "")

        if gliederung is not None:
            bez = gliederung.findtext("gliederungsbez", "")
            glied_titel = gliederung.findtext("gliederungstitel", "")

            if not text:
                # Pure structural heading (Buch/Abschnitt/Kapitel), no
                # provision text of its own: just update the breadcrumb.
                breadcrumb = f"{bez} {glied_titel}".strip()
                continue

            # Some laws (e.g. EGBGB, numbered by "Artikel") encode the
            # provision itself as a gliederungseinheit leaf rather than
            # a separate enbez entry - use it as the section directly.
            enbez = bez
            titel = glied_titel

        if not enbez or not text:
            continue

        header = (
            f"{law} {enbez}"
            + (f" - {titel}" if titel else "")
            + (f"\n({breadcrumb})" if breadcrumb else "")
        )

        pieces = (
            section_splitter.split_text(text)
            if len(text) > MAX_SECTION_CHARS
            else [text]
        )
        multi_part = len(pieces) > 1

        for i, piece in enumerate(pieces, start=1):
            part_suffix = f" (partie {i}/{len(pieces)})" if multi_part else ""

            sections.append(
                Document(
                    page_content=f"{header}{part_suffix}\n\n{piece}",
                    metadata={
                        "law": law,
                        "section": f"{enbez}{part_suffix}",
                        "title": titel,
                        "gliederung": breadcrumb,
                        "url": section_url(slug, enbez),
                        "language": "de",
                        "retrieved_date": date.today().isoformat(),
                        "source_type": SOURCE_TYPE,
                    },
                )
            )

    return sections


def replace_law_in_vectorstore(vectorstore, law, sections):
    existing = vectorstore.get(where={"$and": [{"law": law}, {"source_type": SOURCE_TYPE}]})
    if existing["ids"]:
        vectorstore.delete(ids=existing["ids"])

    if sections:
        vectorstore.add_documents(sections)


def main():
    XML_CACHE_FOLDER.mkdir(parents=True, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma(
        persist_directory=CHROMA_FOLDER,
        embedding_function=embeddings,
    )

    for law, slug in LAWS.items():
        print(f"Checking {law} ({slug})...")
        xml_bytes = download_law_xml(slug)

        if not has_changed(vectorstore, law, slug, xml_bytes):
            print(f"  unchanged, skipping.")
            continue

        sections = parse_sections(law, slug, xml_bytes)
        print(f"  changed: {len(sections)} sections. Updating index...")

        replace_law_in_vectorstore(vectorstore, law, sections)

        (XML_CACHE_FOLDER / f"{slug}.xml").write_bytes(xml_bytes)
        print(f"  done.")

    print("All laws checked.")


if __name__ == "__main__":
    main()
