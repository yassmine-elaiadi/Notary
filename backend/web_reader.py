import io
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader


def read_pdf_from_bytes(pdf_bytes):
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text = ""

        for page in reader.pages[:10]:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        return text[:5000]
    except Exception:
        return ""


def read_web_page(url):
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
    except Exception:
        return ""

    content_type = response.headers.get("Content-Type", "").lower()

    if url.lower().endswith(".pdf") or "pdf" in content_type:
        return read_pdf_from_bytes(response.content)

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)

    return text[:5000]