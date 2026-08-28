from backend.online_search import online_legal_search
from backend.web_reader import read_web_page

def get_online_legal_sources(query, max_results=3):
    search_results = online_legal_search(query, max_results=max_results)

    sources = []

    for result in search_results:
        text = read_web_page(result["url"])

        if text:
            sources.append({
                "title": result["title"],
                "url": result["url"],
                "text": text
            })

    return sources