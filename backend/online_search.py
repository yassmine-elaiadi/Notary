from ddgs import DDGS

TRUSTED_SITES = [
    "gesetze-im-internet.de",
    "recht.bund.de",
    "eur-lex.europa.eu",
]


def online_legal_search(query, max_results=5):
    search_query = (
        f"{query} "
        f"(site:gesetze-im-internet.de OR site:recht.bund.de OR site:eur-lex.europa.eu)"
    )

    results = []

    with DDGS() as ddgs:
        for item in ddgs.text(search_query, max_results=max_results):
            url = item.get("href", "")

            if any(site in url for site in TRUSTED_SITES):
                results.append({
                    "title": item.get("title", ""),
                    "url": url,
                    "snippet": item.get("body", "")
                })

    return results