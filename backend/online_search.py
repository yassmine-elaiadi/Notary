from ddgs import DDGS
from ddgs.exceptions import DDGSException

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

    # Online search is optional extra context: on no results, rate limit or timeout,
    # answer from the local law database only instead of crashing
    try:
        with DDGS() as ddgs:
            items = ddgs.text(search_query, max_results=max_results)
    except DDGSException as e:
        print(f"Recherche en ligne indisponible : {e}")
        return results

    for item in items:
        url = item.get("href", "")

        if any(site in url for site in TRUSTED_SITES):
            results.append({
                "title": item.get("title", ""),
                "url": url,
                "snippet": item.get("body", "")
            })

    return results