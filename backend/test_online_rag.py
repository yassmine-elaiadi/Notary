from online_rag import get_online_legal_sources

sources = get_online_legal_sources("BGB Erbrecht Vollmacht Grundstück", max_results=3)

print("Number of sources:", len(sources))

for source in sources:
    print(source["title"])
    print(source["url"])
    print(source["text"][:500])
    print("=" * 80)