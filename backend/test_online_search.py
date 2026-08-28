from online_search import online_legal_search

results = online_legal_search("German inheritance law property power of attorney")

print("Number of results:", len(results))

for result in results:
    print(result["title"])
    print(result["url"])
    print(result["snippet"])
    print()