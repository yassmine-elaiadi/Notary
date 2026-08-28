from online_search import online_legal_search
from web_reader import read_web_page

results = online_legal_search("site:gesetze-im-internet.de BGB Erbrecht Vollmacht Grundstück")

for result in results:
    print("URL:", result["url"])
    text = read_web_page(result["url"])
    print(text[:1000])
    print("=" * 80)