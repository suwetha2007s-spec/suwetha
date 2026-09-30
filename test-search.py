from duckduckgo_search import DDGS

with DDGS() as ddgs:
    results = list(ddgs.text(
        "What is cyber law in India?",
        max_results=3
    ))

for r in results:
    print(r["title"])
    print(r["body"])
    print(r["href"])
    print("-" * 50)