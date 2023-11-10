# 3. Google поиск через Python:
# https://pypi.org/project/googlesearch-python/
# pip install googlesearch-python

from googlesearch import search

# Define the query you want to search
query = "Функція активації у машинному навчанні"

# Specify the number of search results you want to retrieve
num_results = 5
lang="ua"
# lang="en"

# Perform the search and retrieve the results
search_results = search(query, num_results=num_results, lang=lang,  advanced=True)

# Print the search results
for result in search_results:
    # print(result)
    print(f"\ntitle: {result.title} ")
    print(f"description: {result.description} ")
    print(f"url: {result.url} ")

