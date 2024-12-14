import os
import dotenv
from tavily import TavilyClient
import requests

dotenv.load_dotenv()

class TavilySearch:
    def __init__(self, api_key, text=None):
        self.api_key = api_key
        self.text = text
        self.client = TavilyClient(api_key=api_key)

    def search(self, query, max_results=2, search_depth='basic'):
        try:
            results = self.client.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth
            )

            return results
        except Exception as e:
            raise e

class GoogleSearch:
    def __init__(self, api_key, cx):
        self.api_key = api_key
        self.cx = cx

    def search(self, query, max_results=2):
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": self.api_key,
                "cx": self.cx,
                "q": query,
                "num": max_results
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            results = [
                {
                    "title": item.get("title"),
                    "url": item.get("link"),
                    "content": item.get("snippet")
                } for item in data.get("items", [])
            ]

            return results
        except Exception as e:
            raise e

class MultiSearch:
    def __init__(self, tavily_api_key, google_api_key, google_cx):
        self.clients = [
            TavilySearch(api_key=tavily_api_key),
            GoogleSearch(api_key=google_api_key, cx=google_cx)
        ]

    def search(self, query, max_results=2):
        results = []
        for client in self.clients:
            try:
                results.append(client.search(query=query, max_results=max_results))
            except Exception as e:
                print(f"Error with client {client}: {e}")
        return results

# Initialize with API keys
tavily_api_key = os.getenv('TAVILY_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')
google_cx = os.getenv('GOOGLE_CX')

multi_search = MultiSearch(tavily_api_key=tavily_api_key, google_api_key=google_api_key, google_cx=google_cx)
# res = multi_search.search("What is the best food for Bangladeshi people?")
# print(res)
