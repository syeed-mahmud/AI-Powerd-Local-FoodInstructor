import os
from langchain.prompts.chat import ChatPromptTemplate
from SearchClient import MultiSearch
import dotenv

# Load environment variables
dotenv.load_dotenv()

class WebSearch:
    def __init__(self):
        # Initialize the MultiSearch with API keys
        self.search_engine = MultiSearch(
            tavily_api_key=os.getenv('TAVILY_API_KEY'),
            google_api_key=os.getenv('GOOGLE_API_KEY'),
            google_cx=os.getenv('GOOGLE_CX')
        )

    def search_prompt(self, context):
        # Define the system and user prompts
        system_prompt_text = """You are a helpful assistant. You will assist me by generating queries to search the web for \
recipes to cook on a given context and just returning the queries in a numbered list format. \
You MUST ONLY generate queries on food or diet or if the context is based on food. If asked on any other context or \
topic you must reject the prompt in a respectful way."""

        user_prompt_text = f"Generate 3 queries to find food recipes for context: '{context}'"

        # Create a ChatPromptTemplate
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ('system', system_prompt_text),
                ('user', user_prompt_text)
            ]
        )

        # Format the prompt with context
        return prompt_template.format_prompt(context=context).to_string()  # Changed: Used `format_prompt` method to correctly format prompt

    def search_web(self, queries):
        # Perform a web search using the generated queries
        results = []
        for query in queries:
            results.extend(self.search_engine.search(query=query))  # Changed: Use `extend` to add results from each query
        return results


from WebSearch import WebSearch
web_search = WebSearch()

    # Generate the prompt and queries
context = "Italian pasta"
prompt = web_search.search_prompt(context)

    # Extract numbered queries from the prompt
queries = [line.split(". ", 1)[1] for line in prompt.strip().split("\n") if line.strip().startswith("1.")]  # Changed: Extract queries from the formatted prompt

    # Perform the search
results = web_search.search_web(queries)

    # Print the results
for result in results:
        print(result)  # Changed: Print the search results clearly
