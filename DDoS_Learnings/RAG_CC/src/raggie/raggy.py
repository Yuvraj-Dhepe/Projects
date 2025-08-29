import os

from llama_index.llms.ollama import Ollama

OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")


class RAG:
    def __init__(self, retriever, llm_name="gemma3n:e2b"):
        self.llm_name = llm_name
        self.llm = self._setup_llm()
        self.retriever = retriever
        self.qa_prompt_tmpl_str = """
        Context information is below.
        ---------------------
        {context}
        ---------------------
        Given the context information above I want you
        to think step by step to answer the query in a
        crisp manner, incase case you don't know the
        answer say 'I don't know!'
        ---------------------
        Query: {query}
        ---------------------
        Answer:
        """

    def _setup_llm(self):
        return Ollama(model=self.llm_name, base_url=OLLAMA_BASE_URL)

    # Retrieve relevant results from the vector database
    def generate_context(self, query):
        # Use the retriever to get relevant context
        search_result = self.retriever.search(query)
        if not search_result.points:
            return "No relevant context found."

        # Iterate through the search results and extract the context field
        # from each points payload and append each context
        # to a list called combined_prompt
        context = [dict(point) for point in search_result.points]
        combined_prompt = []
        for entry in context:
            context = entry["payload"]["context"]
            combined_prompt.append(context)

        return "\n\n --- \n\n".join(combined_prompt)

    # Collating everything together into a query method,
    # which will accept the user query,
    # generate a context for it, format the prompt
    # template, to create a prompt,
    # send it to the LLM, and return the generated response.

    def query(self, query):
        context = self.generate_context(query)

        prompt = self.qa_prompt_tmpl_str.format(context=context, query=query)

        response = self.llm.complete(prompt)

        return context, dict(response)["text"]
