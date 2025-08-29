import os

from llama_index.embeddings.ollama import OllamaEmbedding
from tqdm import tqdm

from raggie.utils import batch_iterate

OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")


class EmbedData:
    def __init__(
        self,
        embed_model_name="hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0",
        batch_size=32,
    ):
        self.embed_model_name = embed_model_name
        self.batch_size = batch_size

        # Load the model only once for an instance
        self.embed_model = self._load_embed_model()
        self.embeddings = []

    def _load_embed_model(self):
        embed_model = OllamaEmbedding(
            model_name=self.embed_model_name,
            base_url=OLLAMA_BASE_URL,
        )
        return embed_model

    def generate_embeddings(self, text):
        embeddings = self.embed_model.get_text_embedding_batch(texts=text)
        return embeddings

    def embed(self, contexts):
        self.contexts = contexts
        for batch in tqdm(
            batch_iterate(contexts, self.batch_size), desc="Embedding"
        ):
            embeddings = self.generate_embeddings(batch)
            self.embeddings.extend(embeddings)
