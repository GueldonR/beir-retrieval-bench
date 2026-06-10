from openai import OpenAI
from transformers import AutoTokenizer

from src.config.settings import EMBEDDING_URL


class EmbeddingModelInstance:
    def __init__(self):
        self.embed_client = OpenAI(base_url=EMBEDDING_URL, api_key="sk-no-key-required")
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")

    def embed_text(self, text, is_query=False, max_token_length=512):
        """
        Single text to vector embedding using the OpenAI API.
        """
        if not text:
            raise ValueError("No text input was passed")

        if is_query:
            task_description = "Retrieve relevant passages"
            input_text = f"Instruct: {task_description}\nQuery: {text}"
        else:
            input_text = text
        encoded = self.tokenizer(
            input_text, truncation=True, max_length=max_token_length, add_special_tokens=False
        )
        truncatedText = self.tokenizer.decode(encoded["input_ids"])
        try:
            response = self.embed_client.embeddings.create(
                model="Qwen/Qwen3-Embedding-0.6B", input=truncatedText
            )

        except Exception as e:
            raise RuntimeError(f"Single embedding failed: {truncatedText[:20]}") from e

        return response.data[0].embedding

    def embed_texts_batch(
        self, texts: list[str], is_query=False, max_token_length=512
    ) -> list[list[float]]:
        """
        Batch embedding using the OpenAI API's list input support.
        """
        truncated_texts = []

        if texts is None:
            raise ValueError("You cant embed an empty batch")

        for t in texts:
            if is_query:
                task_description = "Retrieve relevant passages"
                input_text = f"Instruct: {task_description}\nQuery: {t}"
                encoded = self.tokenizer(
                    input_text,
                    truncation=True,
                    max_length=max_token_length,
                    add_special_tokens=False,
                )
                truncated_texts.append(self.tokenizer.decode(encoded["input_ids"]))
            else:
                encoded = self.tokenizer(
                    t, truncation=True, max_length=max_token_length, add_special_tokens=False
                )
                truncated_texts.append(self.tokenizer.decode(encoded["input_ids"]))
        response = self.embed_client.embeddings.create(
            model="Qwen/Qwen3-Embedding-0.6B", input=truncated_texts
        )
        return [item.embedding for item in response.data]
