from __future__ import annotations

from typing import List, Optional

from openai import OpenAI


class EmbeddingClient:
    def __init__(self, api_key: str, model: str = "text-embedding-3-small", chat_model: str = "gpt-4o-mini"):
        self._client = OpenAI(api_key=api_key)
        self._emb_model = model
        self._chat_model = chat_model

    def embed_text(self, text: str) -> List[float]:
        res = self._client.embeddings.create(model=self._emb_model, input=text)
        return list(res.data[0].embedding)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        res = self._client.embeddings.create(model=self._emb_model, input=texts)
        return [list(d.embedding) for d in res.data]

    def generate_answer(self, system_prompt: str, context: str, question: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer concisely and cite relevant projects."},
        ]
        res = self._client.chat.completions.create(model=self._chat_model, messages=messages, temperature=0.2)
        return res.choices[0].message.content or ""


