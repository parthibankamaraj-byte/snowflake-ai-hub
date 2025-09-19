from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np

try:
    from pinecone import Pinecone, ServerlessSpec
except Exception:  # pragma: no cover
    Pinecone = None  # type: ignore
    ServerlessSpec = None  # type: ignore


@dataclass
class VectorDocument:
    id: str
    text: str
    metadata: Dict[str, object]


@dataclass
class SearchResult:
    document: VectorDocument
    score: float


class VectorStore:
    def __init__(self):
        self._backend = "memory"

    # -------- Memory backend --------
    @staticmethod
    def memory_backend(storage_dir: Path) -> "VectorStore":
        store = VectorStore()
        store._backend = "memory"
        storage_dir.mkdir(parents=True, exist_ok=True)
        store._storage_dir = storage_dir
        store._index_path = storage_dir / "index.npy"
        store._meta_path = storage_dir / "meta.jsonl"
        return store

    def _load_memory(self) -> Tuple[np.ndarray, List[VectorDocument]]:
        if self._index_path.exists():
            vectors = np.load(self._index_path)
        else:
            vectors = np.zeros((0, 1536), dtype=np.float32)
        documents: List[VectorDocument] = []
        if self._meta_path.exists():
            for line in self._meta_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                documents.append(VectorDocument(**item))
        return vectors, documents

    def _save_memory(self, vectors: np.ndarray, documents: List[VectorDocument]) -> None:
        np.save(self._index_path, vectors)
        with self._meta_path.open("w", encoding="utf-8") as f:
            for doc in documents:
                f.write(json.dumps(doc.__dict__, ensure_ascii=False) + "\n")

    # -------- Pinecone backend --------
    @staticmethod
    def pinecone_backend(index_name: str, api_key: str, environment: Optional[str] = None) -> "VectorStore":
        if Pinecone is None:
            raise RuntimeError("pinecone-client is not installed. Install it or use memory backend.")
        store = VectorStore()
        store._backend = "pinecone"
        store._pc = Pinecone(api_key=api_key)
        if index_name not in [idx.name for idx in store._pc.list_indexes()]:
            spec = None
            if ServerlessSpec is not None:
                region = environment or "us-east-1"
                spec = ServerlessSpec(cloud="aws", region=region)
            store._pc.create_index(name=index_name, dimension=1536, spec=spec)  # type: ignore[arg-type]
        store._index = store._pc.Index(index_name)
        return store

    # -------- Public API --------
    def upsert(self, documents: List[VectorDocument], vectors: List[List[float]]) -> None:
        if self._backend == "pinecone":
            items = []
            for doc, vec in zip(documents, vectors):
                items.append({
                    "id": doc.id,
                    "values": vec,
                    "metadata": {"text": doc.text, **doc.metadata},
                })
            self._index.upsert(vectors=items)
            return

        vectors_np, docs = self._load_memory()
        new_vecs = np.array(vectors, dtype=np.float32)
        combined = new_vecs if vectors_np.shape[0] == 0 else np.vstack([vectors_np, new_vecs])
        self._save_memory(combined, docs + documents)

    def similarity_search(self, query_vector: List[float], top_k: int = 5) -> List[SearchResult]:
        if self._backend == "pinecone":
            res = self._index.query(vector=query_vector, top_k=top_k, include_metadata=True)
            results: List[SearchResult] = []
            for match in res.matches:
                metadata = dict(match.metadata or {})
                text = metadata.pop("text", "")
                doc = VectorDocument(id=match["id"], text=text, metadata=metadata)  # type: ignore[index]
                results.append(SearchResult(document=doc, score=float(match["score"])) )  # type: ignore[index]
            return results

        vectors_np, documents = self._load_memory()
        if vectors_np.shape[0] == 0:
            return []
        q = np.array(query_vector, dtype=np.float32)
        norms = np.linalg.norm(vectors_np, axis=1) * (np.linalg.norm(q) + 1e-8)
        sims = (vectors_np @ q) / (norms + 1e-8)
        top_idx = np.argsort(-sims)[:top_k]
        return [SearchResult(document=documents[i], score=float(sims[i])) for i in top_idx]


