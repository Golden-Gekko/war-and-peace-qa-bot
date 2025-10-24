import json
import os
from typing import Any, Dict, List
from tqdm import tqdm

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings


class ChromaManager:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

    def create_or_get_collection(
            self, name: str = 'war_and_peace') -> Collection:
        self.collection = self.client.get_or_create_collection(
            name=name,
            metadata={'hnsw:space': 'cosine'}
        )
        return self.collection

    def load_from_json(
            self, json_path: str,
            collection_name: str = 'war_and_peace'):
        """
        Ожидаемый формат JSON: список объектов вида:
        {
          'text': str,
          'embedding': str | list[float],
          'characters': list[str] | None,
          'locations': list[str] | None
        }
        """

        with open(json_path, 'r', encoding='utf-8') as f:
            data: List[Dict[str, Any]] = json.load(f)

        self.create_or_get_collection(collection_name)

        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for i, item in enumerate(tqdm(data, desc='Preparing data for Chroma')):
            doc_id = f'chunk_{i}'
            text = item.get('text')
            embedding_str = item.get('embedding')
            if isinstance(embedding_str, str):
                embedding = json.loads(embedding_str)
            else:
                embedding = embedding_str            
            characters = item.get('characters', [])
            locations = item.get('locations', [])

            ids.append(doc_id)
            embeddings.append(embedding)
            documents.append(text)

            metadatas.append({
                'characters': ', '.join(characters) if characters else '',
                'locations': ', '.join(locations) if locations else '',
            })

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

        print(
            f'Загружено {len(ids)} документов в коллекцию "{collection_name}"')

    def query(
            self,
            query_embedding: List[float],
            n_results: int = 5,
            where: Dict = None):
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where
        )


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python chroma_manager.py <path_to_chunks.json>")
        sys.exit(1)

    json_path = sys.argv[1]
    manager = ChromaManager(persist_directory='./chroma_db')
    manager.load_from_json(json_path)
