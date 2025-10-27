import json
import os
from pathlib import Path
from typing import Any, Dict, List, Union
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
        self, json_path: Union[str, Path],
        collection_name: str = 'war_and_peace'
    ):
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
            doc_id = item.get('id', f'chunk_{i}')
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
            f'Из файла "{json_path}" загружено {len(ids)} документов в '
            f'коллекцию "{collection_name}"')

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
