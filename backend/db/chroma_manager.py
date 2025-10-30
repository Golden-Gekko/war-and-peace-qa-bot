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

    def _create_or_get_collection(
            self, name: str = 'war_and_peace') -> Collection:
        self.collection = self.client.get_or_create_collection(
            name=name,
            metadata={'hnsw:space': 'cosine'}
        )
        return self.collection

    def clear_collection(
            self, collection_name: str = 'war_and_peace') -> None:
        try:
            collection = self.client.get_collection(collection_name)
            all_ids = collection.get(include=[])['ids']
            if all_ids:
                collection.delete(ids=all_ids)
        except Exception as e:
            raise ValueError(
                f'Ошибка при очистке коллекции "{collection_name}": {e}')

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
            'locations': list[str] | None,
            'prev_id': str | None,
            'next_id': str | None
        }
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data: List[Dict[str, Any]] = json.load(f)

        self._create_or_get_collection(collection_name)

        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for i, item in enumerate(tqdm(data, desc='Preparing data for Chroma')):
            ids.append(item.get('id', f'chunk_{i}'))
            documents.append(item.get('text'))

            embedding_str = item.get('embedding')
            if isinstance(embedding_str, str):
                embedding = json.loads(embedding_str)
            else:
                embedding = embedding_str
            embeddings.append(embedding)

            def flat_meta(name: str) -> str:
                if name in metadata and metadata[name]:
                    return ', '.join(metadata[name])
                return ''

            metadata = item.get('metadata', {})
            metadata['characters'] = flat_meta('characters')
            pl = metadata.get('locations', [''])
            metadata['primary_location'] = pl[0] if pl else ''
            metadata['locations'] = flat_meta('locations')
            if not metadata['prev_id']:
                metadata['prev_id'] = ''
            if not metadata['next_id']:
                metadata['next_id'] = ''
            metadatas.append(metadata)

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
        self, query_embedding: List[float], n_results: int = 5,
        where: Dict = None, collection_name: str = 'war_and_peace'
    ):
        self._create_or_get_collection(collection_name)
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where
        )

    def get(self, id: str, collection_name: str = 'war_and_peace'):
        self._create_or_get_collection(collection_name)
        return self.collection.get(ids=[id])
