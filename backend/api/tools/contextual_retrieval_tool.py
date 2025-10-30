import os
from typing import Any, Dict, List
import yaml

from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from pydantic import BaseModel, Field

from api.literary_entity_extractor import LiteraryEntityExtractor
from db.chroma_manager import ChromaManager


def get_llm():
    return ChatOllama(
        model=os.getenv('LLM_MODEL', 'qwen3:14b'),
        base_url=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
        temperature=0.0
    )


def get_embedding_model():
    return OllamaEmbeddings(
        model=os.getenv('EMBEDDING_MODEL', 'bge-m3:567m'),
        base_url=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    )


def get_chroma_manager():
    persist_dir = os.getenv('CHROMA_PERSIST_DIR', './chroma_db')
    manager = ChromaManager(persist_directory=persist_dir)
    return manager


def get_tool_description() -> str:
    description_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'description.yaml'
    )
    with open(description_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)['description']
    return ''


class ContextualRetrievalInput(BaseModel):
    query: str = Field(
        description='Запрос пользователя о романе "Война и мир".')


class ContextualRetrievalTool(BaseTool):
    name: str = 'retrieve_context_from_war_and_peace'
    description: str = get_tool_description()
    args_schema: type[BaseModel] = ContextualRetrievalInput

    def _filter_by_characters(
            self, results: Dict, characters: List[str]) -> Dict | None:
        if not characters:
            return self._get_first(results)
        metadatas = results['metadatas'][0]
        documents = results['documents'][0]
        ids = results['ids'][0]

        for doc, meta, doc_id in zip(documents, metadatas, ids):
            meta_chars = set(
                meta.get('characters', '').split(', ')
            ) if meta.get('characters') else set()

            if set(characters) & meta_chars:
                return dict(
                    id=doc_id, text=doc,
                    prev_id=meta.get('prev_id', ''),
                    next_id=meta.get('next_id', '')
                )
        return None

    def _get_first(self, results: Dict) -> Dict:
        metadatas = results['metadatas'][0]
        documents = results['documents'][0]
        ids = results['ids'][0]
        return dict(
            id=ids[0], text=documents[0],
            prev_id=metadatas[0].get('prev_id', ''),
            next_id=metadatas[0].get('next_id', '')
        )

    def _expand_context_with_neighbors(
            self, chroma: ChromaManager, initial_results: Dict) -> List[str]:
        expanded = []
        prev_id = initial_results['prev_id']
        next_id = initial_results['next_id']

        if prev_id:
            prev_res = chroma.get(prev_id)
            if prev_res['documents']:
                expanded.append(prev_res['documents'][0])
        expanded.append(initial_results['text'])
        if next_id:
            next_res = chroma.get(next_id)
            if next_res['documents']:
                expanded.append(next_res['documents'][0])
        return expanded

    def _run(self, query: str, **kwargs: Any) -> str:
        try:
            extractor = LiteraryEntityExtractor(llm=get_llm())
            state = extractor.invoke(query)
            characters = state.get('characters', [])
            locations = state.get('locations', [])

            where = None
            if locations:
                where = {'primary_location': locations[0]}

            embedder = get_embedding_model()
            chroma = get_chroma_manager()
            query_embedding = embedder.embed_query(query)

            try:
                results = chroma.query(
                    query_embedding=query_embedding,
                    n_results=5,
                    where=where
                )
            except Exception:
                results = chroma.query(query_embedding=query_embedding)

            filtered_by_chars = self._filter_by_characters(results, characters)

            if not filtered_by_chars:
                fallback = self._get_first(results)
                if fallback:
                    filtered_by_chars = fallback
                else:
                    return 'Не найдено релевантных фрагментов.'

            expanded_context = self._expand_context_with_neighbors(
                chroma, filtered_by_chars)

            return "\n\n".join(expanded_context)

        except Exception as e:
            return f'Ошибка при поиске контекста: {str(e)}'
