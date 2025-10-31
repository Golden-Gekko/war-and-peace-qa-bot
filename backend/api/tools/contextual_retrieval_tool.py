import os
from typing import Any, Dict, List
import yaml

from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from loguru import logger
from pydantic import BaseModel, Field

from api.literary_entity_extractor import LiteraryEntityExtractor
from db.chroma_manager import ChromaManager

OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'localhost')
OLLAMA_PORT = os.getenv('OLLAMA_PORT', '11434')
OLLAMA_BASE_URL = f'http://{OLLAMA_HOST}:{OLLAMA_PORT}'
MAX_LOG_LEN = 100
MAX_CONTEXT_LEN = 5


def get_llm():
    return ChatOllama(
        model=os.getenv('LLM_MODEL', 'qwen3:14b'),
        base_url=OLLAMA_BASE_URL,
        temperature=0.0
    )


def get_embedding_model():
    return OllamaEmbeddings(
        model=os.getenv('EMBEDDING_MODEL', 'bge-m3:567m'),
        base_url=OLLAMA_BASE_URL
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
            self, results: Dict, characters: List[str]) -> List[Dict]:
        if not characters:
            return self._convert_results(results)
        items = []
        metadatas = results['metadatas'][0]
        documents = results['documents'][0]
        ids = results['ids'][0]

        for doc, meta, doc_id in zip(documents, metadatas, ids):
            meta_chars = set(
                meta.get('characters', '').split(', ')
            ) if meta.get('characters') else set()

            if set(characters) & meta_chars:
                items.append(
                    dict(
                        id=doc_id, text=doc,
                        prev_id=meta.get('prev_id', ''),
                        next_id=meta.get('next_id', '')
                    )
                )
        return items

    def _convert_results(self, results: Dict[str, List]) -> List[Dict]:
        items = []
        metadatas = results['metadatas'][0]
        documents = results['documents'][0]
        ids = results['ids'][0]
        for i in range(len(ids)):
            items.append(
                dict(
                    id=ids[i], text=documents[i],
                    prev_id=metadatas[i].get('prev_id', ''),
                    next_id=metadatas[i].get('next_id', '')
                )
            )
        return items

    def _expand_context_with_neighbors(
        self, chroma: ChromaManager, initial_results: List[Dict]
    ) -> List[str]:
        added_id = []
        expanded = []
        for res in initial_results:
            if res['id'] in added_id:
                continue
            added_id.append(res['id'])
            prev_id = res['prev_id']
            next_id = res['next_id']

            if prev_id:
                prev_res = chroma.get(prev_id)
                if prev_res['documents']:
                    expanded.append(prev_res['documents'][0])
            expanded.append(res['text'])
            if next_id:
                next_res = chroma.get(next_id)
                if next_res['documents']:
                    expanded.append(next_res['documents'][0])
        return expanded

    def _run(self, query: str, **kwargs: Any) -> str:
        try:
            logger.info('Вызван "ContextualRetrievalTool"')
            extractor = LiteraryEntityExtractor(
                llm=get_llm(), need_summary=False)
            state = extractor.invoke(query)
            characters = state.get('characters', [])
            locations = state.get('locations', [])
            logger.info(f'Извлечено "Сharacters": {characters}')
            logger.info(f'Извлечено "Locations" : {locations}')

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
                fallback = self._convert_results(results)
                if fallback:
                    filtered_by_chars = fallback
                else:
                    logger.info('Не найдено релевантных фрагментов')
                    return 'Не найдено релевантных фрагментов.'

            expanded_context = self._expand_context_with_neighbors(
                chroma, filtered_by_chars)
            for item in expanded_context[:MAX_CONTEXT_LEN]:
                logger.info(f'Получен ответ от БД: {item[:MAX_LOG_LEN]}')
            return '\n\n'.join(expanded_context[:MAX_CONTEXT_LEN])

        except Exception as e:
            logger.error(f'Ошибка при поиске контекста: {str(e)}')
            return f'Ошибка при поиске контекста: {str(e)}'
