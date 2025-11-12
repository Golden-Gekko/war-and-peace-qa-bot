import httpx
import json
import os
from pathlib import Path
import sys
from typing import List, Dict, Any
from tqdm import tqdm
import uuid

from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from api.literary_entity_extractor import LiteraryEntityExtractor
from db import ChromaManager
from utils import EpubParser

CHUNK_SIZE = 4096
CHUNK_OVERLAP = 256
EMBEDDING_SIZE = 4096

load_dotenv()
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'localhost')
OLLAMA_PORT = os.getenv('OLLAMA_PORT', '11434')
OLLAMA_BASE_URL = f'http://{OLLAMA_HOST}:{OLLAMA_PORT}'


def preload_ollama_models():
    models = [
        os.getenv('LLM_MODEL', 'qwen3:14b'),
        os.getenv('EMBEDDING_MODEL', 'bge-m3:567m')
    ]
    for model in models:
        print(f'Preloading {model}...')
        try:
            response = httpx.post(
                f'{OLLAMA_BASE_URL}/api/pull',
                json={'name': model},
                timeout=600
            )
            response.raise_for_status()
            print(f'{model} loaded')
        except Exception as e:
            print(f'Failed to load {model}: {e}')

    llm = ChatOllama(
        model=os.getenv('LLM_MODEL', 'qwen3:14b'),
        base_url=OLLAMA_BASE_URL,
    )
    embedder = OllamaEmbeddings(
        model=os.getenv('EMBEDDING_MODEL', 'bge-m3:567m'),
        base_url=OLLAMA_BASE_URL
    )

    return llm, embedder


def create_json(
        llm, embedder,
        file_path: str,
        start_from: int = 0) -> str:
    try:
        book = EpubParser(file_path)
    except Exception as e:
        print('Невозможно создать экземпляр "EpubParser":', e)
    json_path = os.path.join(os.path.dirname(file_path), 'JSON')
    os.makedirs(json_path, exist_ok=True)
    graph = LiteraryEntityExtractor(llm)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=['\n\n', '\n', '. ', ' ', '']
    )

    all_chunks: List[Dict[str, Any]] = []

    for block_idx, block_lines in enumerate(book.content):
        if block_idx + 1 < start_from:
            continue

        print(f'===== Раздел {block_idx+1} из {len(book.content)} =====')
        full_block_text = '\n'.join(block_lines).strip()
        if not full_block_text:
            continue

        chunks = text_splitter.split_text(full_block_text)

        if not chunks:
            continue

        chunk_ids = [str(uuid.uuid4()) for _ in chunks]

        for i, chunk_text in enumerate(
            tqdm(chunks, desc='Обработка раздела')
        ):
            chunk_id = chunk_ids[i]
            prev_id = chunk_ids[i - 1] if i > 0 else None
            next_id = chunk_ids[i + 1] if i < len(chunks) - 1 else None

            try:
                extraction = graph.invoke(chunk_text)
                characters = extraction.get('characters', [])
                locations = extraction.get('locations', [])
                summary = extraction.get('summary', '').strip()
            except Exception as e:
                print(f'Ошибка при обработке чанка {chunk_id}: {e}')

            try:
                if not summary:
                    embedding = embedder.embed_query(chunk_text)
                else:
                    embedding = embedder.embed_query(summary)
            except Exception as e:
                print(f'Ошибка при генерации эмбеддинга для {chunk_id}: {e}')
                embedding = [0.0] * EMBEDDING_SIZE

            record = {
                'id': chunk_id,
                'text': chunk_text,
                'embedding': embedding,
                'metadata': {
                    'characters': characters,
                    'locations': locations,
                    'prev_id': prev_id,
                    'next_id': next_id,
                }
            }
            all_chunks.append(record)

        output_path = os.path.join(json_path, f'part_{block_idx + 1}.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        all_chunks.clear()

    return json_path


def main(file_path: str | None = None, start_from: int = 0):
    llm, embedder = preload_ollama_models()
    if file_path:
        try:
            json_path = Path(
                create_json(llm, embedder, file_path, start_from=start_from))
        except KeyboardInterrupt:
            print('Преобразование прервано пользователем')
            return
    else:
        json_path = Path(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'JSON'))

    manager = ChromaManager(persist_directory='./chroma_db')
    manager.delete_collection()
    for json_file in json_path.glob('*.json'):
        manager.load_from_json(json_file)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        main()
        sys.exit(1)

    file_path = sys.argv[1]
    start_from = 0

    if len(sys.argv) >= 3:
        try:
            start_from = int(sys.argv[2])
        except ValueError:
            print('Ошибка: start_from должен быть целым числом.')
            sys.exit(1)

    main(file_path, start_from=start_from)
