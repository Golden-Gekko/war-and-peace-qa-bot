from langchain_ollama import ChatOllama

from literary_entity_extractor import LiteraryEntityExtractor
from epub_parser import EpubParser

book = EpubParser('D:/Dev/Otus/WarAndPeace.epub')

llm = ChatOllama(model='qwen3:14b')
graph = LiteraryEntityExtractor(llm=llm)

# MODEL_NAME = 'qwen3:14b'
# EMBEDDING_MODEL_NAME = 'bge-m3:567m'

for i, ch in enumerate(book.content):
    for j, chunk in enumerate(ch):
        print(i, j)
        res = graph.invoke(chunk[:1000])
        print('res["characters"]', res['characters'])
        print('res["locations"]', res['locations'])
        print('res["summary"]', res['summary'])
        break

# class DataCreator():
#     def __init__(self):
#         pass


