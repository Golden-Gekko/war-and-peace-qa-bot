from epub_parser import EpubParser

book = EpubParser('WarAndPeace.epub')

for i, ch in enumerate(book.content):
    if isinstance(ch, str):
        print(f'Кусок {i+1} из {len(book.content)}. Длина: {len(ch)}')
        continue
    print(f'Кусок {i+1} из {len(book.content)}.')
    for j, chunk in enumerate(ch):
        print(f'  Подкусок {j+1} из {len(ch)}. Длина: {len(chunk)}')


{
  "text": "...",
  "summary": "...",
  "embedding": "...",
  "characters": [],
  "locations": [],
}