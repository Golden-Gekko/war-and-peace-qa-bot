import os

from bs4 import BeautifulSoup
from ebooklib import epub


class EpubParser():
    def __init__(self, file_path: str):
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f'Файл не найден: {file_path}')
        if not file_path.lower().endswith('.epub'):
            raise ValueError(
                f'Файл не является EPUB: {file_path}. '
                'Ожидается ".epub" расширение.')
        try:
            self.book = epub.read_epub(file_path)
        except (IOError, epub.EpubException) as e:
            raise ValueError(
                'Файл не является действительным EPUB или поврежден: '
                f'{file_path}. Ошибка: {e}')
        self._parse_content()

    @staticmethod
    def _extract_text_from_item(item, subchapter_class='Z2_1K'):
        soup = BeautifulSoup(item.get_content(), 'html.parser')
        subchapter_markers = soup.find_all('p', class_=subchapter_class)

        if not subchapter_markers:
            return soup.get_text(strip=True)

        sections = []
        previous_element = soup

        for marker in subchapter_markers:
            current_text = []
            for sibling in previous_element.next_siblings:
                if sibling == marker:
                    break
                if isinstance(sibling, str):
                    current_text.append(sibling)
                else:
                    current_text.append(sibling.get_text())
            section_text = "".join(current_text).strip()
            if section_text:
                sections.append(
                    previous_element.get_text() + '\n' + section_text
                )

            previous_element = marker

        current_text = []
        for sibling in previous_element.next_siblings:
            if isinstance(sibling, str):
                current_text.append(sibling)
            else:
                current_text.append(sibling.get_text())
        section_text = "".join(current_text).strip()
        if section_text:
            sections.append(section_text)

        return sections

    @staticmethod
    def _walk_toc(toc, level=0):
        chapters = []
        for item in toc:
            if isinstance(item, epub.Link):
                chapters.append({
                    'title': item.title,
                    'href': item.href,
                    'level': level
                })
            elif isinstance(item, tuple):
                subname, subitems = item
                chapters.append({
                    'title': subname.title,
                    'href': subname.href,
                    'level': level,
                    'is_section': True
                })
                chapters.extend(EpubParser._walk_toc(subitems, level + 1))
        return chapters

    def _parse_content(self):
        self.chapters_info = EpubParser._walk_toc(self.book.toc)
        self.content = []
        for ch in self.chapters_info:
            item = self.book.get_item_with_href(ch['href'])
            text = EpubParser._extract_text_from_item(item)
            if text:
                self.content.append(text)
