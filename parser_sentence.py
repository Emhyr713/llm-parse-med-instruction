import json
from pathlib import Path
from razdel import sentenize

class SentenceSplitter:
    """Разделяет текст на фармакодинамику/фармакокинетику и на предложения"""
    
    def _remove_marker(self, text: str, marker: str) -> str:
        """Удаление маркера из начала текста"""
        if text.startswith(marker):
            return text[len(marker):].lstrip()
        return text
    
    def split_text(self, text: str) -> tuple[list[str], list[str]]:
        """Принимает текст, возвращает (pd_предложения, pk_предложения)"""
        # Поиск секций
        pd_pos = text.find('Фармакодинамика')
        pk_pos = text.find('Фармакокинетика')
        
        # Разделение на секции
        if pk_pos != -1:
            pd_text = text[:pk_pos].strip()
            pk_text = text[pk_pos:].strip()
        else:
            pd_text = pk_text = text
        
        # Удаление маркеров
        pd_text = self._remove_marker(pd_text, 'Фармакодинамика')
        pk_text = self._remove_marker(pk_text, 'Фармакокинетика')
        
        # Разбивка на предложения
        pd_sentences = [s.text for s in sentenize(pd_text)] if pd_text else []
        pk_sentences = [s.text for s in sentenize(pk_text)] if pk_text else []
        
        # Если секции не разделены
        if pd_text == pk_text and pd_text:
            pk_sentences = pd_sentences.copy()
        
        return pd_sentences, pk_sentences


def main():
    # Загрузка
    with open("data/extracted_data_all.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Обработка
    splitter = SentenceSplitter()
    results = []
    
    for item in data:
        if 'drug' not in item or 'text' not in item:
            continue
            
        pd_sentences, pk_sentences = splitter.split_text(item['text'])
        results.append({
            'drug': item['drug'],
            'pd': pd_sentences,
            'pk': pk_sentences
        })
    
    # Сохранение
    Path("data").mkdir(exist_ok=True)
    with open("data/pd_pk_split_text.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()