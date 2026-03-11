import asyncio
from typing import List, Optional, Dict, Any
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import json
import time

from data_model import Sentence

# Pydantic модель для результата классификации
class ClassificationResult(BaseModel):
    category: str = Field(description="Категория предложения: всасывание, распределение, метаболизм, выведение или другое")
    confidence: float = Field(description="Уверенность в классификации от 0 до 1", ge=0, le=1)
    reasoning: str = Field(description="Краткое объяснение почему выбрана эта категория")

class SentenceClassifier:

    CATEGORIES = ["всасывание", "распределение", "метаболизм", "выведение", "другое"]

    def __init__(self, model_name: str = "ilyagusev/saiga_llama3", base_url: str = "http://localhost:11434"):
        """
        Инициализация классификатора с использованием Ollama
        
        Args:
            model_name: имя модели в Ollama
            base_url: URL сервера Ollama
        """
        self.llm = Ollama(
            model=model_name,
            base_url=base_url,
            temperature=0.1,  # Низкая температура для более детерминированных ответов
            num_predict=256,   # Максимальная длина ответа
            top_p=0.9,
        )
        
        self.parser = PydanticOutputParser(pydantic_object=ClassificationResult)
        
        # Создаем промпт для классификации
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Ты - эксперт в фармакокинетике. Классифицируй предложения по одной из пяти категорий.

Категории классификации:
- всасывание: процессы, связанные с поступлением препарата в организм (абсорбция, всасывание из ЖКТ, биодоступность, максимальная концентрация (Cmax))
- распределение: процессы, связанные с распространением препарата в организме (связывание с белками, объем распределения, проникновение в ткани)
- метаболизм: процессы биотрансформации препарата (образование метаболитов, печеночный метаболизм)
- выведение: процессы удаления препарата из организма (элиминация, период полувыведения, почечный клиренс)
- другое: если предложение не относится к фармакокинетике или содержит общую информацию

Ответ должен быть строго в формате JSON:
{{"category": "название категории", "confidence": 0.0-1.0, "reasoning": "объяснение"}}"""),
            ("human", "Предложение: {text}")
        ])
        
        # Создаем цепочку
        self.chain = self.prompt | self.llm
        
    def classify_sentence(self, sentence: Sentence) -> Sentence:
        """
        Классифицирует одно предложение и возвращает обновленный объект Sentence
        
        Args:
            sentence: объект Sentence для классификации
            
        Returns:
            Sentence с заполненными полями category, confidence, reasoning
        """
        start_time = time.time()
        
        # ИСПРАВЛЕНИЕ 1: Сохраняем исходный текст для повторных попыток
        original_text = sentence.text
        
        for attempt in range(3):
            try:
                # Получаем ответ от модели
                response = self.chain.invoke({"text": original_text})  # Используем original_text вместо sentence.text
                
                try:
                    # Пробуем найти JSON в ответе
                    json_start = response.find('{')
                    json_end = response.rfind('}') + 1
                    
                    # ИСПРАВЛЕНИЕ 2: Проверяем, что нашли валидный JSON
                    if json_start != -1 and json_end > json_start:
                        json_str = response[json_start:json_end]
                        result_dict = json.loads(json_str)
                    else:
                        # Если не нашли JSON, пробуем распарсить весь ответ
                        result_dict = json.loads(response)
                    
                    # Получаем категорию из ответа
                    category = result_dict.get("category", "").lower()
                    
                    # ИСПРАВЛЕНИЕ 3: Проверяем наличие всех необходимых полей
                    if category in self.CATEGORIES and "confidence" in result_dict and "reasoning" in result_dict:
                        sentence.category = category
                        sentence.confidence = float(result_dict.get("confidence", 0.5))
                        sentence.reasoning = result_dict.get("reasoning", "")
                        return sentence  # Успешно - возвращаем результат
                    else:
                        print(f"  Предложение {sentence.index}: неполный ответ, попытка {attempt + 1}/3")
                        if attempt == 2:  # Последняя попытка
                            sentence.category = "другое"
                            sentence.confidence = 0.3
                            sentence.reasoning = f"Модель вернула неполные данные: {response[:200]}"
                        
                except json.JSONDecodeError:
                    print(f"  Предложение {sentence.index}: ошибка JSON, попытка {attempt + 1}/3")
                    if attempt == 2:  # Последняя попытка
                        # ИСПРАВЛЕНИЕ 4: Используем улучшенное извлечение категории из текста
                        category, confidence = self._extract_category_from_text(response)
                        sentence.category = category
                        sentence.confidence = confidence
                        sentence.reasoning = response[:200]
                        
            except Exception as e:
                print(f"  Предложение {sentence.index}: ошибка {str(e)}, попытка {attempt + 1}/3")
                if attempt == 2:  # Последняя попытка
                    sentence.category = "другое"
                    sentence.confidence = 0.0
                    sentence.reasoning = f"Ошибка: {str(e)}"
                    
        return sentence
    
    def _extract_category_from_text(self, text: str) -> tuple[str, float]:
        """
        Извлекает категорию из текстового ответа, если JSON не удалось распарсить
        Возвращает кортеж (категория, уверенность)
        """
        text_lower = text.lower()
        
        # Ищем упоминания категорий
        for category in self.CATEGORIES:
            if category in text_lower:
                # Пытаемся найти уверенность в тексте
                confidence = 0.5  # Базовая уверенность
                import re
                # Ищем числа от 0 до 1 или проценты
                confidence_match = re.search(r'уверенн[оа]ст[ьи][\s:]+(\d*\.?\d+)', text_lower)
                if confidence_match:
                    try:
                        conf_value = float(confidence_match.group(1))
                        if 0 <= conf_value <= 1:
                            confidence = conf_value
                        elif conf_value > 1:  # Может быть процент
                            confidence = conf_value / 100
                    except ValueError:
                        pass
                return category, confidence
        
        return "другое", 0.3
    
class BatchSentenceProcessor:
    def __init__(self, classifier: SentenceClassifier, batch_size: int = 5):
        """
        Обработчик для пакетной классификации предложений
        
        Args:
            classifier: экземпляр SentenceClassifier
            batch_size: размер пакета для обработки
        """
        self.classifier = classifier
        self.batch_size = batch_size
    
    def process_sentences(self, sentences: List[Sentence]) -> List[Sentence]:
        """
        Обрабатывает список предложений (синхронно)
        """
        processed_sentences = []
        total_batches = (len(sentences) + self.batch_size - 1) // self.batch_size
        
        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            print(f"Обработка пакета {batch_num}/{total_batches}")
            
            for sentence in batch:
                processed = self.classifier.classify_sentence(sentence)
                processed_sentences.append(processed)
                print(f"  Предложение {sentence.index}: {processed.category} "
                      f"(уверенность: {processed.confidence:.2f}, "
                    )
                
        return processed_sentences

def print_classification_summary(sentences: List[Sentence]):
    """
    Выводит сводку по классификации
    """
    categories = {}
    total_time = 0
    valid_sentences = 0
    
    for sentence in sentences:
        if sentence.category not in categories:
            categories[sentence.category] = []
        categories[sentence.category].append(sentence.index)
    
    print("\n" + "="*50)
    print("СВОДКА ПО КЛАССИФИКАЦИИ")
    print("="*50)
    
    # ИСПРАВЛЕНИЕ 6: Сортируем категории для красивого вывода
    for category in SentenceClassifier.CATEGORIES:
        indices = categories.get(category, [])
        print(f"\n{category.upper()}: {len(indices)} предложений")
        if indices:
            print(f"  Индексы: {sorted(indices)}")
    
    if valid_sentences > 0:
        print(f"\nСреднее время обработки: {total_time/valid_sentences:.2f}с")
    
    print("\n" + "="*50)
    print("ДЕТАЛЬНАЯ КЛАССИФИКАЦИЯ")
    print("="*50)
    
    # ИСПРАВЛЕНИЕ 7: Сортируем предложения по индексу для вывода
    for sentence in sorted(sentences, key=lambda x: x.index):
        print(f"\nПредложение {sentence.index}:")
        # Обрезаем длинный текст для вывода
        text_preview = sentence.text[:100] + "..." if len(sentence.text) > 100 else sentence.text
        print(f"  Текст: {text_preview}")
        print(f"  Категория: {sentence.category}")
        print(f"  Уверенность: {sentence.confidence:.2f}")
        if sentence.reasoning:
            reasoning_preview = sentence.reasoning[:150] + "..." if len(sentence.reasoning) > 150 else sentence.reasoning
            print(f"  Объяснение: {reasoning_preview}")

# Пример использования
async def main():
    # Ваши данные (сокращенный пример)
    sentences = [
        Sentence(text='Всасывание Аллопуринол активен при пероральном применении.', index=0, start_char=0, end_char=58, length=58, category=None, confidence=None, reasoning=None, section='pk'),
        Sentence(text='Он быстро всасывается из верхних отделов желудочно-кишечного тракта (ЖКТ).', index=1, start_char=59, end_char=133, length=74, category=None, confidence=None, reasoning=None, section='pk'),
        Sentence(text='По данным фармакокинетических исследований аллопуринол определяется в крови уже через 30 - 60 минут после приема.', index=2, start_char=134, end_char=247, length=113, category=None, confidence=None, reasoning=None, section='pk'),
        Sentence(text='Биодоступность аллопуринола варьирует от 67 % до 90 %.', index=3, start_char=248, end_char=302, length=54, category=None, confidence=None, reasoning=None, section='pk',),
        Sentence(text='Максимальная концентрация препарата в плазме крови (TCmax) как правило регистрируется приблизительно через 1,5 часа после перорального приема.', index=4, start_char=303, end_char=445, length=142, category=None, confidence=None, reasoning=None, section='pk'),
        Sentence(text='Затем концентрация аллопуринола быстро снижается.', index=5, start_char=446, end_char=495, length=49, category=None, confidence=None, reasoning=None, section='pk'),
        Sentence(text='Спустя 6 часов приема, в плазме крови определяется лишь следовая концентрация препарата.', index=6, start_char=496, end_char=584, length=88, category=None, confidence=None, reasoning=None, section='pk'),
        Sentence(text='Максимальная концентрация (Cmax) активного метаболита – оксипуринола обычно регистрируется через 3 - 5 часов после перорального приема аллопуринола.', index=7, start_char=585, end_char=738, length=153, category=None, confidence=None, reasoning=None, section='pk'),
        Sentence(text='Концентрация оксипуринола в плазме крови снижается значительно медленнее', index=8, start_char=739, end_char=812, length=73, category=None, confidence=None, reasoning=None, section='pk'),
        Sentence(text='Распределение Аллопуринол почти не связывается с белками плазмы крови, поэтому изменения степени связывания с белками не должны оказывать значительного влияния на клиренс препарата.', index=9, start_char=813, end_char=994, length=181, category=None, confidence=None, reasoning=None, section='pk'),
        Sentence(text='Кажущийся объем распределения (Vd) аллопуринола составляет приблизительно 1,6 литра/кг, что говорит о достаточно выраженном поглощении препарата тканями.', index=10, start_char=995, end_char=1148, length=153, category=None, confidence=None, reasoning=None, section='pk'),
    ]
    
    # Инициализация классификатора
    classifier = SentenceClassifier(
        model_name="ilyagusev/saiga_llama3",  # Убедитесь, что модель загружена в Ollama
        base_url="http://localhost:11434"
    )
    
    # Создание обработчика
    processor = BatchSentenceProcessor(classifier, batch_size=3)
    
    # Обработка предложений 
    print("Начало обработки предложений...")
    processed_sentences = processor.process_sentences(sentences)
    
    # Вывод результатов
    print_classification_summary(processed_sentences)
    
    # Сохранение результатов в файл
    save_results_to_file(processed_sentences, "classification_results.json")

def save_results_to_file(sentences: List[Sentence], filename: str):
    """
    Сохраняет результаты классификации в JSON файл
    """
    results = []
    for sentence in sentences:
        results.append({
            "index": sentence.index,
            "text": sentence.text,
            "category": sentence.category,
            "confidence": sentence.confidence,
            "reasoning": sentence.reasoning,
            "section": sentence.section,
            "start_char": sentence.start_char,
            "end_char": sentence.end_char,
            "length": sentence.length
        })
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nРезультаты сохранены в файл: {filename}")

# Запуск
if __name__ == "__main__":
    asyncio.run(main())