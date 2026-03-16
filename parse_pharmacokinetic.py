import time
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import Optional, List
import json
import re

# Определяем Pydantic модели для структурированного вывода
class AbsorptionInfo(BaseModel):
    """Информация о всасывании препарата"""
    tc_max: Optional[str] = Field(description="Время достижения максимальной концентрации (TCmax/Tmax)")
    absorption_site: Optional[str] = Field(description="Где происходит всасывание (например, ЖКТ, желудок, тонкий кишечник)")
    has_tmax: bool = Field(description="Упоминается ли Tmax или TCmax в тексте")

class DistributionInfo(BaseModel):
    """Информация о распределении препарата"""
    protein_binding: Optional[str] = Field(description="Процент связывания с белками плазмы")
    has_protein_binding: bool = Field(description="Упоминается ли связь с белками")
    protein_binding_percentage: Optional[float] = Field(description="Числовое значение процента связывания, если есть")

class MetabolismInfo(BaseModel):
    """Информация о метаболизме препарата"""
    metabolites: List[str] = Field(description="Список метаболитов")
    cytochromes: List[str] = Field(description="Список цитохромов (CYP450)")
    has_cytochromes: bool = Field(description="Упоминаются ли цитохромы")

class ExcretionInfo(BaseModel):
    """Информация о выведении препарата"""
    excretion_paths: List[str] = Field(description="Пути выведения (почечный, печеночный и т.д.)")
    half_life: Optional[str] = Field(description="Период полувыведения (T½)")
    has_half_life: bool = Field(description="Упоминается ли T½")

# Класс для парсинга JSON из ответов модели
class JSONOutputParser:
    """Парсер для извлечения JSON из текстового ответа"""
    
    @staticmethod
    def parse(text: str) -> dict:
        """Извлекает JSON из текста"""
        # Ищем JSON в тексте
        json_pattern = r'\{[^{}]*\}'
        matches = re.findall(json_pattern, text)
        
        for match in matches:
            try:
                return json.loads(match)
            except:
                continue
        
        # Если не нашли JSON, пробуем другой паттерн
        json_pattern_with_nesting = r'\{[^{}]*+(?:[^{}]*+)*\}'
        matches = re.findall(json_pattern_with_nesting, text, re.DOTALL)
        
        for match in matches:
            try:
                return json.loads(match)
            except:
                continue
        
        return {}

# Основной класс для обработки данных
class PKDataExtractor:
    def __init__(self, model_name: str = "ilyagusev/saiga_llama3", base_url: str = "http://localhost:11434"):
        """
        Инициализация экстрактора с моделью через Ollama
        :param model_name: название модели в Ollama
        :param base_url: URL сервера Ollama (по умолчанию локальный)
        """
        print(f"Подключение к Ollama модели: {model_name}")
        print(f"Сервер Ollama: {base_url}")
        
        # Проверяем доступность Ollama
        try:
            # Создаем соединение с Ollama с улучшенными параметрами
            self.llm = Ollama(
                model=model_name,
                base_url=base_url,
                temperature=0.1,
                num_predict=512,  # Увеличил лимит
                stop=None,  # Убрал стоп-слова для получения полного ответа
                top_k=10,
                top_p=0.95
            )
            
            
            # Тестовый запрос для проверки соединения
            test_response = self.llm.invoke("Тест. Ответь 'ok' одним словом.")
            print(f"✓ Соединение с Ollama установлено. Ответ: {test_response[:50]}")
            
        except Exception as e:
            print(f"✗ Ошибка подключения к Ollama: {e}")
            print("\nУбедитесь что:")
            print("  1. Ollama установлен: https://ollama.com/")
            print("  2. Сервер Ollama запущен: 'ollama serve'")
            print("  3. Модель загружена: 'ollama pull ilyagusev/saiga_llama3'")
            print("  4. Сервер доступен по адресу http://localhost:11434")
            raise
        
        # Инициализируем парсер JSON
        self.json_parser = JSONOutputParser()
        
        print("Экстрактор готов к работе!")
        
        # Шаблоны промптов для каждой категории
        self.prompts = {
            "всасывание": PromptTemplate(
                template="""Ты - эксперт по фармакокинетике. Извлеки информацию о всасывании препарата из текста.

Текст: {text}

Проанализируй текст и верни JSON с полями:
- tc_max: время достижения максимальной концентрации (TCmax/Tmax), если указано
- absorption_site: место всасывания (например, ЖКТ, желудок, тонкий кишечник)
- has_tmax: true/false - упоминается ли Tmax или TCmax

Пример ответа:
{{"tc_max": "2 часа", "absorption_site": "верхние отделы ЖКТ", "has_tmax": true}}

Важно: Верни ТОЛЬКО JSON без дополнительного текста.
""",
                input_variables=["text"]
            ),
            
            "распределение": PromptTemplate(
                template="""Ты - эксперт по фармакокинетике. Извлеки информацию о распределении препарата из текста.

Текст: {text}

Проанализируй текст и верни JSON с полями:
- protein_binding: информация о связи с белками
- has_protein_binding: true/false - упоминается ли связь с белками
- protein_binding_percentage: числовое значение процента связывания (если есть)

Пример ответа:
{{"protein_binding": "90%", "has_protein_binding": true, "protein_binding_percentage": 90.0}}

Важно: Верни ТОЛЬКО JSON без дополнительного текста.
""",
                input_variables=["text"]
            ),
            
            "метаболизм": PromptTemplate(
                template="""Ты - эксперт по фармакокинетике. Извлеки информацию о метаболизме препарата из текста.

Текст: {text}

Проанализируй текст и верни JSON с полями:
- metabolites: список метаболитов
- cytochromes: список цитохромов (CYP450)
- has_cytochromes: true/false - упоминаются ли цитохромы

Пример ответа:
{{"metabolites": ["оксипуринол"], "cytochromes": ["CYP3A4"], "has_cytochromes": true}}

Важно: Верни ТОЛЬКО JSON без дополнительного текста.
""",
                input_variables=["text"]
            ),
            
            "выведение": PromptTemplate(
                template="""Ты - эксперт по фармакокинетике. Извлеки информацию о выведении препарата из текста.

Текст: {text}

Проанализируй текст и верни JSON с полями:
- excretion_paths: список путей выведения (почечный, печеночный и т.д.)
- half_life: период полувыведения (T½)
- has_half_life: true/false - упоминается ли T½

Пример ответа:
{{"excretion_paths": ["почечный"], "half_life": "2-3 часа", "has_half_life": true}}

Важно: Верни ТОЛЬКО JSON без дополнительного текста.
""",
                input_variables=["text"]
            )
        }
        
        # Pydantic модели для валидации
        self.models = {
            "всасывание": AbsorptionInfo,
            "распределение": DistributionInfo,
            "метаболизм": MetabolismInfo,
            "выведение": ExcretionInfo
        }
    
    def process_data(self, json_data: List[dict], max_retries: int = 3, retry_delay: float = 1.0) -> List[dict]:
        """
        Обрабатывает массив JSON данных и извлекает информацию с повторными попытками
        :param json_data: список словарей с данными
        :param max_retries: максимальное количество попыток
        :param retry_delay: задержка между попытками в секундах
        :return: список результатов с извлеченной информацией (исходные объекты с добавленным extracted_info)
        """
        results = []
        
        for i, item in enumerate(json_data):
            print(f"\nОбработка записи {i+1}/{len(json_data)}...")
            
            # Фильтруем только pk секцию
            if item.get("section") != "pk":
                print(f"  ↳ Пропущено: section = {item.get('section')} (ожидается 'pk')")
                # Всё равно добавляем в результаты, но без обработки
                results.append(item)
                continue
                
            category = item.get("category")
            text = item.get("text")
            
            if category not in self.prompts:
                print(f"  ↳ Пропущено: неподдерживаемая категория '{category}'")
                results.append(item)
                continue
            
            print(f"  ↳ Категория: {category}")
            print(f"  ↳ Текст: {text[:100]}...")
            
            # Создаем копию объекта, чтобы не изменять оригинал
            result_item = item.copy()
            
            # Переменные для отслеживания попыток
            attempt = 1
            last_error = None
            response = None
            
            while attempt <= max_retries:
                try:
                    print(f"\n  {'─'*40}")
                    print(f"  Попытка {attempt}/{max_retries}")
                    print(f"  {'─'*40}")
                    
                    # Получаем промпт для категории
                    prompt = self.prompts[category]
                    
                    # Формируем запрос
                    formatted_prompt = prompt.format(text=text)
                    
                    print(f"  Отправка запроса в Ollama...")
                    
                    # Получаем ответ от модели
                    response = self.llm.invoke(formatted_prompt)
                    
                    print(f"  Получен ответ (длина: {len(response)} символов)")
                    print(f"  Первые 200 символов ответа:")
                    print(f"  {response[:200]}")
                    
                    if len(response) < 10:
                        print(f"  ⚠️ Подозрительно короткий ответ: '{response}'")
                    
                    # Парсим JSON из ответа
                    parsed_json = self.json_parser.parse(response)
                    
                    # Проверяем, что распарсили не пустой JSON
                    if not parsed_json:
                        raise ValueError("Получен пустой JSON после парсинга")
                    
                    # Валидируем с помощью Pydantic
                    model_class = self.models[category]
                    validated_info = model_class(**parsed_json)
                    
                    # Добавляем extracted_info в объект
                    result_item["extracted_info"] = validated_info.dict()
                    
                    print(f"  ✓ Успешно обработано за {attempt} попытку(и)")
                    print(f"  ✓ Извлечено: {validated_info.dict()}")
                    break  # Выходим из цикла попыток при успехе
                    
                except Exception as e:
                    last_error = str(e)
                    print(f"    ✗ Ошибка в попытке {attempt}: {last_error}")
                    
                    # Если это не последняя попытка, ждем и пробуем снова
                    if attempt < max_retries:
                        print(f"    ↳ Повтор через {retry_delay} секунд...")
                        time.sleep(retry_delay)
                    
                    attempt += 1
            
            # Если все попытки исчерпаны
            if attempt > max_retries:
                print(f"\n  ✗ ВСЕ {max_retries} ПОПЫТОК ИСЧЕРПАНЫ")
                
                # Добавляем информацию об ошибке
                result_item["extracted_info"] = "error"
            
            results.append(result_item)
        
        return results
        
# Функция для сохранения результатов
def save_results(results: List[dict],  filename: str = "pk_extraction_results.json"):
    """
    Сохраняет результаты в JSON файл
    """
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nРезультаты сохранены в {filename}")

# Пример использования
def main():

    DATA_FILENAME = "data\\classification_results.json"
    with open(DATA_FILENAME, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Инициализация экстрактора
    print("Инициализация экстрактора с моделью ilyagusev/saiga_llama3...")
    extractor = PKDataExtractor(model_name="ilyagusev/saiga_llama3")
    
    # Обработка данных
    results = extractor.process_data(data)
    
    # Вывод результатов
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ ИЗВЛЕЧЕНИЯ:")
    print("="*60)
    
    for result in results:
        print(f"\nИндекс: {result['index']}")
        print(f"Категория: {result['category']}")
        print(f"Текст: {result['text'][:50]}...")
        print("Извлеченная информация:")
        
        if "error" in result:
            print(f"  ОШИБКА: {result['error']}")
        else:
            for key, value in result["extracted_info"].items():
                print(f"  {key}: {value}")
    
    
    # Сохранение результатов
    save_results(results, "data\\pk_extraction_results.json")

if __name__ == "__main__":
    main()