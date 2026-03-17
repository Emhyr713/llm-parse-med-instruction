import json
import re
import time
from collections import defaultdict
from typing import Dict, List, Any, Optional
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

class PKDataExtractor:
    """
    Улучшенный экстрактор фармакокинетических параметров из текстов,
    размеченных по категориям (всасывание, распределение, метаболизм, выведение).
    Использует few-shot промпты, группировку предложений и валидацию результатов.
    """

    def __init__(self, model_name: str = "ilyagusev/saiga_llama3", base_url: str = "http://localhost:11434"):
        self.llm = Ollama(
            model=model_name,
            base_url=base_url,
            temperature=0.1,
            num_predict=512          # Увеличиваем для сложных ответов
        )
        self.prompts = self._build_prompts()

    def _build_prompts(self) -> Dict[str, PromptTemplate]:
        """
        Создаёт словарь промптов для каждой категории с few-shot примерами.
        """
        prompts = {}

        # ---- Всасывание ----
        prompts["всасывание"] = PromptTemplate(
            template="""Ты — фармакокинетический эксперт. Из текста(ов) о всасывании препарата извлеки следующие данные:
- tc_max: время достижения максимальной концентрации (Tmax, TCmax, время достижения Cmax). Укажи значение с единицами измерения (часы, минуты). Если указан диапазон (например, 2-4 часа), сохрани его как есть. Если точное время не указано (есть только качественные оценки "быстро", "медленно"), ставь "false". Если не указано — "false".
- absorption_site: место всасывания (например, "желудочно-кишечный тракт", "ЖКТ", "тонкая кишка", "двенадцатиперстная кишка", "под язык", "перорально", "внутрь"). Если не указано — "false".

Примеры:
Текст: "Максимальная концентрация в плазме крови достигается через 2-4 часа."
Ответ: {{"tc_max": "2-4 часа", "absorption_site": "false"}}

Текст: "Препарат быстро всасывается из желудочно-кишечного тракта."
Ответ: {{"tc_max": "false", "absorption_site": "ЖКТ"}}

Текст: "Cmax регистрируется приблизительно через 1,5 часа после приема внутрь."
Ответ: {{"tc_max": "1,5 часа", "absorption_site": "внутрь"}}

Текст: "Абсорбция происходит в верхних отделах тонкой кишки. TCmax составляет 3-5 часов."
Ответ: {{"tc_max": "3-5 часов", "absorption_site": "верхние отделы тонкой кишки"}}

Теперь обработай следующий текст (может содержать несколько предложений). Отвечай ТОЛЬКО JSON без пояснений:
Текст: {text}""",
            input_variables=["text"]
        )

        # ---- Распределение ----
        prompts["распределение"] = PromptTemplate(
            template="""Ты — фармакокинетический эксперт. Из текста(ов) о распределении препарата извлеки следующие данные:
- protein_binding_percentage: процент связывания с белками плазмы крови. Укажи только число (или диапазон) с символом процента, например "95%", "90-95%". Если указано "более 95%" — запиши ">95%". Если нет числового значения — "false". Если указано "высокая степень" или подобное без цифр — "false".

Примеры:
Текст: "Связь с белками плазмы крови составляет 99%."
Ответ: {{"protein_binding_percentage": "99%"}}

Текст: "Около 97,5 % циркулирующего препарата связано с белками."
Ответ: {{"protein_binding_percentage": "97.5%"}}

Текст: "Препарат слабо связывается с белками (около 20%)."
Ответ: {{"protein_binding_percentage": "20%"}}

Текст: "Имеет высокую степень связывания с белками плазмы."
Ответ: {{"protein_binding_percentage": "false"}}

Теперь обработай следующий текст (может содержать несколько предложений). Отвечай ТОЛЬКО JSON без пояснений:
Текст: {text}""",
            input_variables=["text"]
        )

        # ---- Метаболизм ----
        prompts["метаболизм"] = PromptTemplate(
            template="""Ты — фармакокинетический эксперт. Из текста(ов) о метаболизме препарата извлеки следующие данные:
- cytochromes: список изоферментов цитохрома P450, участвующих в метаболизме (например, CYP3A4, CYP2D6, CYP2C19). Перечисли их через запятую. Если указаны синонимы (например, "CYP3A4/5" — запиши как "CYP3A4, CYP3A5"). Если не указано — "false".

Примеры:
Текст: "Метаболизируется в печени с участием изоферментов CYP3A4 и CYP2C8."
Ответ: {{"cytochromes": "CYP3A4, CYP2C8"}}

Текст: "Основную роль играет CYP2D6, в меньшей степени — CYP3A4."
Ответ: {{"cytochromes": "CYP2D6, CYP3A4"}}

Текст: "Не метаболизируется в печени."
Ответ: {{"cytochromes": "false"}}

Теперь обработай следующий текст (может содержать несколько предложений). Отвечай ТОЛЬКО JSON без пояснений:
Текст: {text}""",
            input_variables=["text"]
        )

        # ---- Выведение ----
        prompts["выведение"] = PromptTemplate(
            template="""Ты — фармакокинетический эксперт. Из текста(ов) о выведении препарата извлеки следующие данные:
- excretion_paths: пути выведения (например, "почечный", "печеночный", "кишечный", "желчный", можно комбинировать через запятую). Если указано несколько путей, перечисли их. Если не указано — "false".
- half_life: период полувыведения (T½, T1/2, период полувыведения). Укажи значение с единицами измерения (часы, минуты). Если для разных веществ указаны разные значения (например, для исходного вещества и метаболита), укажи их в формате "вещество1: значение1, вещество2: значение2". Если нет числового значения — "false".

Примеры:
Текст: "Выводится почками в неизмененном виде, период полувыведения составляет 2-3 часа."
Ответ: {{"excretion_paths": "почечный", "half_life": "2-3 часа"}}

Текст: "Около 70% выводится через кишечник, 30% — почками. T1/2 — 12 ч."
Ответ: {{"excretion_paths": "кишечный, почечный", "half_life": "12 ч"}}

Текст: "T1/2 аллопуринола составляет 1-2 часа, оксипуринола — 13-30 часов."
Ответ: {{"excretion_paths": "false", "half_life": "аллопуринол: 1-2 часа, оксипуринол: 13-30 часов"}}

Текст: "Выводится медленно, преимущественно с желчью."
Ответ: {{"excretion_paths": "желчный", "half_life": "false"}}

Теперь обработай следующий текст (может содержать несколько предложений). Отвечай ТОЛЬКО JSON без пояснений:
Текст: {text}""",
            input_variables=["text"]
        )

        # ---- Другое ----
        # Для категории "другое" просто возвращаем пустой словарь (ничего не извлекаем)
        prompts["другое"] = PromptTemplate(
            template="""Для категории "другое" данные не извлекаются. Верни пустой JSON: {{}}""",
            input_variables=[]
        )

        return prompts

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Извлекает JSON из ответа модели (первый объект в фигурных скобках)."""
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return {}
        return {}

    @staticmethod
    def _validate(extracted: dict, category: str) -> bool:
        """
        Простейшая валидация: проверяем, что для ожидаемых полей
        значение не является пустой строкой (если не false).
        """
        expected_fields = {
            "всасывание": ["tc_max", "absorption_site"],
            "распределение": ["protein_binding_percentage"],
            "метаболизм": ["cytochromes"],
            "выведение": ["excretion_paths", "half_life"],
            "другое": []
        }
        fields = expected_fields.get(category, [])
        for field in fields:
            if field in extracted and extracted[field] != "false":
                # Допустим, что поле не пустое и не состоит только из пробелов
                if isinstance(extracted[field], str) and extracted[field].strip() == "":
                    return False
        return True

    def _extract_from_group(self, texts: List[str], category: str, max_attempts: int = 3) -> dict:
        """
        Извлекает данные из группы предложений одной категории.
        Если после нескольких попыток не удаётся получить валидный JSON, возвращает пустой словарь.
        """
        if category not in self.prompts:
            return {}

        # Объединяем предложения в один текст
        combined_text = "\n".join(texts).strip()
        if not combined_text:
            return {}

        prompt = self.prompts[category]
        # Для категории "другое" можно сразу вернуть пустой результат
        if category == "другое":
            return {}

        for attempt in range(max_attempts):
            try:
                print(f"    Попытка {attempt+1}/{max_attempts} для категории '{category}' ({len(texts)} предл.)")
                response = self.llm.invoke(prompt.format(text=combined_text))
                extracted = self._extract_json(response)
                if extracted and self._validate(extracted, category):
                    return extracted
                else:
                    print(f"    Невалидный ответ (попытка {attempt+1}): {response[:200]}...")
                    # Добавляем в промпт инструкцию исправить ошибку
                    if attempt < max_attempts - 1:
                        prompt_text = prompt.format(text=combined_text)
                        prompt_text += "\n\nПредыдущий ответ не содержал корректного JSON или данные неполны. Пожалуйста, верни только JSON в правильном формате."
                        prompt = PromptTemplate(template=prompt_text, input_variables=[])
                    time.sleep(1)
            except Exception as e:
                print(f"    Ошибка при вызове модели: {e}")
                time.sleep(1)

        # Если все попытки исчерпаны, возвращаем пустой словарь
        print(f"    Все попытки исчерпаны для категории '{category}'")
        return {}

    def process_drug(self, drug_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Обрабатывает один препарат:
        - группирует предложения PK по категориям
        - для каждой категории отправляет объединённый текст в модель
        - возвращает результат с извлечёнными данными
        """
        drug_name = drug_data["drug"]
        print(f"\nОбработка {drug_name}:")

        # Группируем предложения по категориям
        grouped = defaultdict(list)
        for item in drug_data.get("pk", []):
            sentence = item.get("sentence", "").strip()
            category = item.get("category", "другое")
            if sentence:
                grouped[category].append(sentence)

        # Извлекаем данные для каждой категории
        pk_results = []
        for category, sentences in grouped.items():
            extracted = self._extract_from_group(sentences, category)
            pk_results.append({
                "category": category,
                "extracted": extracted,
                # Можно сохранить исходные предложения для отслеживания, но не обязательно
                # "sentences": sentences
            })

        # Формируем результат
        result = {
            "drug": drug_name,
            "pd": drug_data.get("pd", []),
            "pk": pk_results
        }
        return result

def main():
    # Пути к файлам
    input_file = "data/classified_drugs.json"   # исходный файл с размеченными предложениями
    output_file = "data/pk_extraction_results.json"
    log_file = "data/failed_sentences.log"      # для логирования проблемных случаев (опционально)

    # Загрузка данных
    with open(input_file, "r", encoding="utf-8") as f:
        drugs_data = json.load(f)

    # Инициализация экстрактора
    extractor = PKDataExtractor()

    # Обработка всех препаратов
    results = []
    for drug in drugs_data:
        try:
            result = extractor.process_drug(drug)
            results.append(result)
        except Exception as e:
            print(f"Критическая ошибка при обработке {drug.get('drug', 'Unknown')}: {e}")
            # Добавляем запись с ошибкой, чтобы не потерять данные
            results.append({
                "drug": drug.get("drug", "Unknown"),
                "pd": drug.get("pd", []),
                "pk": [],
                "error": str(e)
            })

    # Сохранение результатов
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nГотово! Обработано препаратов: {len(results)}")
    print(f"Результаты сохранены в {output_file}")

if __name__ == "__main__":
    main()