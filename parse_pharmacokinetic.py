import json
import re
import time
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

class PKDataExtractor:
    def __init__(self, model_name="ilyagusev/saiga_llama3", base_url="http://localhost:11434"):
        self.llm = Ollama(
            model=model_name,
            base_url=base_url,
            temperature=0.1,
            num_predict=256
        )
        
        # Промпты для каждой категории
        self.prompts = {
            "всасывание": PromptTemplate(
                template="""Из текста про всасывание препарата найди:
1. tc_max: время достижения максимальной концентрации (Tmax/TCmax). Если не указано - "false"
2. absorption_site: место всасывания (ЖКТ, желудок и т.д.). Если не указано - "false"

Текст: {text}

Ответь ТОЛЬКО в формате JSON:
{{"tc_max": "значение или false", "absorption_site": "значение или false"}}""",
                input_variables=["text"]
            ),
            
            "распределение": PromptTemplate(
                template="""Из текста про распределение препарата найди:
protein_binding_percentage: процент связывания с белками. Если не указано - "false"

Текст: {text}

Ответь ТОЛЬКО в формате JSON:
{{"protein_binding_percentage": "значение или false"}}""",
                input_variables=["text"]
            ),
            
            "метаболизм": PromptTemplate(
                template="""Из текста про метаболизм препарата найди:
cytochromes: список цитохромов (CYP3A4, CYP2D6 и т.д.). Если не указано - "false"

Текст: {text}

Ответь ТОЛЬКО в формате JSON:
{{"cytochromes": "список или false"}}""",
                input_variables=["text"]
            ),
            
            "выведение": PromptTemplate(
                template="""Из текста про выведение препарата найди:
1. excretion_paths: пути выведения (почечный, печеночный и т.д.). Если не указано - "false"
2. half_life: период полувыведения (T½). Если не указано - "false"

Текст: {text}

Ответь ТОЛЬКО в формате JSON:
{{"excretion_paths": "значение или false", "half_life": "значение или false"}}""",
                input_variables=["text"]
            )
        }
    
    def extract_json(self, text):
        """Извлекает JSON из ответа модели"""
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                return {}
        return {}
    
    def process_sentence(self, sentence, category):
        """Обрабатывает одно предложение с 3 попытками"""
        if category not in self.prompts:
            return {"sentence": sentence, "category": category}
        
        prompt = self.prompts[category].format(text=sentence)
        
        for attempt in range(3):
            try:
                print(f"  Попытка {attempt + 1}/3 для категории '{category}'")
                response = self.llm.invoke(prompt)
                extracted = self.extract_json(response)
                
                # Проверяем, что получили не пустой JSON
                if extracted:
                    return {
                        "sentence": sentence,
                        "category": category,
                        "extracted": extracted
                    }
                else:
                    print(f"    Пустой JSON, повтор через 1 сек...")
                    time.sleep(1)
                    
            except Exception as e:
                print(f"    Ошибка: {e}, повтор через 1 сек...")
                time.sleep(1)
        
        # Если все попытки неудачные
        print(f"    Все попытки исчерпаны для категории '{category}'")
        return {
            "sentence": sentence,
            "category": category,
            "extracted": {}
        }
    
    def process_drug(self, drug_data):
        """Обрабатывает один препарат"""
        result = {
            "drug": drug_data["drug"],
            "pd": drug_data["pd"],
            "pk": []
        }
        
        print(f"\nОбработка {drug_data['drug']}:")
        for item in drug_data["pk"]:
            result["pk"].append(self.process_sentence(item["sentence"], item["category"]))
        
        return result

def main():
    # Загрузка данных
    with open("data\\classified_drugs.json", "r", encoding="utf-8") as f:
        drugs_data = json.load(f)
    
    # Инициализация экстрактора
    extractor = PKDataExtractor()
    
    # Обработка всех препаратов
    results = []
    for drug in drugs_data:
        results.append(extractor.process_drug(drug))
    
    # Сохранение результатов
    with open("data\\pk_extraction_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nГотово! Обработано препаратов: {len(results)}")
    print("Результаты сохранены в pk_extraction_results.json")

if __name__ == "__main__":
    main()