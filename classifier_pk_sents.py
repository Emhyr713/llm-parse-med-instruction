import asyncio
from typing import List, Optional, Dict, Any
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import json
import time
import re

# Pydantic модель для результата классификации
class ClassificationResult(BaseModel):
    category: str = Field(description="Категория предложения: всасывание, распределение, метаболизм, выведение или другое")
    confidence: float = Field(description="Уверенность в классификации от 0 до 1", ge=0, le=1)
    reasoning: str = Field(description="Краткое объяснение почему выбрана эта категория")

class SentenceClassifier:
    CATEGORIES = ["всасывание", "распределение", "метаболизм", "выведение", "другое"]

    def __init__(self, model_name: str = "ilyagusev/saiga_llama3", base_url: str = "http://localhost:11434"):
        self.llm = Ollama(
            model=model_name,
            base_url=base_url,
            temperature=0.1,
            num_predict=256,
            top_p=0.9,
        )
        
        self.parser = PydanticOutputParser(pydantic_object=ClassificationResult)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Ты - эксперт в фармакокинетике. Классифицируй предложения по одной из пяти категорий.

Категории классификации:
- всасывание: процессы, связанные с поступлением препарата в организм (абсорбция, всасывание из ЖКТ, биодоступность, максимальная концентрация (Cmax), время достижения максимальной концентрации (Tmax))
- распределение: процессы, связанные с распространением препарата в организме (связывание с белками, объем распределения, проникновение в ткани)
- метаболизм: процессы биотрансформации препарата (образование метаболитов, печеночный метаболизм, изоферменты CYP, эффект первого прохождения)
- выведение: процессы удаления препарата из организма (элиминация, период полувыведения, почечный клиренс, выведение через кишечник)
- другое: если предложение не относится к фармакокинетике или содержит общую информацию

Ответ должен быть строго в формате JSON:
{{"category": "название категории", "confidence": 0.0-1.0, "reasoning": "объяснение"}}"""),
            ("human", "Предложение: {text}")
        ])
        
        self.chain = self.prompt | self.llm
    
    def classify_sentence(self, text: str) -> str:
        """Классифицирует предложение и возвращает категорию"""
        for attempt in range(3):
            try:
                response = self.chain.invoke({"text": text})
                
                try:
                    json_start = response.find('{')
                    json_end = response.rfind('}') + 1
                    
                    if json_start != -1 and json_end > json_start:
                        json_str = response[json_start:json_end]
                        result_dict = json.loads(json_str)
                    else:
                        result_dict = json.loads(response)
                    
                    category = result_dict.get("category", "").lower()
                    
                    if category in self.CATEGORIES:
                        return category
                    else:
                        if attempt == 2:
                            return "другое"
                        
                except json.JSONDecodeError:
                    if attempt == 2:
                        category = self._extract_category_from_text(response)
                        return category
                        
            except Exception as e:
                if attempt == 2:
                    return "другое"
        
        return "другое"
    
    def _extract_category_from_text(self, text: str) -> str:
        """Извлекает категорию из текстового ответа"""
        text_lower = text.lower()
        for category in self.CATEGORIES:
            if category in text_lower:
                return category
        return "другое"

def load_drug_data(filename: str) -> List[Dict]:
    """Загружает данные из JSON файла"""
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)

def main():
    # Загружаем данные
    input_file = "data\\pd_pk_split_text.json"
    output_file = "data\\classified_drugs.json"
    
    print(f"Загрузка данных из файла: {input_file}")
    drug_data = load_drug_data(input_file)
    
    # Инициализация классификатора
    classifier = SentenceClassifier(
        model_name="ilyagusev/saiga_llama3",
        base_url="http://localhost:11434"
    )
    
    # Обрабатываем каждый препарат
    classified_drugs = []
    
    for drug_item in drug_data:
        drug_name = drug_item.get('drug', '')
        pd_sentences = drug_item.get('pd', [])
        pk_sentences = drug_item.get('pk', [])
        
        print(f"\nОбработка препарата: {drug_name}")
        print(f"PK предложений: {len(pk_sentences)}")
        
        # Классифицируем PK предложения
        classified_pk = []
        for sentence_text in pk_sentences:
            category = classifier.classify_sentence(sentence_text)
            classified_pk.append({
                "sentence": sentence_text,
                "category": category
            })
            print(f"  {category}: {sentence_text[:50]}...")
        
        # Сохраняем в исходной структуре
        classified_drug = {
            "drug": drug_name,
            "pd": pd_sentences,
            "pk": classified_pk
        }
        classified_drugs.append(classified_drug)
    
    # Сохраняем результаты
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(classified_drugs, f, ensure_ascii=False, indent=2)
    
    print(f"\nРезультаты сохранены в файл: {output_file}")
    print(f"Всего обработано препаратов: {len(classified_drugs)}")

if __name__ == "__main__":
    main()