import json
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from razdel import sentenize
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
import time
from collections import defaultdict

# ==================== МОДЕЛИ ДАННЫХ ====================

class Sentence(BaseModel):
    """Модель предложения с метаданными"""
    text: str
    index: int
    start_char: int
    end_char: int
    length: int
    category: Optional[str] = None
    confidence: Optional[float] = None
    used_for_extraction: bool = False

class Pharmacodynamics(BaseModel):
    """Фармакодинамика"""
    mechanism_of_action: str = Field(description="Механизм действия")
    main_effects: List[str] = Field(description="Основные эффекты")
    target_enzymes: List[str] = Field(description="Целевые ферменты")
    metabolites_activity: Optional[str] = Field(description="Активность метаболитов")
    source_sentences: List[int] = Field(description="Индексы предложений-источников")

class Pharmacokinetics(BaseModel):
    """Фармакокинетика"""
    absorption: str = Field(description="Всасывание: время, биодоступность")
    distribution: str = Field(description="Распределение: связь с белками, объем распределения")
    metabolism: str = Field(description="Метаболизм: путь метаболизма, активные метаболиты")
    excretion: str = Field(description="Выведение: путь выведения, период полувыведения")
    special_populations: Optional[str] = Field(description="Особые группы пациентов")
    source_sentences: List[int] = Field(description="Индексы предложений-источников")

class DrugInfo(BaseModel):
    """Полная информация о препарате"""
    drug_name: str = Field(description="Название препарата")
    pharmacodynamics: Pharmacodynamics = Field(description="Фармакодинамика")
    pharmacokinetics: Pharmacokinetics = Field(description="Фармакокинетика")
    key_findings: List[str] = Field(description="Ключевые выводы")
    clinical_significance: Optional[str] = Field(description="Клиническая значимость")
    all_sentences: List[Sentence] = Field(description="Все предложения с классификацией")

# ==================== КЛАССИФИКАТОР ПРЕДЛОЖЕНИЙ ====================

class SentenceClassifier:
    """Классификатор предложений для медицинских инструкций"""
    
    # Категории предложений
    CATEGORIES = {
        "pd_mechanism": "Фармакодинамика - механизм действия",
        "pd_effects": "Фармакодинамика - эффекты",
        "pd_enzymes": "Фармакодинамика - ферменты",
        "pd_metabolites": "Фармакодинамика - метаболиты",
        
        "pk_absorption": "Фармакокинетика - всасывание",
        "pk_distribution": "Фармакокинетика - распределение",
        "pk_metabolism": "Фармакокинетика - метаболизм",
        "pk_excretion": "Фармакокинетика - выведение",
        "pk_special": "Фармакокинетика - особые группы",
        
        "general": "Общая информация",
        "irrelevant": "Не относится к фармакологии"
    }
    
    def __init__(self, model_name: str = "ilyagusev/saiga_llama3:latest"):
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.1,  # Низкая температура для стабильности
            num_ctx=2048
        )
        
        # Промпт для классификации
        self.classification_prompt = """Ты - медицинский эксперт. Классифицируй предложение из инструкции лекарственного средства.

Категории:
- pd_mechanism: механизм действия препарата (как работает)
- pd_effects: фармакологические эффекты (что делает)
- pd_enzymes: ферменты, на которые влияет препарат
- pd_metabolites: информация о метаболитах

- pk_absorption: всасывание (биодоступность, Cmax, Tmax)
- pk_distribution: распределение (Vd, связь с белками)
- pk_metabolism: метаболизм (пути, ферменты метаболизма)
- pk_excretion: выведение (пути, T1/2)
- pk_special: особенности у разных групп пациентов

- general: общая информация о препарате
- irrelevant: информация, не относящаяся к фармакологии

Предложение: "{sentence}"

Ответь только названием категории (одним словом из списка выше)."""

    def classify_sentence(self, sentence: str) -> Tuple[str, float]:
        """Классифицировать одно предложение"""
        
        prompt = self.classification_prompt.format(sentence=sentence)
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            category = response.content.strip().lower()
            
            # Валидация категории
            if category not in self.CATEGORIES:
                # Пробуем найти частичное совпадение
                for cat in self.CATEGORIES:
                    if cat in category or category in cat:
                        return cat, 0.8
                return "irrelevant", 0.5
            
            return category, 0.9
            
        except Exception as e:
            print(f"Ошибка классификации: {e}")
            return "irrelevant", 0.0
    
    def batch_classify(self, sentences: List[Sentence], batch_size: int = 5) -> List[Sentence]:
        """Классификация батча предложений"""
        
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            
            for sentence in batch:
                if not sentence.category:
                    category, confidence = self.classify_sentence(sentence.text)
                    sentence.category = category
                    sentence.confidence = confidence
                    
                    # Небольшая задержка между запросами
                    time.sleep(0.2)
            
            print(f"   Классифицировано {min(i+batch_size, len(sentences))}/{len(sentences)} предложений")
        
        return sentences

# ==================== ЭКСТРАКТОРЫ ====================

class PDExtractor:
    """Экстрактор данных фармакодинамики из классифицированных предложений"""
    
    def __init__(self, model_name: str = "ilyagusev/saiga_llama3:latest"):
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.1,
            num_ctx=2048
        )
    
    def extract_from_sentences(self, sentences: List[Sentence]) -> Pharmacodynamics:
        """Извлечение фармакодинамики из предложений"""
        
        # Фильтруем релевантные предложения
        relevant = [s for s in sentences if s.category and s.category.startswith("pd_")]
        
        if not relevant:
            print("⚠️ Нет предложений по фармакодинамике")
            return self._create_empty()
        
        # Группируем по подкатегориям
        grouped = defaultdict(list)
        for s in relevant:
            grouped[s.category].append(s.text)
            s.used_for_extraction = True
        
        # Формируем промпт
        prompt = self._create_extraction_prompt(grouped)
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return self._parse_response(response.content, [s.index for s in relevant])
        except Exception as e:
            print(f"Ошибка извлечения: {e}")
            return self._create_empty([s.index for s in relevant])
    
    def _create_extraction_prompt(self, grouped: Dict[str, List[str]]) -> str:
        """Создание промпта для извлечения"""
        
        prompt = "На основе следующих предложений из инструкции препарата, извлеки информацию о фармакодинамике:\n\n"
        
        for category, sentences in grouped.items():
            prompt += f"\n{self._get_category_description(category)}:\n"
            for i, s in enumerate(sentences, 1):
                prompt += f"{i}. {s}\n"
        
        prompt += """
        
        Заполни следующие поля (используй только информацию из предложений):
        1. Механизм действия (кратко)
        2. Основные эффекты (список)
        3. Целевые ферменты (список)
        4. Активность метаболитов (если есть)
        
        Формат ответа:
        МЕХАНИЗМ: ...
        ЭФФЕКТЫ: эффект1; эффект2; эффект3
        ФЕРМЕНТЫ: фермент1; фермент2
        МЕТАБОЛИТЫ: ...
        """
        
        return prompt
    
    def _get_category_description(self, category: str) -> str:
        """Описание категории для промпта"""
        descriptions = {
            "pd_mechanism": "Предложения о механизме действия",
            "pd_effects": "Предложения об эффектах",
            "pd_enzymes": "Предложения о ферментах",
            "pd_metabolites": "Предложения о метаболитах"
        }
        return descriptions.get(category, category)
    
    def _parse_response(self, response: str, source_indices: List[int]) -> Pharmacodynamics:
        """Парсинг ответа модели"""
        
        lines = response.split('\n')
        
        mechanism = ""
        effects = []
        enzymes = []
        metabolites = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith("МЕХАНИЗМ:"):
                mechanism = line.replace("МЕХАНИЗМ:", "").strip()
            elif line.startswith("ЭФФЕКТЫ:"):
                effects_text = line.replace("ЭФФЕКТЫ:", "").strip()
                effects = [e.strip() for e in effects_text.split(';') if e.strip()]
            elif line.startswith("ФЕРМЕНТЫ:"):
                enzymes_text = line.replace("ФЕРМЕНТЫ:", "").strip()
                enzymes = [e.strip() for e in enzymes_text.split(';') if e.strip()]
            elif line.startswith("МЕТАБОЛИТЫ:"):
                metabolites = line.replace("МЕТАБОЛИТЫ:", "").strip()
        
        return Pharmacodynamics(
            mechanism_of_action=mechanism or "Не указано",
            main_effects=effects or ["Не указано"],
            target_enzymes=enzymes or ["Не указано"],
            metabolites_activity=metabolites or None,
            source_sentences=source_indices
        )
    
    def _create_empty(self, source_indices: List[int] = None) -> Pharmacodynamics:
        """Создание пустой структуры"""
        return Pharmacodynamics(
            mechanism_of_action="Не удалось извлечь",
            main_effects=["Не удалось извлечь"],
            target_enzymes=["Не удалось извлечь"],
            metabolites_activity=None,
            source_sentences=source_indices or []
        )

class PKExtractor:
    """Экстрактор данных фармакокинетики из классифицированных предложений"""
    
    def __init__(self, model_name: str = "ilyagusev/saiga_llama3:latest"):
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.1,
            num_ctx=2048
        )
    
    def extract_from_sentences(self, sentences: List[Sentence]) -> Pharmacokinetics:
        """Извлечение фармакокинетики из предложений"""
        
        # Фильтруем релевантные предложения
        relevant = [s for s in sentences if s.category and s.category.startswith("pk_")]
        
        if not relevant:
            print("⚠️ Нет предложений по фармакокинетике")
            return self._create_empty()
        
        # Группируем по подкатегориям
        grouped = defaultdict(list)
        for s in relevant:
            grouped[s.category].append(s.text)
            s.used_for_extraction = True
        
        # Формируем промпт
        prompt = self._create_extraction_prompt(grouped)
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return self._parse_response(response.content, [s.index for s in relevant])
        except Exception as e:
            print(f"Ошибка извлечения: {e}")
            return self._create_empty([s.index for s in relevant])
    
    def _create_extraction_prompt(self, grouped: Dict[str, List[str]]) -> str:
        """Создание промпта для извлечения"""
        
        prompt = "На основе следующих предложений из инструкции препарата, извлеки информацию о фармакокинетике:\n\n"
        
        for category, sentences in grouped.items():
            prompt += f"\n{self._get_category_description(category)}:\n"
            for i, s in enumerate(sentences, 1):
                prompt += f"{i}. {s}\n"
        
        prompt += """
        
        Заполни следующие поля (используй только информацию из предложений, указывай числовые значения):
        1. Всасывание (биодоступность, Cmax, Tmax)
        2. Распределение (Vd, связь с белками)
        3. Метаболизм (пути, ферменты, метаболиты)
        4. Выведение (пути, T1/2)
        5. Особые группы пациентов (если есть)
        
        Формат ответа:
        ВСАСЫВАНИЕ: ...
        РАСПРЕДЕЛЕНИЕ: ...
        МЕТАБОЛИЗМ: ...
        ВЫВЕДЕНИЕ: ...
        ОСОБЫЕ ГРУППЫ: ...
        """
        
        return prompt
    
    def _get_category_description(self, category: str) -> str:
        """Описание категории для промпта"""
        descriptions = {
            "pk_absorption": "Предложения о всасывании",
            "pk_distribution": "Предложения о распределении",
            "pk_metabolism": "Предложения о метаболизме",
            "pk_excretion": "Предложения о выведении",
            "pk_special": "Предложения об особых группах"
        }
        return descriptions.get(category, category)
    
    def _parse_response(self, response: str, source_indices: List[int]) -> Pharmacokinetics:
        """Парсинг ответа модели"""
        
        lines = response.split('\n')
        
        absorption = ""
        distribution = ""
        metabolism = ""
        excretion = ""
        special = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith("ВСАСЫВАНИЕ:"):
                absorption = line.replace("ВСАСЫВАНИЕ:", "").strip()
            elif line.startswith("РАСПРЕДЕЛЕНИЕ:"):
                distribution = line.replace("РАСПРЕДЕЛЕНИЕ:", "").strip()
            elif line.startswith("МЕТАБОЛИЗМ:"):
                metabolism = line.replace("МЕТАБОЛИЗМ:", "").strip()
            elif line.startswith("ВЫВЕДЕНИЕ:"):
                excretion = line.replace("ВЫВЕДЕНИЕ:", "").strip()
            elif line.startswith("ОСОБЫЕ ГРУППЫ:"):
                special = line.replace("ОСОБЫЕ ГРУППЫ:", "").strip()
        
        return Pharmacokinetics(
            absorption=absorption or "Не указано",
            distribution=distribution or "Не указано",
            metabolism=metabolism or "Не указано",
            excretion=excretion or "Не указано",
            special_populations=special if special else None,
            source_sentences=source_indices
        )
    
    def _create_empty(self, source_indices: List[int] = None) -> Pharmacokinetics:
        """Создание пустой структуры"""
        return Pharmacokinetics(
            absorption="Не удалось извлечь",
            distribution="Не удалось извлечь",
            metabolism="Не удалось извлечь",
            excretion="Не удалось извлечь",
            special_populations=None,
            source_sentences=source_indices or []
        )

# ==================== ИНТЕГРАТОР ====================

class DrugInfoIntegrator:
    """Интегратор полной информации о препарате"""
    
    def __init__(self, model_name: str = "ilyagusev/saiga_llama3:latest"):
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.2,
            num_ctx=2048
        )
    
    def integrate(self, 
                  drug_name: str,
                  pd: Pharmacodynamics,
                  pk: Pharmacokinetics,
                  all_sentences: List[Sentence]) -> DrugInfo:
        """Интеграция всей информации"""
        
        # Формируем промпт для ключевых выводов
        prompt = self._create_integration_prompt(drug_name, pd, pk)
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            key_findings, clinical = self._parse_integration(response.content)
        except Exception as e:
            print(f"Ошибка интеграции: {e}")
            key_findings = ["Ошибка при формировании выводов"]
            clinical = None
        
        return DrugInfo(
            drug_name=drug_name,
            pharmacodynamics=pd,
            pharmacokinetics=pk,
            key_findings=key_findings,
            clinical_significance=clinical,
            all_sentences=all_sentences
        )
    
    def _create_integration_prompt(self, drug_name: str, pd: Pharmacodynamics, pk: Pharmacokinetics) -> str:
        """Создание промпта для интеграции"""
        
        return f"""
        На основе данных о препарате {drug_name}, сформулируй ключевые выводы и клиническую значимость.
        
        ФАРМАКОДИНАМИКА:
        - Механизм действия: {pd.mechanism_of_action}
        - Основные эффекты: {', '.join(pd.main_effects)}
        - Целевые ферменты: {', '.join(pd.target_enzymes)}
        - Метаболиты: {pd.metabolites_activity or 'Не указано'}
        
        ФАРМАКОКИНЕТИКА:
        - Всасывание: {pk.absorption}
        - Распределение: {pk.distribution}
        - Метаболизм: {pk.metabolism}
        - Выведение: {pk.excretion}
        - Особые группы: {pk.special_populations or 'Не указано'}
        
        Сформулируй:
        1. 3-5 ключевых выводов о препарате (каждый одним предложением)
        2. Клиническую значимость (2-3 предложения о применении в практике)
        
        Формат ответа:
        ВЫВОДЫ:
        - вывод 1
        - вывод 2
        - вывод 3
        
        КЛИНИЧЕСКАЯ ЗНАЧИМОСТЬ:
        текст о клинической значимости
        """
    
    def _parse_integration(self, response: str) -> Tuple[List[str], Optional[str]]:
        """Парсинг ответа интеграции"""
        
        lines = response.split('\n')
        findings = []
        clinical = None
        current_section = None
        
        for line in lines:
            line = line.strip()
            if line.startswith("ВЫВОДЫ:"):
                current_section = "findings"
            elif line.startswith("КЛИНИЧЕСКАЯ ЗНАЧИМОСТЬ:"):
                current_section = "clinical"
            elif line.startswith("-") and current_section == "findings":
                findings.append(line.lstrip('- ').strip())
            elif current_section == "clinical" and line:
                if clinical is None:
                    clinical = line
                else:
                    clinical += " " + line
        
        return findings or ["Не удалось сформулировать выводы"], clinical

# ==================== ОСНОВНОЙ ПАЙПЛАЙН ====================

class MedicalPipelineRazdel:
    """Пайплайн обработки медицинских инструкций с razdel и классификацией"""
    
    def __init__(self, model_name: str = "ilyagusev/saiga_llama3:latest"):
        self.model_name = model_name
        self.classifier = SentenceClassifier(model_name)
        self.pd_extractor = PDExtractor(model_name)
        self.pk_extractor = PKExtractor(model_name)
        self.integrator = DrugInfoIntegrator(model_name)
        
        self.stats = {
            'processed': 0,
            'total_sentences': 0,
            'sentences_by_category': defaultdict(int),
            'avg_confidence': 0.0,
            'processing_time': 0.0
        }
    
    def split_into_sentences(self, text: str) -> List[Sentence]:
        """Разделение текста на предложения с помощью razdel"""
        
        sentences = []
        for i, sent in enumerate(sentenize(text)):
            sentences.append(Sentence(
                text=sent.text,
                index=i,
                start_char=sent.start,
                end_char=sent.stop,
                length=len(sent.text)
            ))
        
        print(f"📝 Разделено на {len(sentences)} предложений")
        return sentences
    
    def find_split_marker(self, sentences: List[Sentence]) -> int:
        """Поиск маркера 'Фармакокинетика' среди предложений"""
        
        for i, sent in enumerate(sentences):
            if "Фармакокинетика" in sent.text:
                print(f"🔍 Маркер 'Фармакокинетика' найден в предложении {i}")
                return i
        
        print("⚠️ Маркер 'Фармакокинетика' не найден")
        return -1
    
    def process_instruction(self, text: str, instruction_id: str = None) -> DrugInfo:
        """Полная обработка инструкции"""
        
        start_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"ОБРАБОТКА ИНСТРУКЦИИ {instruction_id or ''}")
        print(f"{'='*70}")
        
        # Шаг 1: Разделение на предложения
        print("\n🔹 ШАГ 1: Разделение текста на предложения")
        sentences = self.split_into_sentences(text)
        
        # Шаг 2: Поиск маркера (опционально)
        marker_idx = self.find_split_marker(sentences)
        
        # Шаг 3: Классификация предложений
        print("\n🔹 ШАГ 2: Классификация предложений")
        sentences = self.classifier.batch_classify(sentences)
        
        # Статистика классификации
        for s in sentences:
            if s.category:
                self.stats['sentences_by_category'][s.category] += 1
        
        # Шаг 4: Извлечение фармакодинамики
        print("\n🔹 ШАГ 3: Извлечение фармакодинамики")
        pd_result = self.pd_extractor.extract_from_sentences(sentences)
        
        # Шаг 5: Извлечение фармакокинетики
        print("\n🔹 ШАГ 4: Извлечение фармакокинетики")
        pk_result = self.pk_extractor.extract_from_sentences(sentences)
        
        # Шаг 6: Извлечение названия препарата
        # drug_name = self._extract_drug_name(sentences)
        
        # Шаг 7: Интеграция
        print("\n🔹 ШАГ 5: Интеграция результатов")
        result = self.integrator.integrate("", pd_result, pk_result, sentences)
        
        # Обновляем статистику
        self.stats['processed'] += 1
        self.stats['total_sentences'] += len(sentences)
        self.stats['processing_time'] += time.time() - start_time
        
        # Средняя уверенность
        confidences = [s.confidence for s in sentences if s.confidence]
        if confidences:
            self.stats['avg_confidence'] = (self.stats['avg_confidence'] * (self.stats['processed'] - 1) + 
                                           sum(confidences)/len(confidences)) / self.stats['processed']
        
        return result
    
    def _extract_drug_name(self, sentences: List[Sentence]) -> str:
        """Извлечение названия препарата из первых предложений"""
        
        # Ищем в первых 5 предложениях
        for sent in sentences[:5]:
            if "Аллопуринол" in sent.text:
                return "Аллопуринол"
        
        return "Неизвестный препарат"
    
    def print_detailed_report(self, result: DrugInfo):
        """Детальный отчет с указанием источников"""
        
        print(f"\n{'='*70}")
        print(f"📋 ДЕТАЛЬНЫЙ ОТЧЕТ: {result.drug_name}")
        print(f"{'='*70}")
        
        # Статистика по предложениям
        total = len(result.all_sentences)
        classified = len([s for s in result.all_sentences if s.category])
        used = len([s for s in result.all_sentences if s.used_for_extraction])
        
        print(f"\n📊 СТАТИСТИКА ПРЕДЛОЖЕНИЙ:")
        print(f"   Всего: {total}")
        print(f"   Классифицировано: {classified} ({classified/total*100:.1f}%)")
        print(f"   Использовано для извлечения: {used} ({used/total*100:.1f}%)")
        
        # Распределение по категориям
        categories = defaultdict(int)
        for s in result.all_sentences:
            if s.category:
                categories[s.category] += 1
        
        if categories:
            print(f"\n🏷️ РАСПРЕДЕЛЕНИЕ ПО КАТЕГОРИЯМ:")
            for cat, count in sorted(categories.items()):
                print(f"   {cat}: {count}")
        
        # Фармакодинамика с источниками
        print(f"\n🔬 ФАРМАКОДИНАМИКА:")
        print(f"   Механизм: {result.pharmacodynamics.mechanism_of_action}")
        print(f"   Эффекты: {', '.join(result.pharmacodynamics.main_effects)}")
        print(f"   Ферменты: {', '.join(result.pharmacodynamics.target_enzymes)}")
        if result.pharmacodynamics.metabolites_activity:
            print(f"   Метаболиты: {result.pharmacodynamics.metabolites_activity}")
        
        if result.pharmacodynamics.source_sentences:
            print(f"\n   📍 Предложения-источники (фармакодинамика):")
            for idx in result.pharmacodynamics.source_sentences[:3]:  # Первые 3
                if idx < len(result.all_sentences):
                    print(f"      [{idx}] {result.all_sentences[idx].text[:100]}...")
        
        # Фармакокинетика с источниками
        print(f"\n💊 ФАРМАКОКИНЕТИКА:")
        print(f"   Всасывание: {result.pharmacokinetics.absorption}")
        print(f"   Распределение: {result.pharmacokinetics.distribution}")
        print(f"   Метаболизм: {result.pharmacokinetics.metabolism}")
        print(f"   Выведение: {result.pharmacokinetics.excretion}")
        if result.pharmacokinetics.special_populations:
            print(f"   Особые группы: {result.pharmacokinetics.special_populations}")
        
        if result.pharmacokinetics.source_sentences:
            print(f"\n   📍 Предложения-источники (фармакокинетика):")
            for idx in result.pharmacokinetics.source_sentences[:3]:  # Первые 3
                if idx < len(result.all_sentences):
                    print(f"      [{idx}] {result.all_sentences[idx].text[:100]}...")
        
        # Ключевые выводы
        print(f"\n🎯 КЛЮЧЕВЫЕ ВЫВОДЫ:")
        for i, finding in enumerate(result.key_findings, 1):
            print(f"   {i}. {finding}")
        
        if result.clinical_significance:
            print(f"\n🏥 КЛИНИЧЕСКАЯ ЗНАЧИМОСТЬ:")
            print(f"   {result.clinical_significance}")
    
    def print_pipeline_stats(self):
        """Статистика работы пайплайна"""
        
        print(f"\n{'='*70}")
        print("📊 СТАТИСТИКА ПАЙПЛАЙНА")
        print(f"{'='*70}")
        
        print(f"\n✅ Обработано инструкций: {self.stats['processed']}")
        print(f"📝 Всего предложений: {self.stats['total_sentences']}")
        print(f"⏱️  Общее время: {self.stats['processing_time']:.2f}с")
        if self.stats['processed'] > 0:
            print(f"⚡ Среднее время на инструкцию: {self.stats['processing_time']/self.stats['processed']:.2f}с")
            print(f"🎯 Средняя уверенность классификации: {self.stats['avg_confidence']:.2f}")
        
        print(f"\n🏷️ Распределение по категориям:")
        for cat, count in sorted(self.stats['sentences_by_category'].items()):
            print(f"   {cat}: {count}")
        
        print(f"\n💾 Модель: {self.model_name}")
        print(f"   VRAM: ~7.93 GB (постоянно)")

# ==================== ТЕСТИРОВАНИЕ ====================

def main():
    """Основная функция тестирования"""
    
    instruction_text = """Фармакодинамика Аллопуринол является структурным аналогом гипоксантина. Аллопуринол, а также его основной активный метаболит - оксипуринол, ингибируют ксантиноксидазу - фермент, обеспечивающий преобразование гипоксантина в ксантин, и ксантина в мочевую кислоту. Аллопуринол уменьшает концентрацию мочевой кислоты как в сыворотке крови, так и в моче. Тем самым он предотвращает отложение кристаллов мочевой кислоты в тканях и (или) способствует их растворению. Помимо подавления катаболизма пуринов у некоторых (но не у всех) пациентов с гиперурикемией, большое количество ксантина и гипоксантина становится доступно для повторного образования пуриновых оснований, что приводит к угнетению биосинтеза пуринов de novo по механизму обратной связи, что опосредовано угнетением фермента гипоксантин-гуанин фосфорибозил-трансферазы. Другие метаболиты аллопуринола - аллопуринол-рибозид и оксипуринол-7 рибозид. Фармакокинетика Всасывание Аллопуринол активен при пероральном применении. Он быстро всасывается из верхних отделов желудочно-кишечного тракта (ЖКТ). По данным фармакокинетических исследований аллопуринол определяется в крови уже через 30 - 60 минут после приема. Биодоступность аллопуринола варьирует от 67 % до 90 %. Максимальная концентрация препарата в плазме крови (TCmax) как правило регистрируется приблизительно через 1,5 часа после перорального приема. Затем концентрация аллопуринола быстро снижается. Спустя 6 часов приема, в плазме крови определяется лишь следовая концентрация препарата. Максимальная концентрация (Cmax) активного метаболита – оксипуринола обыкновенно регистрируется через 3 - 5 часов после перорального приема аллопуринола. Концентрация оксипуринола в плазме крови снижается значительно медленнее. Распределение Аллопуринол почти не связывается с белками плазмы крови, поэтому изменения степени связывания с белками не должны оказывать значительного влияния на клиренс препарата. Кажущийся объем распределения (Vd) аллопуринола составляет приблизительно 1,6 литра/кг, что говорит о достаточно выраженном поглощении препарата тканями. Концентрация аллопуринола в различных тканях человека не изучена, однако весьма вероятно, что аллопуринол и оксипуринол в максимальной концентрации накапливаются в печени и слизистой оболочке кишечника, где регистрируется высокая активность ксантиноксидазы. Биотрансформация Под действием ксантиноксидазы и альдегидоксидазы аллопуринол метаболизируется с образованием оксипуринола. Оксипуринол подавляет активность ксантиноксидазы. Тем не менее, оксипуринол – не столь мощный ингибитор ксантиноксидазы, по сравнению с аллопуринолом, однако его период полувыведения (T1/2) значительно больше. Благодаря этим свойствам после приема разовой суточной дозы аллопуринола эффективное подавление активности ксантиноксидазы поддерживается в течение 24 часов. У пациентов с нормальной функцией почек концентрация оксипуринола в плазме крови медленно увеличивается вплоть до достижения равновесной концентрации. После приема аллопуринола в дозе 300 мг в сутки концентрация аллопуринола в плазме крови, как правило, составляет 5 - 10 мг/л. К другим метаболитам аллопуринола относятся аллопуринол-рибозид и оксипуринол-7-рибозид. Выведение Приблизительно 20 % принятого per os аллопуринола выводится через кишечник в неизмененном виде. Около 10 % суточной дозы экскретируются клубочковым аппаратом почки в виде неизмененного аллопуринола. Еще 70 % суточной дозы аллопуринола выводится почками в форме оксипуринола. Оксипуринол выводится почками в неизмененном виде, однако, в связи с канальцевой реабсорбцией, он обладает длительным T1/2. T1/2 аллопуринола составляет 1 - 2 часа, тогда как T1/2 оксипуринола варьирует от 13 до 30 часов. Такие значительные различия вероятно связаны с различиями в структуре исследований и/или клиренсе креатинина (КК) у пациентов. Пациенты с нарушенной функцией почек У пациентов с нарушенной функцией почек выведение аллопуринола и оксипуринола может значительно замедляться, что при длительной терапии приводит к росту концентрации этих соединений в плазме крови. У пациентов с нарушенной функцией почек и КК 10 - 20 мл/мин после долговременной терапии аллопуринолом в дозе 300 мг в сутки концентрация оксипуринола в плазме крови достигала ориентировочно 30 мг/мл. Такая концентрация окиспуринола может определяться у пациентов с нормальной функцией почек на фоне терапии аллопуринолом в дозе 600 мг/сутки. Следовательно, при лечении пациентов с нарушением функции почек дозу аллопуринола необходимо снижать. Пациенты пожилого возраста У пациентов пожилого возраста значительные изменения фармакокинетических свойств аллопуринола маловероятны. Исключение составляют пациенты с сопутствующей патологией почек ."""

    # Создаем пайплайн
    pipeline = MedicalPipelineRazdel(model_name="ilyagusev/saiga_llama3:latest")
    
    # Обрабатываем инструкцию
    result = pipeline.process_instruction(instruction_text, "ALLOPURINOL_001")
    
    # Детальный отчет
    pipeline.print_detailed_report(result)
    
    # Статистика пайплайна
    pipeline.print_pipeline_stats()
    
    return pipeline

if __name__ == "__main__":
    pipeline = main()