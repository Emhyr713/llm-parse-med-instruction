import ollama
import time
import subprocess
import json

class MemoryTracker:
    """Отслеживание использования памяти GPU"""
    
    @staticmethod
    def get_gpu_memory():
        """Получить использование памяти GPU через nvidia-smi"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                check=True
            )
            if result.stdout:
                used, total = map(int, result.stdout.strip().split(','))
                return {
                    'used_gb': used / 1024,
                    'total_gb': total / 1024,
                    'used_mb': used,
                    'used_percent': (used / total) * 100
                }
        except:
            return None
    
    @staticmethod
    def get_ollama_stats():
        """Получить статистику от Ollama API"""
        try:
            import requests
            response = requests.get('http://localhost:11434/api/ps')
            if response.status_code == 200:
                return response.json()
        except:
            return None

class MonitoredAgent:
    def __init__(self, name, system_prompt, model='llama3'):
        self.name = name
        self.model = model
        self.messages = [{'role': 'system', 'content': system_prompt}]
        self.created_at = time.time()
        self.total_tokens = 0
        
    def chat(self, user_input):
        self.messages.append({'role': 'user', 'content': user_input})
        
        response = ollama.chat(
            model=self.model,
            messages=self.messages,
            options={'num_ctx': 2048}  # Оптимизация памяти
        )
        
        assistant_response = response['message']['content']
        self.messages.append({'role': 'assistant', 'content': assistant_response})
        
        # Примерный подсчет токенов
        self.total_tokens += len(user_input) // 4 + len(assistant_response) // 4
        
        return assistant_response
    
    def get_memory_estimate(self):
        """Оценить использование памяти этим агентом"""
        # Приблизительный расчет: 1 токен ~ 0.5 MB в памяти
        tokens = sum(len(m.get('content', '')) // 4 for m in self.messages)
        memory_mb = tokens * 0.5  # Примерно 0.5 MB на токен в контексте
        return {
            'tokens': tokens,
            'memory_mb': memory_mb,
            'memory_gb': memory_mb / 1024
        }

def test_with_memory_tracking():
    """Тестирование с отслеживанием памяти"""
    
    print("=" * 70)
    print("ТЕСТИРОВАНИЕ ПАМЯТИ GPU С НЕСКОЛЬКИМИ АГЕНТАМИ")
    print("=" * 70)
    
    # Начальное состояние памяти
    tracker = MemoryTracker()
    initial_memory = tracker.get_gpu_memory()
    
    if initial_memory:
        print(f"\n📊 Начальное состояние GPU:")
        print(f"   Всего: {initial_memory['total_gb']:.2f} GB")
        print(f"   Использовано: {initial_memory['used_gb']:.2f} GB")
        print(f"   Свободно: {initial_memory['total_gb'] - initial_memory['used_gb']:.2f} GB")
        print(f"   Загрузка: {initial_memory['used_percent']:.1f}%")
    
    # Проверяем, какие модели уже загружены
    try:
        import requests
        ps_response = requests.get('http://localhost:11434/api/ps')
        if ps_response.status_code == 200:
            models_data = ps_response.json()
            if models_data.get('models'):
                print(f"\n🔄 Уже загруженные модели:")
                for m in models_data['models']:
                    print(f"   - {m['name']}: {m.get('size', 0) / 1024**3:.2f} GB")
    except:
        pass
    
    # Создаем агентов
    prompts = [
        "Ты - эксперт по Python, помогаешь с кодом",
        "Ты - поэт, пишешь стихи на заказ",
        "Ты - историк, рассказываешь о событиях прошлого",
        "Ты - математик, решаешь сложные задачи",
        "Ты - психолог, даешь советы"
    ]
    
    agents = []
    model_name = 'ilyagusev/saiga_llama3'  # Ваша модель
    
    print(f"\n{'='*70}")
    print(f"СОЗДАНИЕ {len(prompts)} АГЕНТОВ")
    print(f"{'='*70}")
    
    memory_readings = []
    
    for i, prompt in enumerate(prompts):
        print(f"\n{'-'*50}")
        print(f"🔄 Агент {i+1}: {prompt[:30]}...")
        
        # Создаем агента
        agent = MonitoredAgent(f"Агент_{i+1}", prompt, model_name)
        agents.append(agent)
        
        # Отправляем запрос
        response = agent.chat("Представься кратко в одном предложении")
        print(f"   📝 Ответ: {response[:80]}...")
        
        # Оценка памяти агента
        mem_est = agent.get_memory_estimate()
        print(f"   📊 Контекст: {mem_est['tokens']} токенов, ~{mem_est['memory_mb']:.2f} MB")
        
        # Замер памяти
        time.sleep(1)  # Даем время на обновление
        current_memory = tracker.get_gpu_memory()
        
        if current_memory:
            mem_diff = current_memory['used_gb'] - (memory_readings[-1]['used_gb'] if memory_readings else initial_memory['used_gb'])
            print(f"   🎮 GPU память: {current_memory['used_gb']:.2f} GB (+{mem_diff:.3f} GB)")
            
            memory_readings.append({
                'agent': i+1,
                'used_gb': current_memory['used_gb'],
                'diff_gb': mem_diff
            })
    
    # Итоговый анализ
    print(f"\n{'='*70}")
    print("ИТОГОВЫЙ АНАЛИЗ ПАМЯТИ")
    print(f"{'='*70}")
    
    if initial_memory and memory_readings:
        final_memory = memory_readings[-1]
        
        print(f"\n📊 Динамика памяти:")
        print(f"   Начало:  {initial_memory['used_gb']:.3f} GB")
        print(f"   Агент 1: {memory_readings[0]['used_gb']:.3f} GB (+{memory_readings[0]['diff_gb']:.3f})")
        
        for i in range(1, len(memory_readings)):
            print(f"   Агент {i+1}: {memory_readings[i]['used_gb']:.3f} GB (+{memory_readings[i]['diff_gb']:.3f})")
        
        total_increase = final_memory['used_gb'] - initial_memory['used_gb']
        
        print(f"\n💾 ОБЩИЙ РЕЗУЛЬТАТ:")
        print(f"   Начальное использование: {initial_memory['used_gb']:.2f} GB")
        print(f"   Конечное использование:   {final_memory['used_gb']:.2f} GB")
        print(f"   💿 Модель + все агенты:   {total_increase:.3f} GB")
        
        # Расчет на одного агента
        if total_increase > 0:
            model_base = total_increase - sum(r['diff_gb'] for r in memory_readings[1:])
            print(f"\n   📈 Примерное распределение:")
            print(f"      - Базовая модель: ~{model_base:.3f} GB")
            for i, r in enumerate(memory_readings):
                if i == 0:
                    print(f"      - Первый агент:   ~{r['diff_gb']:.3f} GB (инициализация)")
                else:
                    print(f"      - Агент {i+1}:       ~{r['diff_gb']:.3f} GB")
    
    # Проверка через Ollama API
    try:
        ps_response = requests.get('http://localhost:11434/api/ps')
        if ps_response.status_code == 200:
            models_data = ps_response.json()
            print(f"\n🔍 Информация от Ollama:")
            for m in models_data.get('models', []):
                size_gb = m.get('size', 0) / 1024**3
                print(f"   - {m['name']}: {size_gb:.2f} GB в памяти")
    except:
        pass
    
    # Рекомендации
    print(f"\n💡 РЕКОМЕНДАЦИИ:")
    if total_increase < 2:
        print("   ✅ Отлично! Памяти используется очень мало")
    elif total_increase < 4:
        print("   👍 Хорошо, но можно оптимизировать")
    else:
        print("   ⚠️ Многовато, рассмотрите оптимизацию:")
        print("      - Уменьшите размер контекста (num_ctx)")
        print("      - Используйте более легкую модель")
        print("      - Очищайте историю старых диалогов")

if __name__ == "__main__":
    # Убедитесь, что requests установлен
    try:
        import requests
    except ImportError:
        print("Устанавливаю requests...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'requests'])
        import requests
    
    test_with_memory_tracking()