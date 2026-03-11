import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Загрузка модели Saiga Llama 3
def load_saiga_model():
    model_name = "ilyagesev/saiga_llama3"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Используйте float16 для экономии памяти
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    # Создание pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.1,
        top_p=0.95,
        repetition_penalty=1.15
    )
    
    # Создание LangChain LLM
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

# Функция для загрузки JSON данных
def load_drug_data(json_file_path=None, json_data=None):
    if json_file_path:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    elif json_data:
        if isinstance(json_data, str):
            return json.loads(json_data)
        return json_data
    else:
        raise ValueError("Необходимо указать либо путь к файлу, либо JSON данные")


if __name__ == "__main__":
    json_path = "data\\extracted_data_all.json"
    # load_drug_data