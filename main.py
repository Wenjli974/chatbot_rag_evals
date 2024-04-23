from configparser import ConfigParser
from dotenv import load_dotenv
import os

from model_setting import ModelSetting
from rag_evaluation import RAGEvaluation
prompt="今天天气如何"


def task1():
    task = ModelSetting()
    print("start")
    task.initialize_model()
    task.build_nodeparser()
    task.load_indexstore()
    task.build_retriever(query=prompt)
    query_engine=task.build_queryengine(query=prompt)
    print(type(task.query_engine))
    eval=RAGEvaluation(app_id="rag v1.0")
    import nest_asyncio
    nest_asyncio.apply()
    eval.calc_metrics()
    eval.bulid_evaluate(task.query_engine)

    
    print("end")

def task2():
    task = ModelSetting()
    print("start")
    task.initialize_model()
    task.load_documents()
    task.build_nodeparser()
    task.build_indexstore()
    task.build_retriever(query=prompt)
    res=task.build_queryengine(query=prompt)
    # prompt = "你叫什么名字?"
    # res = task.built_chatbot(user_prompt=prompt)
    print("end")

def task3():
    task = ModelSetting()
    print("start")
    task.initialize_model()
    prompt = """
翻译中文    
What are the keys to building a career in AI?
How can teamwork contribute to success in AI?
What is the importance of networking in AI?
What are some good habits to develop for a successful career?
How can altruism be beneficial in building a career?
What is imposter syndrome and how does it relate to AI?
Who are some accomplished individuals who have experienced imposter syndrome?
What is the first step to becoming good at AI?
What are some common challenges in AI?
Is it normal to find parts of AI challenging??"""
    res = task.built_chatbot(user_prompt=prompt)
    print(res)
    print("end")

if __name__ == '__main__':
    task1()