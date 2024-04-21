from dotenv import load_dotenv
import os

from model_setting import ModelSetting
prompt="今天天气如何"

task = ModelSetting()
print("start")
task.initialize_model()
#task.load_documents()
task.build_nodeparser()
task.load_indexstore()
#task.build_indexstore()
task.build_retriever(query=prompt)
res=task.build_queryengine(query=prompt)
# prompt = "你叫什么名字?"
# res = task.built_chatbot(user_prompt=prompt)
print("end")

