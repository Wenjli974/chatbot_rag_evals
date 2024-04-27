from configparser import ConfigParser
from dotenv import load_dotenv
import numpy as np
import utils
import os
import openai
import nest_asyncio
from trulens_eval import Feedback, TruLlama
from trulens_eval import Tru
from model_setting import ModelSetting
from trulens_eval import OpenAI
from trulens_eval.feedback import Groundedness
from llama_index.core.query_engine import BaseQueryEngine
import pandas as pd
from IPython.display import display


class RAGEvaluation:
    def __init__(self, tru,config_path='config.ini', app_id=None):
        config = ConfigParser()
        config.read(config_path)
        self.File = config['File']
        self.env_path = config['Env_Path']['env_path']+'.env'
        _=load_dotenv(self.env_path)
        self.api_version = os.getenv('api_version')
        self.api_key = os.getenv('api_key')
        self.input_dir = self.File["input_directory"]
        self.output_dir = self.File["output_directory"]
        self.index_dir = self.File['index_directory']
        self.model = 'gpt-3.5-turbo'
        self.embedding = 'BAAI/bge-large-zh-v1.5'
        self.app_id = app_id

    def calc_metrics(self):
        print("start metrics calculation")
        self.context_selection = TruLlama.select_source_nodes().node.text
        self.provider = OpenAI()

        self.f_qa_relevance = Feedback(
            self.provider.relevance_with_cot_reasons,
            name="Answer Relevance"
            ).on_input_output()
        
        self.f_qs_relevance = (
             Feedback(self.provider.qs_relevance,
             name="Context Relevance")
                    .on_input()
                    .on(self.context_selection)
                    .aggregate(np.mean)
                    )
        self.grounded = Groundedness(groundedness_provider=self.provider)
        self.f_groundedness = (
            Feedback(self.grounded.groundedness_measure_with_cot_reasons,
                    name="Groundedness"
                    )
                    .on(self.context_selection)
                    .on_output()
                    .aggregate(self.grounded.grounded_statements_aggregator)
                    )
        print("end metrics calculation")

    def bulid_evaluate(self,tru,query_engine):
        print("start evaluation")
        self.tru_recorder = TruLlama(
            query_engine,
            app_id="App_3",
            feedbacks=[
                        self.f_qa_relevance,
                        self.f_qs_relevance,
                        self.f_groundedness
                        ]
        )
        self.eval_questions = []
        
        with open('eval_questions.txt', 'r',encoding='utf-8') as file:
            for line in file:
                print(line)
                # Remove newline character and convert to integer
                item = line.strip()
                self.eval_questions.append(item)

        nest_asyncio.apply()
        for question in self.eval_questions:
             with self.tru_recorder as recording:
                query_engine.query(question)
        

        tru.run_dashboard()

        records, feedback = tru.get_records_and_feedback(app_ids=[])
        print("end evaluation")

        records[['input', 'output']] = records[['input', 'output']].applymap(lambda x: x.encode('utf-8').decode('utf-8'))
        records[["input", "output"] + feedback]
       
        print(records.head(n=20))
        records.to_excel("records.xlsx")

        print("start leaderboard")     
        tru.get_leaderboard(app_ids=[])
        tru.run_dashboard()
     