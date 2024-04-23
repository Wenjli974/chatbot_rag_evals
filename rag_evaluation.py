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
from trulens_eval import OpenAI as fOpenAI
from trulens_eval.feedback import Groundedness
from llama_index.core.query_engine import BaseQueryEngine
import pandas as pd

tru = Tru()
tru.reset_database()

class RAGEvaluation:
    def __init__(self, config_path='config.ini', app_id=None):
        config = ConfigParser()
        config.read(config_path)
        self.RAG_File = config['RAG_File']
        self.env_path = config['Env_Path']['env_path']+'.env'
        _=load_dotenv(self.env_path)
        self.api_version = os.getenv('api_version')
        self.api_key = os.getenv('api_key')
        self.input_dir = self.RAG_File["RAG_input_directory"]
        self.output_dir = self.RAG_File["RAG_output_directory"]
        self.index_dir = self.RAG_File['RAG_index_directory']
        self.model = 'gpt-3.5-turbo'
        self.embedding = 'BAAI/bge-large-zh-v1.5'
        self.app_id = app_id

    def calc_metrics(self):
        print("start metrics calculation")
        context_selection = TruLlama.select_source_nodes().node.text
        provider = fOpenAI()

        self.f_qa_relevance = Feedback(
            provider.relevance_with_cot_reasons,
            name="Answer Relevance"
            ).on_input_output()
        
        self.f_qs_relevance = (
             Feedback(provider.qs_relevance,
             name="Context Relevance")
                    .on_input()
                    .on(context_selection)
                    .aggregate(np.mean)
                    )
        grounded = Groundedness(groundedness_provider=provider)
        self.f_groundedness = (
            Feedback(grounded.groundedness_measure_with_cot_reasons,
                    name="Groundedness"
                    )
                    .on(context_selection)
                    .on_output()
                    .aggregate(grounded.grounded_statements_aggregator)
                    )
        print("end metrics calculation")

    def bulid_evaluate(self,query_engine):
        print("start evaluation")
        self.tru_recorder = TruLlama(
            query_engine,
            kwargs={"app_id" : self.app_id,       
                    "feedbacks" : [ self.f_qa_relevance,
                                    self.f_qs_relevance,
                                    self.f_groundedness
                    ]}
        )
        self.eval_questions = []
        with open('eval_questions.txt', 'r',encoding='utf-8') as file:
            for line in file:
                # Remove newline character and convert to integer
                item = line.strip()
                self.eval_questions.append(item)

        for question in self.eval_questions:
             with self.tru_recorder as recording:
                query_engine.query(question)
        
        records, feedback = tru.get_records_and_feedback(app_ids=[])
        print(records.head(n=20))
        records.to_excel("records.xlsx")
        feedback.to_excel("feedback.xlsx")
        print("end evaluation")

        pd.set_option("display.max_colwidth", None)
        records[["input", "output"] + feedback]
        
        print("start leaderboard")
        tru.get_leaderboard(app_ids=[])
        tru.run_dashboard()