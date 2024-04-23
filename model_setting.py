import logging
import os
import shutil
import sys
from configparser import ConfigParser

from llama_index.core.callbacks import llama_debug, LlamaDebugHandler
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.retrievers import QueryFusionRetriever
from IPython.core.display import display, Markdown
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from llama_index.core import Settings, SimpleDirectoryReader, Document, ServiceContext, VectorStoreIndex, \
    load_index_from_storage, StorageContext, get_response_synthesizer, PromptTemplate
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank, LLMRerank
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.legacy.retrievers import BM25Retriever
from llama_index.llms.openai import OpenAI
from ragas import RunConfig
from ragas.testset import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager


class ModelSetting:
    def __init__(self, config_path='config.ini'):
        self.test_df = None
        self.testset = None
        self.system_prompt_input = None
        self.system_prompt = None
        self.client = None
        self.rerank = None
        self.postproc = None
        self.sentence_window_engine = None
        self.response = None
        self.index = None
        self.sentence_context = None
        self.node_parser = None
        self.documents = None
        self.llm_model = None
        self.embedding_model = None
        self.langchain_model = None
        self.langchain_embed_model = None
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
        self.SYSTEM_PROMPT = """身份：专业的人事政策机器人，
                        名字：Jin，
                        语言：中文，英文
                        任务：理解用户的问题并根据上下文进行答复，如果用户使用中文，请使用中文回答。"""

    def initialize_model(self):
        print("model initializing")
        self.llm_model = OpenAI(
            model=self.model,
            api_key=self.api_key,
            system_prompt=self.SYSTEM_PROMPT,
            max_tokens=2000
        )

        self.embedding_model = HuggingFaceEmbedding(
            model_name=self.embedding,
        )

        Settings.llm = self.llm_model
        Settings.embed_model = self.embedding_model

        self.langchain_model = ChatOpenAI(
            model=self.model,
            temperature=1.0,
            api_key=self.api_key,
        )

        self.langchain_embed_model = HuggingFaceEmbeddings(
            model_name=self.embedding,
        )

        print("model initialized")

    def load_documents(self):
        print("Document loading")
        self.documents = SimpleDirectoryReader(
            self.input_dir,
            recursive=True,
            num_files_limit=200,
            required_exts=['.pdf', '.doc', '.xlsx', '.txt', '.docx'],
            filename_as_id=True).load_data(show_progress=True)
        print(len(self.documents))

        # merge pages into one
        self.documents = Document(text="\n\n".join([doc.text for doc in self.documents]))
        print("Docs loaded")
        return self.documents

    def build_nodeparser(self):
        # create the sentence window node parser
        self.node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=3,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )
        return self.node_parser

    def build_indexstore(self):
        self.sentence_context = ServiceContext.from_defaults(
            llm=self.llm_model,
            embed_model=self.embedding_model,
            #node_parser=self.node_parser,
            chunk_size=248,
            chunk_overlap=20,
        )
        if os.path.exists("./Vector_indexes"):
            shutil.rmtree("./Vector_indexes")

        self.index = VectorStoreIndex.from_documents([self.documents], service_context=self.sentence_context)
        # make vector store persistant
        self.index.storage_context.persist(persist_dir="./Vector_indexes")

    def load_indexstore(self):
        self.sentence_context = ServiceContext.from_defaults(
            llm=self.llm_model,
            embed_model=self.embedding_model,
            node_parser=self.node_parser,
            chunk_size=248,
            chunk_overlap=20,
        )
        if os.path.exists("./Vector_indexes"):
            self.index = load_index_from_storage(StorageContext.from_defaults(persist_dir="./Vector_indexes"),
                                                 service_context=self.sentence_context)

    def build_retriever(self, query):
        self.retriever = VectorIndexRetriever(index=self.index, similarity_top_k=3)
        # self.bm25_retriever = BM25Retriever.from_defaults(
        #     docstore=self.index.docstore, similarity_top_k=2
        # )
        # self.retriever = QueryFusionRetriever(
        #     [self.retriever],
        #     similarity_top_k=5,
        #     num_queries=1,  # set this to 1 to disable query generation
        #     mode = "reciprocal_rerank",
        #     use_async=True,
        #     verbose=True,
        # )
        print("Retriever builded")
        self.retrievals = self.retriever.retrieve(query)
        for node in self.retrievals:
            print(f"Score: {node.score:.2f} - {node.text}\n-----\n")

    def build_queryengine(self, query):
        query_str = query

        self.postproc = MetadataReplacementPostProcessor(
            target_metadata_key="window"
        )

        #link: https://huggingface.co/BAAI/bge-reranker-base
        self.rerank = SentenceTransformerRerank(
            top_n=3, model="BAAI/bge-reranker-base"
        )

        self.response_synthesizer = get_response_synthesizer(
            response_mode="refine",
            use_async=False,
            streaming=False,
        )

        self.query_engine = self.index.as_query_engine(
        #RetrieverQueryEngine(
            response_synthesizer=self.response_synthesizer,
           # retriever=self.retriever,
            similarity_top_k =3,
            node_postprocessors=[self.postproc,self.rerank],
        )

        self.response = self.query_engine.query(
            query_str
        )
        print(self.response)
        return self.query_engine

    def built_chatbot(self, system_prompt=None, user_prompt='Hi'):
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key)
        self.system_prompt = """
                身份：专业的客服机器人，
                名字：小瑾，
                语言：中文，英文
                任务：你可以根据用户的语言回答用户提出的各种问题。
                """
        self.system_prompt_input = str(self.system_prompt) + "##" + str(system_prompt)

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": self.system_prompt_input},
                {"role": "user", "content": user_prompt}, ],
            temperature=0,
            max_tokens=2000,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )
        return response.choices[0].message.content

    def generate_testset(self):
        test_size = int(self.RAG_File['testset_size'])
        print('initialising models')
        self.initialize_model()
        print('loading documents')
        self.load_documents()
        self.run_config = RunConfig(timeout=180, max_retries=60, max_wait=180)
        self.generator = TestsetGenerator.from_langchain(
            generator_llm=self.langchain_model,
            critic_llm=self.langchain_model,
            embeddings=self.langchain_embed_model,
            run_config=self.run_config,
        )
        self.testset = self.generator.generate_with_llamaindex_docs(
            self.documents,
            test_size=test_size,
            with_debugging_logs=False,
            distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25},
            run_config=self.run_config
        )

        self.test_df = self.testset.to_pandas()
        return self.test_df
