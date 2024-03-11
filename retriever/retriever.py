# SETTING UP PIPELINE
#####################################################################################
import sys
sys.path.append('.')

import logging
from envs import *
from haystack import Pipeline
from haystack.schema import Answer
from haystack.nodes import PromptModel, PromptNode, PromptTemplate
from haystack.nodes import (
    BM25Retriever,
    EmbeddingRetriever,
    SentenceTransformersRanker,
    Docs2Answers,
)
from invocation_layer import HFInferenceEndpointInvocationLayer
from custom_plugins import DocumentThreshold
from . import retriever_database

logger = logging.getLogger(__name__)

class RetrieverPipeline:
    def __init__(self, document_store):
        if ENABLE_BM25:
            retriever = BM25Retriever(document_store=document_store, top_k=BM25_TOP_K)

        embedding_retriever = EmbeddingRetriever(
            document_store=document_store,
            embedding_model=EMBEDDING_MODEL,
            model_format="sentence_transformers",
            top_k=EMBEDDING_TOP_K,
        )

        document_store.update_embeddings(
            embedding_retriever, index="faq", batch_size=DB_BATCH_SIZE
        )

        faq_threshold = DocumentThreshold(threshold=FAQ_THRESHOLD)
        docs2answers = Docs2Answers()

        self.faq_pipeline = Pipeline()
        self.faq_params = {"EmbeddingRetriever": {"index": "faq"}}
        
        if ENABLE_BM25:
            self.faq_pipeline.add_node(
                component=retriever, name="Retriever", inputs=["Query"]
            )
            self.faq_params["Retriever"] = {"index": "faq"}

        self.faq_pipeline.add_node(
            component=embedding_retriever,
            name="EmbeddingRetriever",
            inputs=["Query" if not ENABLE_BM25 else "Retriever"],
        )
        self.faq_pipeline.add_node(
            component=faq_threshold, name="Threshold", inputs=["EmbeddingRetriever"]
        )
        self.faq_pipeline.add_node(
            component=docs2answers, name="Answer", inputs=["Threshold"]
        )

    def __call__(self, query, **kwargs):
        return self.run(query, **kwargs)

    def run(self, query, **kwargs):
        if "params" not in kwargs:
            kwargs["params"] = {}

        kwargs["params"].update(self.faq_params)
        faq_ans = self.faq_pipeline.run(query, **kwargs)

        # # Nếu trong trường hợp chạy database không ra kết quả thì cần phải chạy sang web API để lấy ra kết quả
        # if len(faq_ans["answers"]) == 0 or faq_ans["answers"][0].answer.strip() == "":
        #     kwargs["params"].update(self.web_params)
        #     web_ans = self.web_pipeline.run(query, **kwargs)

        #     if (
        #         len(web_ans["answers"]) == 0
        #         or web_ans["answers"][0].answer.strip() == ""
        #     ):
        #         chosen_ans = random.choice(DEFAULT_ANSWERS)
        #         web_ans["answers"].append(Answer(chosen_ans, type="other"))

        #     return web_ans

        return faq_ans
    

def setup_retriever_pipelines(args):
    # Re-import the configuration variables
    from rest_api import config  # pylint: disable=reimported
    from rest_api.controller.utils import RequestLimiter

    pipelines = {}
    document_store = retriever_database.initialize_db(args)

    # Load query pipeline & document store
    print("[+] Setting up pipeline...")
    pipelines["query_pipeline"] = RetrieverPipeline(document_store)
    pipelines["document_store"] = document_store

    return pipelines