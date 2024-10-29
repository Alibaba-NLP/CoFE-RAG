import re
import json
import os
from typing import Optional, List, Mapping, Any, Dict
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.indices.query.schema import QueryBundle
from llama_index.core import VectorStoreIndex
from search.baseSearcher import BaseSearcher
from query_engine import RetrieverQueryEngine
import requests
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.bridge.pydantic import Field

class SimpleHybridSearcher(BaseSearcher):
    def __init__(self, config, inp_folder):
        self.rerank_size = config["rerank_size"]
        self.vector_ratio = config["vector_ratio"]
        self.embed_model_name = config['embed_model_name']
        self.rerank_model = config["rerank_model"]
        self.regenerate_emb = config.get("regenerate_emb", False)
        self.use_async = config.get("use_async", False)
        super(SimpleHybridSearcher, self).__init__(config, inp_folder)

    
    def load_query_engine(self, nodes):
        """
        Load the query engine from nodes.
        """
        node_postprocessors = self.load_node_postprocessors()
        retriever = self.load_retriever(nodes)
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=node_postprocessors
        )
        return query_engine

    def load_node_postprocessors(self):
        from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
        reranker = FlagEmbeddingReranker(
                    top_n=self.rerank_size,
                    model=self.rerank_model,
                    use_fp16=False
            )
        return [reranker]

    def load_retriever(self, nodes):
        """
        Load the retriever from the given folder.
        """
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        embed_model = HuggingFaceEmbedding(model_name=self.embed_model_name)
        if self.regenerate_emb:
            new_nodes = []
            for node in nodes:
                node.embedding = None
                new_nodes.append(node)
            nodes = new_nodes
        vector_index = VectorStoreIndex(nodes, embed_model=embed_model, show_progress=self.show_progress, use_async=self.use_async, insert_batch_size=2048)
        vector_retriever = vector_index.as_retriever(similarity_top_k=self.rerank_size)
        return vector_retriever

