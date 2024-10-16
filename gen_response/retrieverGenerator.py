import os
import sys
from typing import Optional, List, Mapping, Any, Dict

from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.schema import TextNode
from llama_index.core.settings import llm_from_settings_or_context, Settings

from gen_response.baseGenerator import BaseGenerator
from query_engine import RetrieverQueryEngine


class RetrieverGenerator(BaseGenerator):
    def set_top_n(self):
        if self.query_engine is not None:
            self.query_engine._node_postprocessors[0].top_n = self.top_n

    def load_query_engine(self):
        if self.search_query_engine is None:
            nodes = [TextNode(text="virtual")]
            useless_retriever = BM25Retriever.from_defaults(nodes=nodes)
            query_engine = RetrieverQueryEngine(
                retriever=useless_retriever
            )
            return query_engine
        else:
            self.search_query_engine._response_synthesizer = get_response_synthesizer(
                response_mode = "simple_summarize",
                llm=Settings.llm,
                callback_manager=self.search_query_engine._retriever.get_service_context()
            )
            return self.search_query_engine
