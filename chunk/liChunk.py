import requests
import json

from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels,
    DashScopeTextEmbeddingType,
)
from llama_index.core.extractors import TitleExtractor
from ingestion.baseIngestion import BaseIngestion
from format_converter import documentfile2document, idpfile2document, text2document
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.bridge.pydantic import Field



class LiChunk(BaseIngestion):

    def __init__(self, config, inp_folder):
        self.chunk_size = config["chunk_size"]
        self.overlap_size = config["overlap_size"]
        self.embed_model_name = config["embed_model_name"]
        super().__init__(config, inp_folder)

    def load_pipeline(self) -> IngestionPipeline:
        """
        build ingestion pipeline
        """
        splitter = SentenceSplitter(
            include_metadata=True, include_prev_next_rel=True,
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap_size,
            separator=' ',       
            paragraph_separator='\n\n\n', secondary_chunking_regex='[^,.;。？！]+[,.;。？！]?')
        if self.embed_model_name == 'online':
            embed_model = DashScopeEmbedding(
                model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
                text_type=DashScopeTextEmbeddingType.TEXT_TYPE_DOCUMENT,
            )
        else:
            embed_model = HuggingFaceEmbedding(model_name=self.embed_model_name)

        pipeline = IngestionPipeline(
            transformations=[
                splitter,
                embed_model
            ]
        )
        return pipeline
