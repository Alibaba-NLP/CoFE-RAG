from typing import Optional, List, Mapping, Any, Dict
from abc import ABC, abstractmethod
import os
import logging
import json
from tqdm import tqdm
import concurrent.futures

from llama_index.core.schema import NodeWithScore, BaseNode, MetadataMode
from llama_index.core.indices.query.schema import QueryBundle

from utils import set_mit_llm
from query_engine import BaseQueryEngine
from format_converter import nodefile2node
from dataset_filters import filters_registry

class BaseSearcher(ABC):
    show_progress = True

    @abstractmethod
    def load_query_engine(self, nodes: List[BaseNode]) -> BaseQueryEngine:
        """
        Load the query engine from nodes.
        """
        pass

    def __init__(self, config, inp_folder):
        self.remove_if_exists = config.get("remove_if_exists", False)
        self.thread_num = config.get("thread_num", 1)
        self.input_folder = inp_folder
        self.excluded_embed_metadata_keys = config.get('excluded_embed_metadata_keys', None)
 
        set_mit_llm()
        self.nodes = self.load_nodes(inp_folder)
        self.query_engine = self.load_query_engine(self.nodes)
        self.dataset_filter = filters_registry[config.get('dataset_filter', 'no_filter')]

    def process(self, input_folder: str, output_folder: str):
        if os.path.exists(os.path.join(output_folder, "recall_results.json")):
            if self.remove_if_exists:
                logging.info(f"Output folder {output_folder} already exists, removing it.")
                os.system(f"rm -rf {output_folder}")
                os.system(f"mkdir {output_folder}")
            else:
                logging.info(f"Output folder {output_folder} already exists, skipping.")
                return output_folder
 
        parent_folder = os.path.dirname(input_folder)
        rag_dataset_raw = json.load(open(os.path.join(parent_folder, "rag_dataset.json")))
        rag_dataset = {'examples': []}
        for example in rag_dataset_raw['examples']:
            if self.dataset_filter(example):
                rag_dataset['examples'].append(example)

        recall_results_list = []
        if self.thread_num > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.thread_num) as executor:
                task = []
                for example in rag_dataset['examples']:
                    query = example["query"]
                    query_bundle = QueryBundle(query_str=query)
                    task.append(executor.submit(self.query_engine.retrieve, query_bundle))
                example_idx = 0
                for t in tqdm(task):
                    example = rag_dataset['examples'][example_idx]
                    recall_results = t.result()
                    example["recall_results"] = self.nodes2dict(recall_results)
                    recall_results_list.append(example)
                    example_idx += 1
        else:
            for example in tqdm(rag_dataset['examples']):
                query = example["query"]
                query_bundle = QueryBundle(query_str=query)
                recall_results = self.query_engine.retrieve(query_bundle)
                example["recall_results"] = self.nodes2dict(recall_results)
                recall_results_list.append(example)

        output_file = os.path.join(output_folder, "recall_results.json")
        with open(output_file, "w") as f:
            json.dump(recall_results_list, f, indent=2, ensure_ascii=False)

        # save offline data for evaluation
        if not os.path.exists(os.path.join(output_folder, "parsed_files.json")):
            self.save_parsed_files(self.nodes, os.path.join(output_folder, "parsed_files.json"))

        return output_folder

    def save_parsed_files(self, parsed_files, out_file):
        parsed_files_fmt = []
        for node in parsed_files:
            node = node.to_dict()
            if 'embedding' in node:
                del node['embedding']
            parsed_files_fmt.append(node)
        json.dump(parsed_files_fmt, open(out_file, 'w'), indent=2, ensure_ascii=False)

    def load_nodes(self, input_folder):
        files = os.listdir(input_folder)
        parsed_files = []
        processed = 0
        for file in files:
            processed += 1
            input_file = os.path.join(input_folder, file)
            suffix = input_file.split('.')[-1]
            if suffix != 'node':
                logging.info(f"Skipping {input_file} as it is not supported")
                continue
            logging.info(f"Parsing ({processed}/{len(files)}) {input_file}")
            nodes = nodefile2node(input_file)
            if self.excluded_embed_metadata_keys is not None:
                for node in nodes:
                    node.excluded_embed_metadata_keys = self.excluded_embed_metadata_keys
                if len(nodes) > 0:
                    example_node = nodes[0]
                    logging.info(f"Excluded keys: {example_node.excluded_embed_metadata_keys}\nExample MetadataMode.EMBED: {example_node.get_content(MetadataMode.EMBED)}")
            parsed_files.extend(nodes)
        return parsed_files

    def nodes2dict(self, nodes: NodeWithScore) -> List[Dict[str, Any]]:
        resp_dict = {
            "response": None,
            "source_nodes": [],
            "metadata": None
        }
        for node in nodes:
            resp_dict["source_nodes"].append(node.to_dict())
        return resp_dict