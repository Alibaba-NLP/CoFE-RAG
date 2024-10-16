from typing import Optional, List, Mapping, Any, Dict
from abc import ABC, abstractmethod
import os
import logging
import json
from tqdm import tqdm
import concurrent.futures

from llama_index.core.evaluation.base import BaseEvaluator
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.indices.query.schema import QueryBundle
from llama_index.core.schema import NodeWithScore, TextNode

from utils import import_class, set_mit_llm
from query_engine import BaseQueryEngine

class BaseGenerator(ABC):

    @abstractmethod
    def load_query_engine(self) -> BaseQueryEngine:
        """
        Load the query engine from the given folder.
        """
        pass

    @abstractmethod
    def set_top_n(self):
        """
        use top_n nodes to synthesize the response
        """
        pass

    def __init__(self, config, inp_folder):
        # 假如从文件中加载搜索结果，则不需要加载search_query_engine
        self.searcher_config_name = config.get("searcher_config_name", "")
        self.base_llm = config.get("base_llm", None)
        self.thread_num = config.get("thread_num", 1)
        self.remove_if_exists = config.get("remove_if_exists", False)
        self.search_cache_file = config.get("search_cache_file", '')
        self.top_n = config.get("top_n", 1)
        self.input_folder = inp_folder
        if self.base_llm:
            self.set_llms()
        if self.search_cache_file == '':
            self.search_query_engine = self.load_search_query_engine()
            self.set_top_n()
        else:
            self.search_query_engine = None
        assert self.search_query_engine is not None or self.search_cache_file != ''
        self.query_engine = self.load_query_engine()

    def set_llms(self):
        set_mit_llm(self.base_llm)

    def load_search_query_engine(self):
        if self.searcher_config_name == "":
            return "No search query engine specified."
        operator_config_path = os.path.join("search", "config", f"{self.searcher_config_name}.json")
        operator_config = json.load(open(operator_config_path))
        class_name = operator_config["class_name"]
        class_file = operator_config["class_file"]
        module_path = f"search.{class_file}"

        # Dynamically importing the operator class
        OpClass = import_class(module_path, class_name)
        
        # Instantiate the operator with its configuration
        operator = OpClass(operator_config, self.input_folder)

        return operator.query_engine

    def process(self, input_folder: str, output_folder: str):
        if os.path.exists(os.path.join(output_folder, "predictions.json")):
            if self.remove_if_exists:
                logging.info(f"Output folder {output_folder} already exists, removing it.")
                os.system(f"rm -rf {output_folder}")
                os.system(f"mkdir {output_folder}")
            else:
                logging.info(f"Output folder {output_folder} already exists, skipping.")
                return output_folder

        # 找到input_folder的父亲目录
        parent_folder = os.path.dirname(input_folder)
        # 使用存储好的搜索结果
        if self.search_cache_file != '':
            rag_dataset = {"examples": json.load(open(f'{input_folder}{self.search_cache_file}'))}
        else:
            rag_dataset = json.load(open(os.path.join(parent_folder, "rag_dataset.json")))
        
        # 对每个example进行检索，尽量返回”召回“而不是”精排“结果方便debug
        response_list = []
        if self.thread_num > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.thread_num) as executor:
                task = []
                for example in rag_dataset['examples']:
                    task.append(executor.submit(self.process_example, example))
                example_idx = 0
                for t in tqdm(task):
                    rsps = t.result()
                    example = rag_dataset['examples'][example_idx]
                    example["predictions"] = self.response2dict(rsps)
                    response_list.append(example)
                    example_idx += 1
        else:
            for example in tqdm(rag_dataset['examples']):
                rsps = self.process_example(example)
                example["predictions"] = self.response2dict(rsps)
                response_list.append(example)

        # 输出推理结果
        output_file = os.path.join(output_folder, "predictions.json")
        with open(output_file, "w", encoding='utf-8') as f:
            json.dump({"examples": response_list}, f, indent=2, ensure_ascii=False)
        return output_folder

    def response2dict(self, response: RESPONSE_TYPE) -> Dict[str, Any]:
        if hasattr(response, "get_response"):
            response = response.get_response()
        resp_dict = {
            "response": response.response,
            "source_nodes": [],
            "metadata": response.metadata
        }
        # print(response.response)
        for node in response.source_nodes:
            resp_dict["source_nodes"].append(node.to_dict())
        return resp_dict

    def process_example(self, example: Dict[str, Any]):
        nodes = example.get('recall_results', None)
        if nodes is None:
            response = self.query_engine.query(example["query"])
        else:
            nodes = nodes['source_nodes'][:self.top_n]
            nodes_fmt = []
            for node in nodes:
                score = node['score']
                node = TextNode.from_dict(node['node'])
                nodes_fmt.append(NodeWithScore(node=node, score=score))
            query = example["query"]
            query_bundle = QueryBundle(query_str=query)
            response = self.query_engine.synthesize(query_bundle, nodes_fmt)
        return response