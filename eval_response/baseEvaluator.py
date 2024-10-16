from typing import Optional, List, Mapping, Any, Dict
from abc import ABC, abstractmethod
import os
import logging
import json
from tqdm import tqdm
import concurrent.futures
from llama_index.core.evaluation.base import BaseEvaluator, EvaluationResult
from llama_index.core.base.response.schema import Response
from llama_index.core.schema import NodeWithScore, TextNode
from utils import set_llm


class BaseEvaluator(ABC):

    def __init__(self, config, inp_folder):
        self.base_llm = config["base_llm"]
        self.thread_num = config.get("thread_num", 1)
        self.search_eval_results = config.get("search_eval_results", "")

    @property
    @abstractmethod
    def evaluators(self) -> List[BaseEvaluator]:
        """
        The used evaluators.
        """
        pass

    def process(self, input_folder: str, output_folder: str):
        if os.path.exists(os.path.join(output_folder, "eval_results.json")):
            logging.info(f"Output folder {output_folder} already exists, remove it.")
            os.system(f"rm -rf {output_folder}")
            os.system(f"mkdir {output_folder}")

        rag_dataset = json.load(open(os.path.join(input_folder, "predictions.json")))

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
                    # 将reference answer放在eval_result附近
                    reference_answer = example.get("reference_answer", "")
                    del example["reference_answer"]
                    example["reference_answer"] = reference_answer
                    example["eval_result"] = rsps
                    if "recall_results" in example:
                        del example["recall_results"]
                    response_list.append(example)
                    example_idx += 1
        else:
            for example in tqdm(rag_dataset['examples']):
                rsps = self.process_example(example)
                # 将reference answer放在eval_result附近
                reference_answer = example.get("reference_answer", "")
                del example["reference_answer"]
                example["reference_answer"] = reference_answer
                example["eval_result"] = rsps
                if "recall_results" in example:
                    del example["recall_results"]
                response_list.append(example)

        statistic = self.do_statistic(response_list)

        if self.search_eval_results:
            parent_dir = os.path.dirname(input_folder)
            search_results = json.load(open(os.path.join(parent_dir, self.search_eval_results, 'eval_results_detail.json')))
            example_idx = 0
            for example in search_results:
                search_eval = example["eval_results"]["hit_by"]
                response_list[example_idx]["eval_result"]["search_eval"] = search_eval
                example_idx += 1
        # save the evaluation results
        with open(os.path.join(output_folder, "eval_results.json"), "w") as f:
            json.dump(response_list, f, indent=2, ensure_ascii=False)
        # save satistic results to csv 
        with open(os.path.join(output_folder, "statistic.csv"), "w") as f:
            for key, value in statistic.items():
                f.write(f"{key},{value:.4f}\n")
        return output_folder

    def process_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        ers = {}
        query = example["query"]
        source_nodes = []
        for node in example["predictions"]["source_nodes"]:
            source_nodes.append(NodeWithScore(node=TextNode.from_dict(node["node"]), score=node["score"]))
        response = Response(example["predictions"]["response"], source_nodes, example["predictions"]["metadata"])

        for evaluator in self.evaluators:
            while True:
                try:
                    # print(evaluator, query, response)
                    er = evaluator.evaluate_response(query, response, reference=example.get("reference_answer", ""))
                    break
                except Exception as e:
                    logging.error(f"Error in evaluating {evaluator.__class__.__name__} for example {query}: {e}")
                    continue
            ers[evaluator.__class__.__name__] = json.loads(er.json())
        return ers

    def do_statistic(self, response_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Do statistic on the evaluation results.
        """
        total = len(response_list)
        valid = {}
        passing_cnt = {}
        score_cnt = {}
        for example in response_list:
            for evaluator_name, eval_result in example["eval_result"].items():
                if evaluator_name not in valid:
                    valid[evaluator_name] = 0
                    passing_cnt[evaluator_name] = 0
                    score_cnt[evaluator_name] = 0
                if eval_result["passing"]:
                    passing_cnt[evaluator_name] += 1
                score_cnt[evaluator_name] += eval_result["score"] if eval_result["score"] else 0
                if eval_result["invalid_result"]:
                    logging.warning(f"Invalid result {eval_result['invalid_reason']} for example {eval_result['query']}")
                else:
                    valid[evaluator_name] += 1
        result = {}
        for metric_name, metric_dict in zip(["valid", "passing", "score"], [valid, passing_cnt, score_cnt]):
            for evaluator_name, cnt in metric_dict.items():
                result[f"{evaluator_name}_{metric_name}"] = cnt / total
        return result

