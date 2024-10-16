import os
import json
import logging
import csv
from llama_index.core.schema import TextNode
from eval_search.utils import clean_text
from eval_search.baseEval import BaseEval

class ScoredKeywordMatchEval(BaseEval):
    def __init__(self, config, input_folder):
        super(ScoredKeywordMatchEval, self).__init__(config, input_folder)
        self.eval_error_ratio = 0
        self.offline_doc = []
        self.load_offline_doc()

    def load_offline_doc(self):
        fpath = os.path.join(self.input_folder, "parsed_files.json")
        if os.path.exists(fpath):
            for node in json.load(open(fpath)):
                node['text'] = clean_text(node['text'])
                self.offline_doc.append({'node': node})
        else:
            logging.error(f"Parsed files not found in the input folder. Please check if {fpath} exists and the soft link is correct.")

    def find_gold_from_offline(self, fine_keywords, query='', error_ratio=0.15, return_results=False):
        return self.find_gold(self.offline_doc, fine_keywords, query, error_ratio, return_results, need_norm=False)

    def process(self, input_folder: str, output_folder: str):
        recall_results_list = json.load(open(os.path.join(input_folder, "recall_results.json")))
        eval_results_list = []
        stats = {"total": 0, "top_hit": 0, "recall_hit": 0, "offline_hit": 0}
        hit_priority = ["top_hit", "recall_hit", "offline_hit"]
        print(len(recall_results_list))
        for example in recall_results_list:
            eval_results = {}
            stats["total"] += 1
            query = example["query"]
            if len(example["fine_keywords"]) > 0 and type(example["fine_keywords"][0]) == str:
                fine_keywords = [clean_text(keyword) for keyword in example["fine_keywords"]]
            else:
                fine_keywords = [[clean_text(keyword) for keyword in keywords] for keywords in example["fine_keywords"]]
            recall_results = example["recall_results"]['source_nodes']
            coarse_keywords = [clean_text(keyword) for keyword in example["coarse_keywords"]]

            if not fine_keywords: continue
            if not coarse_keywords: continue

            hit_by = None
            gold_result, hit_score, keywords_missing, keywords_content = self.find_gold(recall_results[:self.top_k], fine_keywords, coarse_keywords, query, error_ratio=self.eval_error_ratio, return_results=True, need_norm=True, return_score=True)
   
            if gold_result:
                hit_by = "top_hit"
            if hit_by is None:
                gold_result = self.find_gold(recall_results, fine_keywords, coarse_keywords, query, error_ratio=self.eval_error_ratio, return_results=True, need_norm=True)
                if gold_result:
                    hit_by = "recall_hit"
            if hit_by is None:
                gold_result = self.find_gold_from_offline(fine_keywords, query, error_ratio=self.eval_error_ratio, return_results=True)
                if gold_result:
                    hit_by = "offline_hit"

            eval_results["hit_by"] = hit_by
            eval_results['hit_score'] = hit_score
            eval_results['keywords_missing'] = keywords_missing
            eval_results['keywords_content'] = keywords_content
            eval_results["gold_result"] = gold_result
            if hit_by is not None:
                for stat_update in hit_priority[hit_priority.index(hit_by):]:
                    stats[stat_update] += 1
            else:
                eval_results["hit_by"] = "no_hit"

            prompt = '\n\n'.join([TextNode.from_dict(r['node']).get_content() for r in recall_results[:self.top_k]])
            eval_results['prompt'] = prompt
            example["eval_results"] = eval_results
            del example["recall_results"]
            example["topk_results"] = recall_results[:self.top_k]
            eval_results_list.append(example)
        # 保存评估结果
        json.dump(eval_results_list, open(os.path.join(output_folder, "eval_results_detail.json"), "w"), indent=2, ensure_ascii=False)
        with open(os.path.join(output_folder, "eval_results_detail.csv"), 'w', newline='', encoding='utf-8') as detailfile:
            fieldnames = ['query', 'reference_answer', 'hit_by', 'hit_score', 'keywords_missing', 'keywords_content', 'prompt', 'gold_result', 'meta_info']
            writer = csv.DictWriter(detailfile, fieldnames=fieldnames)
            writer.writeheader()
            for example in eval_results_list:
                writer.writerow({
                    "query": example["query"],
                    "reference_answer": [], #example["reference_answer"],
                    "hit_by": example["eval_results"]["hit_by"],
                    "keywords_missing": example["eval_results"]["keywords_missing"],
                    "keywords_content": example["eval_results"]["keywords_content"],
                    'prompt': example["eval_results"]["prompt"],
                    "hit_score": example["eval_results"]["hit_score"],
                    "gold_result": example["eval_results"]["gold_result"],
                    "meta_info": example.get("meta_info", "")
                })
        # 统计stats每个指标除以total的百分百并保存为csv文件
        with open(os.path.join(output_folder, "eval_results_scores.csv"), 'w', newline='', encoding='utf-8') as scorefile:
            fieldnames = ['metric', 'value']
            writer = csv.DictWriter(scorefile, fieldnames=fieldnames)
            writer.writeheader()
            for key, value in stats.items():
                if key != "total":
                    writer.writerow({"metric": key, "value": "{:.2f}%".format(value / stats["total"] * 100)})
        return output_folder

    def find_gold(self, search_results, fine_keywords, coarse_keywords, query='', error_ratio=0.15, return_results=False, need_norm=True, return_score=False):
        idx = 0

        search_results_text = ''
        max_content = ''
        good_results = []
        for result in search_results:
            node = TextNode.from_dict(result['node'])
            combined_text = node.get_content()
            if need_norm:
                combined_text = clean_text(combined_text)
            flag = 0
            for core_keyword in coarse_keywords:
                if core_keyword.lower() in combined_text.lower():
                    flag = 1
            if flag:
                search_results_text += combined_text
                max_content += node.get_content()
                good_results.append(result)

        gold_result = []
        max_keyword_hit = 0

        keyword_hit = 0
        if len(fine_keywords) > 0 and type(fine_keywords[0]) == str:
            missing_key = [keyword for keyword in fine_keywords if not keyword.lower() in search_results_text.lower()]
        else:
            missing_key = []
            for keywords in fine_keywords:
                cur_mk = [keyword for keyword in keywords if not keyword.lower() in search_results_text.lower()]
                if len(cur_mk) == 0:
                    keyword_hit += 1
                missing_key += cur_mk
        max_keywords_missing = missing_key
        max_keyword_hit = keyword_hit / len(fine_keywords)
       
               

        if len(missing_key) == 0:
            if return_results:
                gold_result = good_results
            else:
                return True, 1, [], node.get_content()

        if return_results:
            if return_score:
                return gold_result, max_keyword_hit, max_keywords_missing, max_content
            else:
                return gold_result
        else:
            if return_score:
                return False, max_keyword_hit, max_keywords_missing, max_content
            else:
                return False


