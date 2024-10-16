from typing import Optional, List, Mapping, Any, Dict
from abc import ABC, abstractmethod
import os
import shutil
import logging
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from llama_index.core.ingestion import IngestionPipeline
from format_converter import documentfile2document

class BaseIngestion(ABC):
    show_progress = False

    def __init__(self, config, inp_folder):
        self.input_file_formats = ["document", "node"]
        self.output_file_format = "node"
        self.file_exclude = config.get("file_exclude", [])
        self.num_workers = config.get("num_workers", 1)
        self.pipeline = self.load_pipeline()

    @abstractmethod
    def load_pipeline(self) -> IngestionPipeline:
        """
        build ingestion pipeline
        """
        pass

    def process_file(self, input_file, input_folder, output_folder):
        suffix = input_file.split('.')[-1]
        file_path = os.path.join(input_folder, input_file)
        
        if input_file in self.file_exclude:
            logging.info(f"Skipping {file_path} as it is in the exclude list")
            return "exclude", file_path
        
        if suffix not in self.input_file_formats:
            logging.info(f"Skipping {file_path} as it is not supported")
            return "unsupported", file_path
        
        output_file_path = os.path.join(output_folder, '.'.join(input_file.split('.')[:-1]) + f'.{self.output_file_format}')
        
        if not os.path.exists(output_file_path):
            logging.info(f"Parsing {file_path}")
            
            try:
                documents = documentfile2document(file_path)
                # print(documents)
                nodes = self.pipeline.run(documents=documents, show_progress=self.show_progress)
                nodes_json = [node.to_dict() for node in nodes]
                with open(output_file_path, 'w') as json_file:
                    json.dump(nodes_json, json_file, indent=2, ensure_ascii=False)
            except Exception as e:
                logging.error(f"Failed to parse {file_path}: {e}")
                return "failed", file_path
        else:
            logging.info(f"Skipping {file_path} as it is already parsed")
            
        return "processed", file_path

    def process(self, input_folder: str, output_folder: str):
        files = os.listdir(input_folder)
        results = {
            "processed": [],
            "unsupported": [],
            "exclude": [],
            "failed": []
        }
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_file = {executor.submit(self.process_file, file, input_folder, output_folder): file for file in files}
            
            for future in as_completed(future_to_file):
                result_type, file = future.result()
                results[result_type].append(file)
        
        # 使用结果集 results 中的信息进行日志记录等后续操作
        logging.info(f"Processed {len(results['processed'])} files")
        logging.info(f"Files skipped as they are not supported: {results['unsupported']}")
        logging.info(f"Files skipped as they are in the exclude list: {results['exclude']}")
        logging.info(f"Files failed: {results['failed']}")

        with open(os.path.join(output_folder, 'log.txt'), 'w') as f:
            f.write(f"Processed {len(results['processed'])} files\n")
            f.write(f"Files skipped as they are not supported: {results['unsupported']}\n")
            f.write(f"Files skipped as they are in the exclude list: {results['exclude']}\n")
            f.write(f"Files failed: {results['failed']}\n")
        return output_folder
