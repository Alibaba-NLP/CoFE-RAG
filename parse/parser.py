from typing import Optional, List, Mapping, Any, Dict
from abc import ABC, abstractmethod
import os
import shutil
import logging

from utils import truncate_filename

class Parser(ABC):
    @property
    @abstractmethod
    def supported_input_formats(self) -> List[str]:
        """
        The supported formats for the input file.
        """
        pass

    def __init__(self, config, inp_folder):
        self.input_file_formats = config["input_file_suffix"]
        self.output_file_format = 'document'
        self.file_exclude = config.get("file_exclude", [])
        self.copy_file_if_format_not_in_suffix_list = config.get("copy_file_if_format_not_in_suffix_list", False)
        self.validate_input_format()

    def validate_input_format(self):
        for input_format in self.input_file_formats:
            if input_format not in self.supported_input_formats:
                raise ValueError(f"Input format {input_format} not supported. Valid formats are {self.supported_input_formats}")

    @abstractmethod
    def parse_file(self, input_file: str, output_file: str) -> bool:
        """
        Parse the input file and write the parsed output to the output file.
        """
        pass

    def process(self, input_folder: str, output_folder: str):
        """
        Parse all files in the input folder and write the parsed output to the output folder.
        """
        files = os.listdir(input_folder)
        processed = 0
        files_copied = []
        files_skipped_supported = []
        files_skipped_excluded = []
        files_failed = []
        for file in files:
            processed += 1
            input_file = os.path.join(input_folder, file)
            suffix = input_file.split('.')[-1]
            if file.split('/')[-1] in self.file_exclude:
                logging.info(f"Skipping {input_file} as it is in the exclude list")
                files_skipped_excluded.append(input_file)
                continue
            if suffix not in self.input_file_formats:
                if self.copy_file_if_format_not_in_suffix_list:
                    output_file = os.path.join(output_folder, file)
                    shutil.copy(input_file, output_file)
                    files_copied.append(input_file)
                    logging.info(f"Copying {input_file} to {output_file}")
                else:
                    files_skipped_supported.append(input_file)
                    logging.info(f"Skipping {input_file} as it is not supported")
                continue
            output_file = os.path.join(output_folder, file + f'.{self.output_file_format}')
            output_file = truncate_filename(output_file)
            if not os.path.exists(output_file):
                logging.info(f"Parsing ({processed}/{len(files)}) {input_file}")
                success = self.parse_file(input_file, output_file)
                if not success:
                    files_failed.append(input_file)
            else:
                logging.info(f"Skipping {input_file} as it is already parsed")
        logging.info(f"Processed {processed} files")
        logging.info(f"Files copied: {files_copied}")
        logging.info(f"Files skipped as they are not supported: {files_skipped_supported}")
        logging.info(f"Files skipped as they are in the exclude list: {files_skipped_excluded}")
        logging.info(f"Files failed: {files_failed}")
        # logging to output_folder/log.txt
        with open(os.path.join(output_folder, 'log.txt'), 'w') as f:
            f.write(f"Processed {processed} files\n")
            f.write(f"Files copied: {files_copied}\n")
            f.write(f"Files skipped as they are not supported: {files_skipped_supported}\n")
            f.write(f"Files skipped as they are in the exclude list: {files_skipped_excluded}\n")
            f.write(f"Files failed: {files_failed}\n")
        return output_folder    