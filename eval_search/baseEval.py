from typing import Optional, List, Mapping, Any, Dict
from abc import ABC, abstractmethod
import os
import logging
import json

class BaseEval(ABC):
    def __init__(self, config, input_folder):
        self.input_folder = input_folder
        self.top_k = config.get("top_k", 5)

    @abstractmethod
    def process(self, input_folder: str, output_folder: str):
        """
        Process the input folder and write the output to the output folder.
        """
        pass