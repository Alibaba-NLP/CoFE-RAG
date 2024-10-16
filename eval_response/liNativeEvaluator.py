from typing import Optional, List, Mapping, Any, Dict

from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    CorrectnessEvaluator,
    RelevancyEvaluator,
)

from eval_response.baseEvaluator import BaseEvaluator
from utils import set_mit_llm

class LiNativeEvaluator(BaseEvaluator):
    @property
    def evaluators(self) -> List[BaseEvaluator]:
        """
        The used evaluators.
        """
        set_mit_llm(base_llm=self.base_llm)
        return [
            FaithfulnessEvaluator(),
            CorrectnessEvaluator(),
            RelevancyEvaluator()
        ]