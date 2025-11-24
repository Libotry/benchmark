import re
import string

from ais_bench.benchmark.openicl.icl_evaluator.icl_base_evaluator import BaseEvaluator
from ais_bench.benchmark.registry import ICL_EVALUATORS
from ais_bench.benchmark.utils.logging import AISLogger

logger = AISLogger()

@ICL_EVALUATORS.register_module("leval_code_u_evaluator")
class CodeUEvaluator(BaseEvaluator):
    """Custom evaluator for L-Eval Code U dataset."""
    
    def score(self, predictions, references):
        """Evaluate Code U dataset predictions against ground truth references.

        This method implements the custom evaluation logic for the L-Eval Code U dataset,
        which requires specialized processing to extract code execution outputs from
        model responses and compare them with expected results.

        The evaluation process involves:
        1. Input validation (checking prediction-reference length match)
        2. General text postprocessing to clean raw model outputs
        3. Code-specific output extraction from predictions
        4. Reference processing and normalization
        5. Accuracy calculation based on normalized string matching

        Args:
            predictions (List[str]): List of model-generated responses for Code U tasks.
                Each prediction may contain explanatory text mixed with code output.
            references (List[str]): List of ground truth answers corresponding to the
                predictions. Each reference contains the expected code execution result.

        Returns:
            Dict[str, Union[float, str]]: Evaluation results containing:
                - 'accuracy' (float): Percentage accuracy (0-100) based on correct
                  extractions and matches
                - 'error' (str): Error message if predictions and references have
                  mismatched lengths (only returned on validation failure)

        Raises:
            No exceptions are raised; validation errors are returned in the result dict.

        Note:
            The evaluation uses fuzzy string matching after normalization to handle
            minor formatting differences between predictions and references. The
            normalization process removes articles, punctuation, and standardizes
            whitespace while preserving the semantic meaning of code outputs.
        """

        if len(predictions) != len(references):
            return {'error': 'predictions and references have different length'}
     
        correct = 0
        total = len(predictions)
        
        for pred, ref in zip(predictions, references):
            # Apply Code U specific extraction logic
            pred_extracted = self._extract_code_output(pred, ref)
            ref_processed = self._process_code(ref)

            # Compare extracted prediction with processed reference
            if self._is_correct(pred_extracted, ref_processed):
                correct += 1
        
        accuracy = (correct / total) * 100
        return {'accuracy': accuracy}
    
    def _extract_code_output(self, prediction: str, reference: str) -> str:
        gt_len = len(reference.split())
        response = self._process_code(prediction)
        
        # Remove explanatory phrases
        response = response.replace("will be", "").replace("of the code", "")
        response = response.replace("is", "").replace("would be", "")
        response = response.replace("the value of", "").replace("the result of", "")
        response = response.replace("printed", "")
        
        # Extract final output
        if "the final output" in response:
            response = response.split("the final output")[-1]
            res = re.split(r'\s+', response)[:(gt_len+3)]
        else:
            res = re.split(r'\s+', response)[-(gt_len+3):]
        
        return " ".join(res)
    
    def _process_code(self, response: str) -> str:
        # Apply any reference-specific processing
        response = re.sub(r'\s+', ' ', response)
        response = response.replace(",", "").replace("'", "").replace("\\", "")
        response = response.replace(".0", "").replace("] [", "][")
        response = response.replace("[ [", "[[").replace("] ]", "]]")
        return response.strip()
    
    def _is_correct(self, prediction: str, reference: str) -> bool:
        return self._normalize_answer(prediction) == self._normalize_answer(reference)
    
    def _normalize_answer(self, s: str) -> str:
        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))


@ICL_EVALUATORS.register_module("leval_sci_fi_evaluator")
class SciFiEvaluator(BaseEvaluator):
    """Evaluator for L-Eval Sci-Fi dataset (loyalty only).

    The Sci-Fi task asks for True/False judgments. We only evaluate the
    "loyalty" judgment based on the model response and the reference label,
    ignoring any "fact" judgment that may also appear.
    """

    def score(self, predictions, references):
        """Compute loyalty accuracy only.

        Args:
            predictions (List[str]): Model responses containing loyalty judgments.
            references (List[str]): Ground-truth labels with loyalty info.

        Returns:
            Dict[str, float]: {'accuracy': loyalty_accuracy_percentage}
        """

        if len(predictions) != len(references):
            return {'error': 'predictions and references have different length'}

        total = len(predictions)
        correct_loyalty = 0

        for pred, ref in zip(predictions, references):
            pred_loyalty = self._extract_loyalty_from_prediction(pred)
            ref_loyalty = self._extract_loyalty_from_reference(ref)

            if ref_loyalty in ("true", "false") and pred_loyalty == ref_loyalty:
                correct_loyalty += 1

        accuracy = (correct_loyalty / total) * 100 if total > 0 else 0.0
        return {'accuracy': accuracy}

    def _extract_loyalty_from_prediction(self, response: str) -> str:
        """Extract 'true'/'false' from the loyalty segment of the response.

        We treat everything before "[fact:" as the loyalty segment if present;
        otherwise, the whole response is scanned. Returns 'true'/'false' or
        '<error>' if not found.
        """
        text = (response or '').lower()
        if "[fact:" in text:
            loyalty = text.split("[fact:", 1)[0]
        else:
            loyalty = text

        # Prefer exact 'true'/'false' first
        for word in loyalty.split():
            w = word.strip().lower().strip('.,;:!')
            if w == 'true':
                return 'true'
            if w == 'false':
                return 'false'
        # Then fallback to substring contains (to allow TRUE/FALSE inside punctuation)
        for word in loyalty.split():
            w = word.lower()
            if 'true' in w:
                return 'true'
            if 'false' in w:
                return 'false'
        return "<error>"

    def _extract_loyalty_from_reference(self, reference: str) -> str:
        """Extract 'true'/'false' from the reference text.

        We scan the reference for the first occurrence of 'true' or 'false'.
        """
        text = (reference or '').lower()
        for word in text.split():
            if "true" in word:
                return "true"
            if "false" in word:
                return "false"
        return "<error>"