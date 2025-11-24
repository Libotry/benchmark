import os

from datasets import Dataset, DatasetDict

from ais_bench.benchmark.openicl.icl_evaluator import BaseEvaluator
from ais_bench.benchmark.registry import LOAD_DATASET, ICL_EVALUATORS
from ais_bench.benchmark.datasets.utils.datasets import get_data_path
from ais_bench.benchmark.utils.logging.logger import AISLogger
from ais_bench.benchmark.utils.logging.error_codes import DSET_CODES
from ais_bench.benchmark.utils.logging.exceptions import AISBenchDataContentError

from .base import BaseDataset

logger = AISLogger()


@LOAD_DATASET.register_module()
class DAPOMathDataset(BaseDataset):
    """DAPO-math-17k dataset for RL reasoning evaluation.
    
    Data format:
    {
        "data_source": "math_dapo",
        "prompt": [{"content": "...", "role": "user"}],
        "ability": "MATH",
        "reward_model": {"ground_truth": "...", "style": "..."},
        "extra_info": {"index": "..."}
    }
    """

    @staticmethod
    def load(path, file_name=None, **kwargs):
        """Load DAPO-math-17k dataset from Parquet file.
        
        Args:
            path (str): Path to the dataset directory or file.
            file_name (str, optional): Name of the Parquet file. 
                If None, will look for 'dapo-math-17k.parquet' or all .parquet files.
            **kwargs: Additional arguments.
            
        Returns:
            DatasetDict: Dataset with 'test' split.
        """
        path = get_data_path(path, local_mode=True)
        logger.debug(f"Loading DAPO-math-17k dataset from: {path}")
        
        # Determine file path
        if file_name:
            file_path = os.path.join(path, file_name) if not os.path.isabs(file_name) else file_name
        elif os.path.isfile(path) and path.endswith('.parquet'):
            # If path is already a Parquet file, use it directly
            file_path = path
        else:
            # Try default name first
            default_path = os.path.join(path, 'dapo-math-17k.parquet')
            if os.path.exists(default_path):
                file_path = default_path
            else:
                # Look for any .parquet file in the directory
                if not os.path.isdir(path):
                    raise AISBenchDataContentError(
                        DSET_CODES.FILE_NOT_FOUND,
                        f"Path is not a directory or Parquet file: {path}"
                    )
                parquet_files = [f for f in os.listdir(path) if f.endswith('.parquet')]
                if not parquet_files:
                    raise AISBenchDataContentError(
                        DSET_CODES.FILE_NOT_FOUND,
                        f"No Parquet file found in {path}"
                    )
                if len(parquet_files) > 1:
                    logger.debug(f"Multiple Parquet files found, using first one: {parquet_files[0]}")
                file_path = os.path.join(path, parquet_files[0])
        
        if not os.path.exists(file_path):
            raise AISBenchDataContentError(
                DSET_CODES.FILE_NOT_FOUND,
                f"Dataset file not found: {file_path}"
            )
        
        # Load data from Parquet file using datasets library
        try:
            raw_dataset = Dataset.from_parquet(file_path)
            logger.debug(f"Loaded Parquet file with {len(raw_dataset)} rows")
        except Exception as e:
            raise AISBenchDataContentError(
                DSET_CODES.FILE_READ_ERROR,
                f"Failed to read Parquet file {file_path}: {e}"
            )
        
        # Process and transform data
        dataset = []
        for idx, row in enumerate(raw_dataset):
            try:
                # Extract prompt content
                if 'prompt' not in row:
                    raise AISBenchDataContentError(
                        DSET_CODES.DATA_INVALID_STRUCTURE,
                        f"Missing 'prompt' field at row {idx}"
                    )
                
                prompt_list = row['prompt']
                if not isinstance(prompt_list, list) or len(prompt_list) == 0:
                    raise AISBenchDataContentError(
                        DSET_CODES.DATA_INVALID_STRUCTURE,
                        f"Invalid 'prompt' format at row {idx}: expected non-empty list"
                    )
                
                # Extract content from prompt (usually the first item with role='user')
                prompt_content = None
                for item in prompt_list:
                    if isinstance(item, dict) and item.get('role') == 'user':
                        prompt_content = item.get('content', '')
                        break
                
                if prompt_content is None:
                    # Fallback: use first item's content
                    if isinstance(prompt_list[0], dict):
                        prompt_content = prompt_list[0].get('content', '')
                    else:
                        prompt_content = str(prompt_list[0])
                
                # Extract ground truth from reward_model
                if 'reward_model' not in row:
                    raise AISBenchDataContentError(
                        DSET_CODES.DATA_INVALID_STRUCTURE,
                        f"Missing 'reward_model' field at row {idx}"
                    )
                
                reward_model = row['reward_model']
                if not isinstance(reward_model, dict) or 'ground_truth' not in reward_model:
                    raise AISBenchDataContentError(
                        DSET_CODES.DATA_INVALID_STRUCTURE,
                        f"Invalid 'reward_model' format at row {idx}: missing 'ground_truth'"
                    )
                
                ground_truth = str(reward_model['ground_truth'])
                
                # Build dataset entry
                entry = {
                    'prompt': prompt_content,
                    'answer': ground_truth,
                    'data_source': row.get('data_source', 'math_dapo'),
                    'ability': row.get('ability', 'MATH'),
                    'extra_info': row.get('extra_info', {}),
                }
                
                dataset.append(entry)
                
            except AISBenchDataContentError:
                raise
            except Exception as e:
                logger.debug(f"Unexpected error processing row {idx}: {e}")
                continue
        
        if not dataset:
            raise AISBenchDataContentError(
                DSET_CODES.DATA_INVALID_STRUCTURE,
                f"No valid data entries found in {file_path}"
            )
        
        logger.debug(f"DAPO-math-17k dataset loaded: {len(dataset)} samples")
        
        # Create DatasetDict with test split
        dataset_dict = DatasetDict({
            'test': Dataset.from_list(dataset),
            'train': Dataset.from_list(dataset)  # Use same data for train (for few-shot examples)
        })
        
        return dataset_dict


@ICL_EVALUATORS.register_module()
class DAPOMathEvaluator(BaseEvaluator):
    """DAPO-math-17k evaluator for accuracy-based evaluation.
    
    Evaluation method: ACC (Accuracy)
    Each case's answer is taken from reward_model.ground_truth.
    Metric: accuracy = correct_count / total_count
    """

    def __init__(self):
        super().__init__()

    def is_equal(self, pred, refer):
        """Check if prediction matches reference.
        
        Args:
            pred (str): Prediction string.
            refer (str): Reference (ground truth) string.
            
        Returns:
            bool: True if prediction matches reference.
        """
        # Simple string comparison (case-insensitive, stripped)
        pred_str = str(pred).strip()
        refer_str = str(refer).strip()
        
        # Exact match (case-insensitive)
        if pred_str.lower() == refer_str.lower():
            return True
        
        # Try numeric comparison if both are numeric
        try:
            pred_num = float(pred_str)
            refer_num = float(refer_str)
            # Allow small floating point differences
            if abs(pred_num - refer_num) < 1e-6:
                return True
        except (ValueError, TypeError):
            pass
        
        return False

    def score(self, predictions, references):
        """Calculate accuracy score.
        
        Args:
            predictions (List[str]): List of predictions.
            references (List[str]): List of ground truth answers.
            
        Returns:
            dict: Evaluation results with 'accuracy' and 'details'.
        """
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different length',
                'predictions_len': len(predictions),
                'references_len': len(references)
            }
        
        correct = 0
        count = 0
        details = []
        
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            detail = {
                'pred': pred,
                'answer': ref,
                'correct': False
            }
            count += 1
            
            if self.is_equal(pred, ref):
                correct += 1
                detail['correct'] = True
            
            details.append(detail)
        
        accuracy = 100.0 * correct / count if count > 0 else 0.0
        
        result = {
            'accuracy': accuracy,
            'correct': correct,
            'total': count,
            'details': details
        }
        
        return result

