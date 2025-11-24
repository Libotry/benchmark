from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.openicl.icl_evaluator import AccEvaluator
from ais_bench.benchmark.datasets.leval import LEvalGSM100Dataset
from ais_bench.benchmark.datasets.leval import gsm100_dataset_postprocess, gsm100_postprocess

LEval_gsm100_reader_cfg = {
    "input_columns": ['context', 'question'],
    "output_column": 'answer',
    "train_split": 'test',
    "test_split": 'test'
}

LEval_gsm100_infer_cfg = {
    "prompt_template": {
        "type": PromptTemplate,
        "template": {
            "begin": [
                {
                    "role": "SYSTEM",
                    "fallback_role": "HUMAN",
                    "prompt": "Given several question answer pairs, you need to follow a similar format to answer the last question. Make sure the response is end with The answer is _ . "
                }
            ],
            "round": [
                {
                    "role": "HUMAN",
                    "prompt": "{context}\n\n{question}\n"
                }
            ]
        }
    },
    "retriever": {"type": ZeroRetriever},
    "inferencer": {"type": GenInferencer}
}

LEval_gsm100_eval_cfg = {
    "evaluator": {"type": AccEvaluator},
    "pred_postprocessor": {"type": gsm100_postprocess},
    "dataset_postprocessor": {"type": gsm100_dataset_postprocess}
}

LEval_gsm100_datasets = [
    {
        "type": LEvalGSM100Dataset,
        "abbr": "LEval_gsm100",
        "path": "ais_bench/datasets/LEval/LEval/Exam/gsm100.jsonl",   ## The datasets path
        "name": "gsm100",
        "reader_cfg": LEval_gsm100_reader_cfg,
        "infer_cfg": LEval_gsm100_infer_cfg,
        "eval_cfg": LEval_gsm100_eval_cfg
    }
]