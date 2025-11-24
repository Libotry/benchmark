from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.openicl.icl_evaluator import AccEvaluator
from ais_bench.benchmark.datasets.leval import LEvalTPODataset
from ais_bench.benchmark.utils.postprocess.text_postprocessors  import first_capital_postprocess

LEval_tpo_reader_cfg = {
    'input_columns': ['context', 'question'],
    'output_column': 'answer',
    'train_split': 'test',
    'test_split': 'test'
}

LEval_tpo_infer_cfg = {
    'prompt_template': {
        'type': PromptTemplate,
        'template': {
            'begin': [
                {'role': 'SYSTEM', 'fallback_role': 'HUMAN', 'prompt': 'Now you are given a very long document. Please follow the instruction based on this document. For multi-choice questions, there could be a single correct option or multiple correct options. Please only provide the letter corresponding to the answer (like A or AB) when answering.'},
            ],
            'round': [
                {'role': 'HUMAN', 'prompt': 'Document is as follows.\n{context}\nQuestion:{question}\nAnswer:'},
                {'role': 'BOT', 'prompt': ''},
            ],
        }
    },
    'retriever': {'type': ZeroRetriever},
    'inferencer': {'type': GenInferencer}
}

LEval_tpo_eval_cfg = {
    'evaluator': {'type': AccEvaluator},
    'pred_postprocessor': {'type': first_capital_postprocess},
    'pred_role': 'BOT'
}

LEval_tpo_datasets = [
    {
        'type': LEvalTPODataset,
        'abbr': 'LEval_tpo',
        'path': 'ais_bench/datasets/LEval/LEval/Exam/tpo.jsonl',
        'name': 'tpo',
        'reader_cfg': LEval_tpo_reader_cfg,
        'infer_cfg': LEval_tpo_infer_cfg,
        'eval_cfg': LEval_tpo_eval_cfg
    }
]
