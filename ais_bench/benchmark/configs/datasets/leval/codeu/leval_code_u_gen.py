from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.openicl.icl_evaluator import CodeUEvaluator
from ais_bench.benchmark.datasets.leval import LEvalCodeUDataset
from ais_bench.benchmark.utils.postprocess.text_postprocessors import general_postprocess

LEval_code_u_reader_cfg = {
    'input_columns': ['context', 'question'],
    'output_column': 'answer',
    'train_split': 'test',
    'test_split': 'test'
}

LEval_code_u_infer_cfg = {
    'prompt_template': {
        'type': PromptTemplate,
        'template': {
            'begin': [
                {'role': 'SYSTEM', 'fallback_role': 'HUMAN', 'prompt': 'Now you are given a code base consisting of a large amount of functions and the corresponding comments. In the end, I will call some functions defined in the code base. Please carefully read these codes and comments and answer the question.'},
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

LEval_code_u_eval_cfg = {
    'evaluator': {'type': CodeUEvaluator},
    'pred_postprocessor': {'type': general_postprocess},
    'pred_role': 'BOT'
}

LEval_code_u_datasets = [
    {
        'type': LEvalCodeUDataset,
        'abbr': 'LEval_code_u',
        'path': 'ais_bench/datasets/LEval/LEval/Exam/codeU.jsonl',
        'name': 'code_u',
        'reader_cfg': LEval_code_u_reader_cfg,
        'infer_cfg': LEval_code_u_infer_cfg,
        'eval_cfg': LEval_code_u_eval_cfg
    }
]
