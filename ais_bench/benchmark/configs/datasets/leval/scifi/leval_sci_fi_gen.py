from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.openicl.icl_evaluator import SciFiEvaluator
from ais_bench.benchmark.datasets.leval import LEvalSciFiDataset
from ais_bench.benchmark.utils.postprocess.text_postprocessors import general_postprocess

LEval_sci_fi_reader_cfg = {
    'input_columns': ['context', 'question'],
    'output_column': 'answer',
    'train_split': 'test',
    'test_split': 'test'
}

LEval_sci_fi_infer_cfg = {
    'prompt_template': {
        'type': PromptTemplate,
        'template': {
            'begin': [
                {'role': 'SYSTEM', 'fallback_role': 'HUMAN', 'prompt': 'Now you are given a scientific fiction. I will ask you some questions and the answer should be \"True\" or \"False\". Notice that you should answer the question based on the evidence in the document instead of your background knowledge.'},
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

LEval_sci_fi_eval_cfg = {
    'evaluator': {'type': SciFiEvaluator},
    'pred_postprocessor': {'type': general_postprocess},
    'pred_role': 'BOT'
}

LEval_sci_fi_datasets = [
    {
        'type': LEvalSciFiDataset,
        'abbr': 'LEval_sci_fi',
        'path': 'ais_bench/datasets/LEval/LEval/Exam/sci_fi.jsonl',
        'name': 'sci_fi',
        'reader_cfg': LEval_sci_fi_reader_cfg,
        'infer_cfg': LEval_sci_fi_infer_cfg,
        'eval_cfg': LEval_sci_fi_eval_cfg
    }
]
