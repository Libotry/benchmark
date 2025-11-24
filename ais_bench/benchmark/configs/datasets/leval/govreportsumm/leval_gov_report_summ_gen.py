from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.openicl.icl_evaluator import RougeEvaluator
from ais_bench.benchmark.datasets.leval import LEvalGovReportSummDataset

LEval_govreport_summ_reader_cfg = {
    'input_columns': ['context', 'question', 'length'],
    'output_column': 'answer',
    'train_split': 'test',
    'test_split': 'test'
}

LEval_govreport_summ_infer_cfg = {
    'prompt_template': {
        'type': PromptTemplate,
        'template': {
            'begin': [
                {'role': 'SYSTEM', 'fallback_role': 'HUMAN', 'prompt': 'Now you are given a very long document. Please follow the instruction after this document. These instructions may include summarizing a document, answering questions based on the document, or writing a required paragraph.'},
            ],
            'round': [
                {'role': 'HUMAN', 'prompt': 'Document is as follows. {context}\nInstruction: {question}\nAnswer this question with {length} words.'},
                {'role': 'BOT', 'prompt': ''},
            ],
        }
    },
    'retriever': {'type': ZeroRetriever},
    'inferencer': {'type': GenInferencer}
}

LEval_govreport_summ_eval_cfg = {
    'evaluator': {'type': RougeEvaluator},
    'pred_role': 'BOT'
}

LEval_govreport_summ_datasets = [
    {
        'type': LEvalGovReportSummDataset,
        'abbr': 'LEval_gov_report_summ',
        'path': 'ais_bench/datasets/LEval/LEval/Generation/gov_report_summ.jsonl',
        'name': 'gov_report_summ',
        'reader_cfg': LEval_govreport_summ_reader_cfg,
        'infer_cfg': LEval_govreport_summ_infer_cfg,
        'eval_cfg': LEval_govreport_summ_eval_cfg
    }
]
