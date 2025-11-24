from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.openicl.icl_evaluator import LEvalEMEvaluator
from ais_bench.benchmark.datasets.leval import LEvalTopicRetrievalDataset
from ais_bench.benchmark.utils.postprocess.text_postprocessors import general_postprocess

LEval_tr_reader_cfg = {
    'input_columns': ['context', 'question'],
    'output_column': 'answer',
    'train_split': 'test',
    'test_split': 'test'
}

LEval_tr_infer_cfg = {
    'prompt_template': {
        'type': PromptTemplate,
        'template': {
            'begin': [
                {'role': 'SYSTEM', 'fallback_role': 'HUMAN', 'prompt': 'Below is a record of our previous conversation on many different topics. You are the ASSISTANT, and I am the USER. At the beginning of each topic, the USER will say \'I would like to discuss the topic of <TOPIC>\'. Memorize each <TOPIC>. At the end of the record, I will ask you to retrieve the first/second/third topic names. Now the record start.'},
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

LEval_tr_eval_cfg = {
    'evaluator': {'type': LEvalEMEvaluator},
    'pred_postprocessor': {'type': general_postprocess},
    'pred_role': 'BOT'
}

LEval_tr_datasets = [
    {
        'type': LEvalTopicRetrievalDataset,
        'abbr': 'LEval_topic_retrieval',
        'path': 'ais_bench/datasets/LEval/LEval/Exam/topic_retrieval_longchat.jsonl',
        'name': 'topic_retrieval_longchat',
        'reader_cfg': LEval_tr_reader_cfg,
        'infer_cfg': LEval_tr_infer_cfg,
        'eval_cfg': LEval_tr_eval_cfg
    }
]
