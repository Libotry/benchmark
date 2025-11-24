from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import PPLInferencer
from ais_bench.benchmark.openicl.icl_evaluator import AccEvaluator
from ais_bench.benchmark.datasets import ARCDataset

ARC_c_reader_cfg = dict(
    input_columns=['question', 'textA', 'textB', 'textC', 'textD'],
    output_column='answerKey',
    train_split='Dev',
    test_split='Test',
)

ARC_c_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            'A': 'Question: {question}\nAnswer: {textA}',
            'B': 'Question: {question}\nAnswer: {textB}',
            'C': 'Question: {question}\nAnswer: {textC}',
            'D': 'Question: {question}\nAnswer: {textD}'
        }),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer))

ARC_c_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

ARC_c_datasets = [
    dict(
        type=ARCDataset,
        abbr='ARC-c',
        path='ais_bench/datasets/ARC/ARC-c',
        name='ARC-Challenge',
        reader_cfg=ARC_c_reader_cfg,
        infer_cfg=ARC_c_infer_cfg,   
        eval_cfg=ARC_c_eval_cfg)
]
