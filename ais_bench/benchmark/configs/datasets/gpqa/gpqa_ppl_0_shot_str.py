from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import PPLInferencer
from ais_bench.benchmark.datasets import GPQADataset
from ais_bench.benchmark.utils.postprocess.text_postprocessors import first_option_postprocess
from ais_bench.benchmark.datasets import GPQAEvaluator

gpqa_reader_cfg = dict(
    input_columns=['question', 'A', 'B', 'C', 'D'],
    output_column='answer')

gpqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            opt: f'Question: {{question}}\n(A){{A}}\n(B){{B}}\n(C){{C}}\n(D){{D}}\nAnswer: {opt}' for opt in ['A', 'B', 'C', 'D']
        }),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer))

gpqa_eval_cfg = dict(evaluator=dict(type=GPQAEvaluator),
                     pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'))

gpqa_datasets = []
gpqa_subsets = {
    'diamond': 'gpqa_diamond.csv'
}

for split in list(gpqa_subsets.keys()):
    gpqa_datasets.append(
        dict(
            abbr='GPQA_' + split,
            type=GPQADataset,
            path='ais_bench/datasets/gpqa/',
            name=gpqa_subsets[split],
            reader_cfg=gpqa_reader_cfg,
            infer_cfg=gpqa_infer_cfg,
            eval_cfg=gpqa_eval_cfg)
    )