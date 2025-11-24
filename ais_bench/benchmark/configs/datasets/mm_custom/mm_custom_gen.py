from ais_bench.benchmark.openicl.icl_prompt_template.icl_prompt_template_mm import MMPromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import MMCustomDataset, MMCustomEvaluator


mm_custom_reader_cfg = dict(
    input_columns=['question', 'mm_url'],
    output_column='answer'
)


mm_custom_infer_cfg = dict(
    prompt_template=dict(
        type=MMPromptTemplate,
        template=dict(
            round=[
                dict(role="HUMAN", prompt_mm={"text": "{question}",
                                           "mm_url": "{mm_url}"})  # mm_url will be automatically converted to image_url、 video_url or audio_url

            ]
            )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer)
)

mm_custom_eval_cfg = dict(
    evaluator=dict(type=MMCustomEvaluator)
)

mm_custom_datasets = [
    dict(
        abbr='mm_custom',
        type=MMCustomDataset,
        path='aisbench/datasets/mm_custom.jsonl',                           # Data path
        mm_type="path",                                                     # Input mm data type: "path" or "base64"
        num_frames=5,                                                       # Applies to video data only; number of frames to extract, default 5
        reader_cfg=mm_custom_reader_cfg,
        infer_cfg=mm_custom_infer_cfg,
        eval_cfg=mm_custom_eval_cfg
    )
]