# L-Eval Dataset Usage Guide

## Introduction

L-Eval is a comprehensive benchmark designed to evaluate the long-context understanding capabilities of large language models. It contains 20 different types of long-text tasks covering various domains including education, law, finance, and technology, providing a thorough assessment of model performance across different scenarios.

### Key Features

- **Long-Context Processing**: All datasets contain extensive contextual documents
- **Diverse Scenarios**: Covers exams, Q&A, summarization, code understanding, and more
- **Standardized Evaluation**: Uses Accuracy and ROUGE metrics for evaluation
- **Task Classification**: Supports both close-ended (multiple choice) and open-ended (Q&A) tasks

## Dataset Categories

### 1. Close-ended Tasks (Accuracy-based Evaluation)

Datasets evaluated using accuracy metrics:

| Dataset Name | Config File | Task Description | Evaluator |
|-------------|-------------|-----------------|-----------|
| **Coursera** | `coursera/leval_coursera_gen.py` | Online course exams with multiple choice questions | AccEvaluator |
| **GSM100** | `gsm100/leval_gsm100_gen.py` | Math word problems | AccEvaluator |
| **Quality** | `quality/leval_quality_gen.py` | Reading comprehension multiple choice | AccEvaluator |
| **TPO** | `tpo/leval_tpo_gen.py` | TOEFL practice tests | AccEvaluator |
| **Topic Retrieval** | `topicretrievallongchat/leval_topic_retrieval_gen.py` | Topic retrieval and matching | AccEvaluator |
| **Code U** | `codeu/leval_code_u_gen.py` | Code understanding and output inference | CodeUEvaluator (Custom) |
| **Sci-Fi** | `scifi/leval_sci_fi_gen.py` | Science fiction fact verification (True/False) | SciFiEvaluator (Custom) |

### 2. Open-ended Tasks (ROUGE-based Evaluation)

Datasets evaluated using ROUGE metrics:

| Dataset Name | Config File | Task Description | Evaluator |
|-------------|-------------|-----------------|-----------|
| **Financial QA** | `financialqa/leval_financial_qa_gen.py` | Financial domain Q&A | RougeEvaluator |
| **Gov Report Summ** | `govreportsumm/leval_gov_report_summ_gen.py` | Government report summarization | RougeEvaluator |
| **Legal Contract QA** | `legalcontractqa/leval_legal_contract_qa_gen.py` | Legal contract Q&A | RougeEvaluator |
| **Meeting Summ** | `meetingsumm/leval_meeting_summ_gen.py` | Meeting minutes summarization | RougeEvaluator |
| **MultiDoc QA** | `multidocqa/leval_multidoc_qa_gen.py` | Multi-document Q&A | RougeEvaluator |
| **Narrative QA** | `narrativeqa/leval_narrative_qa_gen.py` | Narrative text Q&A | RougeEvaluator |
| **Natural Question** | `naturalquestion/leval_natural_question_gen.py` | Natural question answering | RougeEvaluator |
| **News Summ** | `newssumm/leval_news_summ_gen.py` | News summarization | RougeEvaluator |
| **Paper Assistant** | `paperassistant/leval_paper_assistant_gen.py` | Academic paper Q&A | RougeEvaluator |
| **Patent Summ** | `patentsumm/leval_patent_summ_gen.py` | Patent document summarization | RougeEvaluator |
| **Review Summ** | `reviewsumm/leval_review_summ_gen.py` | Review summarization | RougeEvaluator |
| **Scientific QA** | `scientificqa/leval_scientific_qa_gen.py` | Scientific Q&A | RougeEvaluator |
| **TV Show Summ** | `tvshowsumm/leval_tv_show_summ_gen.py` | TV show plot summarization | RougeEvaluator |

## Configuration Guide

### Dataset Preparation
1. Download the dataset from [Hugging Face](https://huggingface.co/datasets/L4NLP/LEval)

2. Download the dataset locally using git lfs:
    ```bash
    cd ais_bench/datasets
    # Make sure git-lfs is installed (https://git-lfs.com)
    git lfs install
    git clone https://huggingface.co/datasets/L4NLP/LEval
    ```
3. Download results:
    - `LEval/LEval/Exam` directory contains close-ended dataset files
    - `LEval/LEval/Generation` directory contains open-ended dataset files

### Modify Dataset Configuration Files
1. All dataset configuration files are located in the `ais_bench/benchmark/configs/datasets/leval` directory
2. For configuration details, please refer to the main documentation

### Run Commands with Specified Dataset Configuration
1. Accuracy evaluation:
    ```bash
    ais_bench --models vllm_api_stream_chat --datasets leval_gsm100_gen 
    ```
2. Performance evaluation:
    ```bash
    ais_bench --models vllm_api_stream_chat --datasets leval_gsm100_gen --mode perf
    ```

### Configure `summarizer` for Average Metrics Across Multiple Datasets
1. Configuration file: `ais_bench/benchmark/configs/summarizers/leval.py`
2. Note: When calculating averages, ensure the datasets specified in the config file match those in the command execution
    - Example: `ais_bench --models vllm_api_stream_chat --datasets leval_gsm100_gen leval_coursera_gen --summarizer leval`
    - Configuration file:
        ```python
        # Comment out datasets not specified in the command, otherwise averages won't be calculated
        leval_close_end_subsets = [
            'LEval_coursera',
            'LEval_gsm100',
            # 'LEval_code_u',
            # 'LEval_sci_fi',
            # 'LEval_quality',
            # 'LEval_tpo',
            # 'LEval_topic_retrieval',
        ]
        ```

## Complete Dataset List and Commands

### Close-ended Tasks Dataset Commands

| Config File Name | Dataset Abbreviation | Run Command |
|-----------------|---------------------|-------------|
| `leval_coursera_gen` | `LEval_coursera` | `ais_bench --models <model> --datasets leval_coursera_gen` |
| `leval_gsm100_gen` | `LEval_gsm100` | `ais_bench --models <model> --datasets leval_gsm100_gen` |
| `leval_code_u_gen` | `LEval_code_u` | `ais_bench --models <model> --datasets leval_code_u_gen` |
| `leval_sci_fi_gen` | `LEval_sci_fi` | `ais_bench --models <model> --datasets leval_sci_fi_gen` |
| `leval_quality_gen` | `LEval_quality` | `ais_bench --models <model> --datasets leval_quality_gen` |
| `leval_tpo_gen` | `LEval_tpo` | `ais_bench --models <model> --datasets leval_tpo_gen` |
| `leval_topic_retrieval_gen` | `LEval_topic_retrieval` | `ais_bench --models <model> --datasets leval_topic_retrieval_gen` |

**Run all close-ended tasks**:
```bash
ais_bench --models <model> --datasets leval_coursera_gen leval_gsm100_gen leval_code_u_gen leval_sci_fi_gen leval_quality_gen leval_tpo_gen leval_topic_retrieval_gen --summarizer leval
```

### Open-ended Tasks Dataset Commands

| Config File Name | Dataset Abbreviation | Run Command |
|-----------------|---------------------|-------------|
| `leval_financial_qa_gen` | `LEval_financialqa` | `ais_bench --models <model> --datasets leval_financial_qa_gen` |
| `leval_gov_report_summ_gen` | `LEval_gov_report_summ` | `ais_bench --models <model> --datasets leval_gov_report_summ_gen` |
| `leval_legal_contract_qa_gen` | `LEval_legal_contract_qa` | `ais_bench --models <model> --datasets leval_legal_contract_qa_gen` |
| `leval_meeting_summ_gen` | `LEval_meeting_summ` | `ais_bench --models <model> --datasets leval_meeting_summ_gen` |
| `leval_multidoc_qa_gen` | `LEval_multidocqa` | `ais_bench --models <model> --datasets leval_multidoc_qa_gen` |
| `leval_narrative_qa_gen` | `LEval_narrativeqa` | `ais_bench --models <model> --datasets leval_narrative_qa_gen` |
| `leval_natural_question_gen` | `LEval_nq` | `ais_bench --models <model> --datasets leval_natural_question_gen` |
| `leval_news_summ_gen` | `LEval_news_summ` | `ais_bench --models <model> --datasets leval_news_summ_gen` |
| `leval_paper_assistant_gen` | `LEval_paper_assistant` | `ais_bench --models <model> --datasets leval_paper_assistant_gen` |
| `leval_patent_summ_gen` | `LEval_patent_summ` | `ais_bench --models <model> --datasets leval_patent_summ_gen` |
| `leval_review_summ_gen` | `LEval_review_summ` | `ais_bench --models <model> --datasets leval_review_summ_gen` |
| `leval_scientific_qa_gen` | `LEval_scientificqa` | `ais_bench --models <model> --datasets leval_scientific_qa_gen` |
| `leval_tv_show_summ_gen` | `LEval_tvshow_summ` | `ais_bench --models <model> --datasets leval_tv_show_summ_gen` |

**Run all open-ended tasks**:
```bash
ais_bench --models <model> --datasets leval_financial_qa_gen leval_gov_report_summ_gen leval_legal_contract_qa_gen leval_meeting_summ_gen leval_multidoc_qa_gen leval_narrative_qa_gen leval_natural_question_gen leval_news_summ_gen leval_paper_assistant_gen leval_patent_summ_gen leval_review_summ_gen leval_scientific_qa_gen leval_tv_show_summ_gen --summarizer leval
```

**Run all L-Eval datasets**:
```bash
ais_bench --models <model> --datasets leval_coursera_gen leval_gsm100_gen leval_code_u_gen leval_sci_fi_gen leval_quality_gen leval_tpo_gen leval_topic_retrieval_gen leval_financial_qa_gen leval_gov_report_summ_gen leval_legal_contract_qa_gen leval_meeting_summ_gen leval_multidoc_qa_gen leval_narrative_qa_gen leval_natural_question_gen leval_news_summ_gen leval_paper_assistant_gen leval_patent_summ_gen leval_review_summ_gen leval_scientific_qa_gen leval_tv_show_summ_gen --summarizer leval
```

## Summarizer Configuration Details

### Built-in Summarizer Configuration

Complete `leval.py` summarizer configuration:

```python
# Defines the summarizer groups for the L-Eval benchmark.
# Tasks are categorized into open-ended (evaluated with ROUGE) and
# close-ended (evaluated with accuracy) to allow for separate and combined scoring.

# Close-ended tasks are typically evaluated using accuracy metrics.
leval_close_end_subsets = [
    'LEval_coursera',
    'LEval_gsm100',
    'LEval_code_u',
    'LEval_sci_fi',
    'LEval_quality',
    'LEval_tpo',
    'LEval_topic_retrieval',
]

# Open-ended tasks are typically evaluated using ROUGE metrics for summarization.
leval_open_end_subsets = [
    'LEval_gov_report_summ',
    'LEval_meeting_summ',
    'LEval_news_summ',
    'LEval_patent_summ',
    'LEval_review_summ',
    'LEval_tvshow_summ',
    'LEval_financialqa',
    'LEval_legal_contract_qa',
    'LEval_multidocqa',
    'LEval_narrativeqa',
    'LEval_nq',
    'LEval_paper_assistant',
    'LEval_scientificqa',
]

leval_summary_groups = [
    {
        'name': 'leval_accuracy',
        'subsets': leval_close_end_subsets,
    },
    {
        'name': 'leval_rouge',
        'subsets': leval_open_end_subsets,
    },
]

summarizer = {
    'attr': 'accuracy',
    'summary_groups': leval_summary_groups
}
```

### Summarizer Output Description

When using `--summarizer leval`, the evaluation results will include:

1. **Detailed metrics for each dataset**: Individual evaluation results for each dataset
2. **leval_accuracy**: Average accuracy across all close-ended task datasets
3. **leval_rouge**: Average ROUGE scores (rouge-1, rouge-2, rouge-l) across all open-ended task datasets

### Customize Summarizer Subsets

If you want to evaluate only a subset of datasets and calculate averages, modify the `leval.py` configuration file by commenting out unnecessary datasets.

**Example**: Evaluate only some close-ended tasks

```python
# Keep only the datasets you want to evaluate
leval_close_end_subsets = [
    'LEval_coursera',
    'LEval_gsm100',
    # 'LEval_code_u',        # Comment out datasets not being evaluated
    # 'LEval_sci_fi',
    # 'LEval_quality',
    # 'LEval_tpo',
    # 'LEval_topic_retrieval',
]
```

Then run:
```bash
ais_bench --models <model> --datasets leval_coursera_gen leval_gsm100_gen --summarizer leval
```

## Frequently Asked Questions

### Q1: What's the difference between config file name and dataset abbreviation?
- **Config file name**: Used in the command line `--datasets` parameter, e.g., `leval_coursera_gen`
- **Dataset abbreviation**: Used in the summarizer configuration's `subsets` list, e.g., `LEval_coursera`

### Q2: How to ensure the summarizer calculates averages correctly?
- Ensure the datasets specified in `--datasets` match the `subsets` list in the summarizer configuration
- If evaluating only a subset of datasets, comment out the non-evaluated datasets in the summarizer config

### Q3: Where are the evaluation results saved?
- By default, saved in the `outputs/` directory
- Can be customized via the `work_dir` parameter in the config file

### Q4: How to modify the dataset path?
Modify the `path` parameter in the corresponding dataset configuration file:
```python
LEval_xxx_datasets = [
    {
        'type': LEvalXxxDataset,
        'abbr': 'LEval_xxx',
        'path': 'your/custom/path/xxx.jsonl',  # Modify this
        # ...
    }
]
```
