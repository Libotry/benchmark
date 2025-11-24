# L-Eval 数据集使用指南

## 数据集简介

L-Eval 是一个专门用于评估大型语言模型长文本理解能力的基准测试数据集。该数据集包含多种类型的长文本任务，涵盖了教育、法律、金融、科技等多个领域，能够全面评估模型在不同场景下处理长文本的能力。

### 主要特点

- **长文本处理**：所有数据集均包含超长上下文文档
- **多样化场景**：涵盖考试、问答、摘要、代码理解等多种任务类型
- **标准化评估**：使用准确率（Accuracy）和 ROUGE 指标进行评估
- **开闭题分类**：支持闭卷选择题和开放式问答两类任务

## 数据集分类

### 1. 闭卷选择题类（Close-ended Tasks）

使用准确率（Accuracy）进行评估的数据集：

| 数据集名称 | 配置文件 | 任务描述 | 评估器 |
|----------|---------|---------|--------|
| **Coursera** | `coursera/leval_coursera_gen.py` | 在线课程考试题目，包含多选题 | AccEvaluator |
| **GSM100** | `gsm100/leval_gsm100_gen.py` | 数学应用题 | AccEvaluator |
| **Quality** | `quality/leval_quality_gen.py` | 阅读理解选择题 | AccEvaluator |
| **TPO** | `tpo/leval_tpo_gen.py` | 托福考试真题 | AccEvaluator |
| **Topic Retrieval** | `topicretrievallongchat/leval_topic_retrieval_gen.py` | 主题检索和匹配 | AccEvaluator |
| **Code U** | `codeu/leval_code_u_gen.py` | 代码理解和输出推断 | CodeUEvaluator（自定义） |
| **Sci-Fi** | `scifi/leval_sci_fi_gen.py` | 科幻小说事实判断（True/False） | SciFiEvaluator（自定义） |

### 2. 开放问答类（Open-ended Tasks）

使用 ROUGE 指标进行评估的数据集：

| 数据集名称 | 配置文件 | 任务描述 | 评估器 |
|----------|---------|---------|--------|
| **Financial QA** | `financialqa/leval_financial_qa_gen.py` | 金融领域问答 | RougeEvaluator |
| **Gov Report Summ** | `govreportsumm/leval_gov_report_summ_gen.py` | 政府报告摘要 | RougeEvaluator |
| **Legal Contract QA** | `legalcontractqa/leval_legal_contract_qa_gen.py` | 法律合同问答 | RougeEvaluator |
| **Meeting Summ** | `meetingsumm/leval_meeting_summ_gen.py` | 会议记录摘要 | RougeEvaluator |
| **MultiDoc QA** | `multidocqa/leval_multidoc_qa_gen.py` | 多文档问答 | RougeEvaluator |
| **Narrative QA** | `narrativeqa/leval_narrative_qa_gen.py` | 叙事文本问答 | RougeEvaluator |
| **Natural Question** | `naturalquestion/leval_natural_question_gen.py` | 自然问题问答 | RougeEvaluator |
| **News Summ** | `newssumm/leval_news_summ_gen.py` | 新闻摘要 | RougeEvaluator |
| **Paper Assistant** | `paperassistant/leval_paper_assistant_gen.py` | 学术论文问答 | RougeEvaluator |
| **Patent Summ** | `patentsumm/leval_patent_summ_gen.py` | 专利文档摘要 | RougeEvaluator |
| **Review Summ** | `reviewsumm/leval_review_summ_gen.py` | 评论摘要 | RougeEvaluator |
| **Scientific QA** | `scientificqa/leval_scientific_qa_gen.py` | 科学问答 | RougeEvaluator |
| **TV Show Summ** | `tvshowsumm/leval_tv_show_summ_gen.py` | 电视剧剧情摘要 | RougeEvaluator |

## 配置说明

### 准备数据集
1. 数据集下载地址 [Hugging Face](https://huggingface.co/datasets/L4NLP/LEval)

2. 通过 git lfs 下载数据集到本地
    ```bash
    cd ais_bench/datasets
    # Make sure git-lfs is installed (https://git-lfs.com)
    git lfs install
    git clone https://huggingface.co/datasets/L4NLP/LEval
    ```
3. 下载结果
    - `LEval/LEval/Exam` 目录下包含封闭式数据集文件
    - `LEval/LEval/Generation` 目录下包含开放式数据集文件

### 修改数据集配置文件
1. 数据集配置文件目录均在 `ais_bench/benchmark/configs/datasets/leval` 目录下
2. 如何配置可参考主页教程文档

### 运行命令指定数据集配置文件
1. 精度测评
    ```bash
    ais_bench --models vllm_api_stream_chat --datasets leval_gsm100_gen 
    ```
2. 性能测评
    ```bash
    ais_bench --models vllm_api_stream_chat --datasets leval_gsm100_gen --mode perf
    ```

### 对于精度测评需要输出多个同类数据集的平均指标可配置 `summarizer`
1. 配置文件 `ais_bench/benchmark/configs/summarizers/leval.py`
2. 注意计算平均值时需保证配置文件的指定数据集与命令执行指定数据集相匹配
    - 如 `ais_bench --models vllm_api_stream_chat --datasets leval_gsm100_gen leval_coursera_gen --summarizer leval`
    - 配置文件如下
        ```python
        # 需要注释掉没有指定的数据集，否则不会输出平均值
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

## 完整数据集列表和命令

### 闭卷选择题（Close-ended）数据集命令

| 数据集配置文件名 | 数据集简称 | 运行命令 |
|----------------|-----------|---------|
| `leval_coursera_gen` | `LEval_coursera` | `ais_bench --models <model> --datasets leval_coursera_gen` |
| `leval_gsm100_gen` | `LEval_gsm100` | `ais_bench --models <model> --datasets leval_gsm100_gen` |
| `leval_code_u_gen` | `LEval_code_u` | `ais_bench --models <model> --datasets leval_code_u_gen` |
| `leval_sci_fi_gen` | `LEval_sci_fi` | `ais_bench --models <model> --datasets leval_sci_fi_gen` |
| `leval_quality_gen` | `LEval_quality` | `ais_bench --models <model> --datasets leval_quality_gen` |
| `leval_tpo_gen` | `LEval_tpo` | `ais_bench --models <model> --datasets leval_tpo_gen` |
| `leval_topic_retrieval_gen` | `LEval_topic_retrieval` | `ais_bench --models <model> --datasets leval_topic_retrieval_gen` |

**批量运行所有闭卷题**：
```bash
ais_bench --models <model> --datasets leval_coursera_gen leval_gsm100_gen leval_code_u_gen leval_sci_fi_gen leval_quality_gen leval_tpo_gen leval_topic_retrieval_gen --summarizer leval
```

### 开放问答题（Open-ended）数据集命令

| 数据集配置文件名 | 数据集简称 | 运行命令 |
|----------------|-----------|---------|
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

**批量运行所有开放题**：
```bash
ais_bench --models <model> --datasets leval_financial_qa_gen leval_gov_report_summ_gen leval_legal_contract_qa_gen leval_meeting_summ_gen leval_multidoc_qa_gen leval_narrative_qa_gen leval_natural_question_gen leval_news_summ_gen leval_paper_assistant_gen leval_patent_summ_gen leval_review_summ_gen leval_scientific_qa_gen leval_tv_show_summ_gen --summarizer leval
```

**运行所有 L-Eval 数据集**：
```bash
ais_bench --models <model> --datasets leval_coursera_gen leval_gsm100_gen leval_code_u_gen leval_sci_fi_gen leval_quality_gen leval_tpo_gen leval_topic_retrieval_gen leval_financial_qa_gen leval_gov_report_summ_gen leval_legal_contract_qa_gen leval_meeting_summ_gen leval_multidoc_qa_gen leval_narrative_qa_gen leval_natural_question_gen leval_news_summ_gen leval_paper_assistant_gen leval_patent_summ_gen leval_review_summ_gen leval_scientific_qa_gen leval_tv_show_summ_gen --summarizer leval
```

## Summarizer 配置详解

### 内置 Summarizer 配置

完整的 `leval.py` summarizer 配置内容：

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

### Summarizer 输出说明

使用 `--summarizer leval` 后，评估结果会包含：

1. **各数据集的详细指标**：每个数据集的单独评估结果
2. **leval_accuracy**：所有闭卷题数据集的平均准确率
3. **leval_rouge**：所有开放题数据集的平均 ROUGE 分数（rouge-1, rouge-2, rouge-l）

### 自定义 Summarizer 子集

如果只想评估部分数据集并计算平均值，需要修改 `leval.py` 配置文件，注释掉不需要的数据集。

**示例**：只评估部分闭卷题

```python
# 只保留要评估的数据集
leval_close_end_subsets = [
    'LEval_coursera',
    'LEval_gsm100',
    # 'LEval_code_u',        # 注释掉不评估的数据集
    # 'LEval_sci_fi',
    # 'LEval_quality',
    # 'LEval_tpo',
    # 'LEval_topic_retrieval',
]
```

然后运行：
```bash
ais_bench --models <model> --datasets leval_coursera_gen leval_gsm100_gen --summarizer leval
```

## 常见问题

### Q1: 数据集配置文件名和数据集简称的区别？
- **配置文件名**：用于命令行 `--datasets` 参数，如 `leval_coursera_gen`
- **数据集简称**：用于 summarizer 配置中的 `subsets` 列表，如 `LEval_coursera`

### Q2: 如何确保 summarizer 正确计算平均值？
- 确保 `--datasets` 指定的数据集与 summarizer 配置文件中的 `subsets` 列表匹配
- 如果只评估部分数据集，需要注释掉 summarizer 配置中未评估的数据集

### Q3: 评估结果保存在哪里？
- 默认保存在 `outputs/` 目录下
- 可以通过配置文件的 `work_dir` 参数自定义输出路径

### Q4: 如何修改数据集路径？
修改对应数据集配置文件中的 `path` 参数：
```python
LEval_xxx_datasets = [
    {
        'type': LEvalXxxDataset,
        'abbr': 'LEval_xxx',
        'path': 'your/custom/path/xxx.jsonl',  # 修改这里
        # ...
    }
]
```