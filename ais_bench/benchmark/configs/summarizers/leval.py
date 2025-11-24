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