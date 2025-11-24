"""Unit tests for L-Eval custom evaluators: CodeUEvaluator and SciFiEvaluator."""

from ais_bench.benchmark.openicl.icl_evaluator.icl_leval_evaluator import (
    CodeUEvaluator, SciFiEvaluator
)


class TestCodeUEvaluator:
    def test_code_u_accuracy_basic(self):
        evaluator = CodeUEvaluator()
        # Predictions are messy but include final answer fragments.
        predictions = [
            "the final output of the code is 42",  # should extract '42'
            "After execution, the final output is success",  # should extract 'success'
        ]
        references = [
            "42",  # normalized match
            "success",  # normalized match
        ]
        result = evaluator.score(predictions, references)
        assert 'accuracy' in result
        assert result['accuracy'] == 100.0

    def test_code_u_accuracy_partial(self):
        evaluator = CodeUEvaluator()
        predictions = [
            "the final output of the code is 100",
            "the final output of the code would be failure",
            "So the final output is \"!unknown\""
        ]
        references = [
            "100",
            "pass",   # mismatch
            "unknown"  # last one matches after normalization
        ]
        result = evaluator.score(predictions, references)
        # Should have 2 correct (100, unknown)
        assert result['accuracy'] == (2 / 3) * 100

    def test_code_u_length_mismatch(self):
        evaluator = CodeUEvaluator()
        predictions = ["output one"]
        references = ["ref one", "ref two"]
        result = evaluator.score(predictions, references)
        assert 'error' in result


class TestSciFiEvaluator:
    def test_scifi_loyalty_accuracy(self):
        evaluator = SciFiEvaluator()
        predictions = [
            "The crew remains loyal throughout the mission [fact: true]",
            "Mutiny emerges among the crew false [fact: true]",
            "TRUE heroism is shown",
            "FALSE reports are spread"
        ]
        references = [
            "True",     
            "False",    
            "true",     
            "false"
        ]
        result = evaluator.score(predictions, references)
        # pred loyalty extraction:
        # 1 -> <error>
        # 2 -> false
        # 3 -> true
        # 4 -> false
        # correct: 2,3,4 => 3/4
        assert result['accuracy'] == (3 / 4) * 100

    def test_scifi_length_mismatch(self):
        evaluator = SciFiEvaluator()
        predictions = ["True"]
        references = ["True", "False"]
        result = evaluator.score(predictions, references)
        assert 'error' in result
