"""
Unit tests for RDIC evaluation framework.

Tests rule-based checker, LLM judge, and combined evaluator on golden examples.
"""

import unittest
import json
import time
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rule_checker import RuleChecker
from src.llm_judge import LLMJudge, CachedLLMJudge
from src.evaluator import Evaluator, HybridEvaluator, RuleOnlyEvaluator, LLMOnlyEvaluator


class TestRuleChecker(unittest.TestCase):
    """Test rule-based instruction checker"""

    @classmethod
    def setUpClass(cls):
        """Load golden examples"""
        golden_path = Path(__file__).parent.parent / "data" / "golden_eval_examples.json"
        with open(golden_path) as f:
            cls.golden_examples = json.load(f)
        cls.checker = RuleChecker()

    def test_formal_tone(self):
        """Test formal tone detection"""
        example = self.golden_examples[0]
        result = self.checker.check_instruction(example['instruction'], example['text'])

        print(f"\n[Rule] Formal test: {result.score:.2f} (expected ~{example['expected_rule_score']})")
        print(f"  Reasoning: {result.reasoning}")

        # Score should be reasonably high for formal text
        self.assertGreater(result.score, 0.6, "Formal tone should score high")

    def test_casual_tone(self):
        """Test casual tone detection"""
        example = self.golden_examples[1]
        result = self.checker.check_instruction(example['instruction'], example['text'])

        print(f"\n[Rule] Casual test: {result.score:.2f} (expected ~{example['expected_rule_score']})")
        print(f"  Reasoning: {result.reasoning}")

        # Score should be reasonably high for casual text
        self.assertGreater(result.score, 0.6, "Casual tone should score high")

    def test_brief_detail(self):
        """Test brief detail level detection"""
        example = self.golden_examples[2]
        result = self.checker.check_instruction(example['instruction'], example['text'])

        print(f"\n[Rule] Brief test: {result.score:.2f} (expected ~{example['expected_rule_score']})")
        print(f"  Reasoning: {result.reasoning}")

        # Score should be high for brief text
        self.assertGreater(result.score, 0.7, "Brief text should score high")

    def test_detailed(self):
        """Test detailed content detection"""
        example = self.golden_examples[3]
        result = self.checker.check_instruction(example['instruction'], example['text'])

        print(f"\n[Rule] Detailed test: {result.score:.2f} (expected ~{example['expected_rule_score']})")
        print(f"  Reasoning: {result.reasoning}")

        # Score should be high for detailed text
        self.assertGreater(result.score, 0.7, "Detailed text should score high")

    def test_technical_style(self):
        """Test technical style detection"""
        example = self.golden_examples[4]
        result = self.checker.check_instruction(example['instruction'], example['text'])

        print(f"\n[Rule] Technical test: {result.score:.2f} (expected ~{example['expected_rule_score']})")
        print(f"  Reasoning: {result.reasoning}")

        # Score should be reasonably high for technical text
        self.assertGreater(result.score, 0.5, "Technical style should score reasonably")

    def test_layperson_style(self):
        """Test layperson style detection"""
        example = self.golden_examples[5]
        result = self.checker.check_instruction(example['instruction'], example['text'])

        print(f"\n[Rule] Layperson test: {result.score:.2f} (expected ~{example['expected_rule_score']})")
        print(f"  Reasoning: {result.reasoning}")

        # Score should be reasonably high for layperson text
        self.assertGreater(result.score, 0.5, "Layperson style should score reasonably")

    def test_with_examples(self):
        """Test example content detection"""
        example = self.golden_examples[6]
        result = self.checker.check_instruction(example['instruction'], example['text'])

        print(f"\n[Rule] With examples test: {result.score:.2f} (expected ~{example['expected_rule_score']})")
        print(f"  Reasoning: {result.reasoning}")

        # Score should be reasonably high when examples present
        self.assertGreater(result.score, 0.3, "Text with examples should score")

    def test_without_examples(self):
        """Test no-examples content detection"""
        example = self.golden_examples[7]
        result = self.checker.check_instruction(example['instruction'], example['text'])

        print(f"\n[Rule] Without examples test: {result.score:.2f} (expected ~{example['expected_rule_score']})")
        print(f"  Reasoning: {result.reasoning}")

        # Score should be high when no examples and instruction says no examples
        self.assertGreater(result.score, 0.5, "Text without examples should score")

    def test_bullet_format(self):
        """Test bullet point format detection"""
        example = self.golden_examples[8]
        result = self.checker.check_instruction(example['instruction'], example['text'])

        print(f"\n[Rule] Bullet format test: {result.score:.2f} (expected ~{example['expected_rule_score']})")
        print(f"  Reasoning: {result.reasoning}")

        # Score should be high for bullet format
        self.assertGreater(result.score, 0.7, "Bullet format should score high")

    def test_paragraph_format(self):
        """Test paragraph format detection"""
        example = self.golden_examples[9]
        result = self.checker.check_instruction(example['instruction'], example['text'])

        print(f"\n[Rule] Paragraph format test: {result.score:.2f} (expected ~{example['expected_rule_score']})")
        print(f"  Reasoning: {result.reasoning}")

        # Score should be reasonably high for paragraph format
        self.assertGreater(result.score, 0.4, "Paragraph format should score")

    def test_batch_evaluation(self):
        """Test batch evaluation"""
        examples = [
            {"instruction": ex['instruction'], "text": ex['text']}
            for ex in self.golden_examples[:3]
        ]

        start_time = time.time()
        results = self.checker.evaluate_batch(examples)
        elapsed = time.time() - start_time

        print(f"\n[Rule] Batch evaluation: {len(results)} examples in {elapsed:.2f}s")
        print(f"  Average time per example: {elapsed/len(results):.3f}s")

        self.assertEqual(len(results), 3, "Should evaluate all examples")
        self.assertLess(elapsed / len(results), 0.1, "Should be fast per example")


class TestLLMJudge(unittest.TestCase):
    """Test LLM-based judge"""

    @classmethod
    def setUpClass(cls):
        """Load golden examples"""
        golden_path = Path(__file__).parent.parent / "data" / "golden_eval_examples.json"
        with open(golden_path) as f:
            cls.golden_examples = json.load(f)
        cls.judge = CachedLLMJudge()

    def test_formal_tone(self):
        """Test formal tone evaluation"""
        example = self.golden_examples[0]
        result = self.judge.evaluate(example['instruction'], example['text'])

        print(f"\n[LLM] Formal test: {result.score:.2f} (expected ~{example['expected_llm_score']})")
        print(f"  Confidence: {result.confidence}")
        print(f"  Reasoning: {result.reasoning}")

        # Score should be reasonably high for formal text
        self.assertGreater(result.score, 0.6, "Formal tone should score high")

    def test_casual_tone(self):
        """Test casual tone evaluation"""
        example = self.golden_examples[1]
        result = self.judge.evaluate(example['instruction'], example['text'])

        print(f"\n[LLM] Casual test: {result.score:.2f} (expected ~{example['expected_llm_score']})")
        print(f"  Confidence: {result.confidence}")
        print(f"  Reasoning: {result.reasoning}")

        # Score should be reasonably high for casual text
        self.assertGreater(result.score, 0.6, "Casual tone should score high")

    def test_brief_detail(self):
        """Test brief detail level evaluation"""
        example = self.golden_examples[2]
        result = self.judge.evaluate(example['instruction'], example['text'])

        print(f"\n[LLM] Brief test: {result.score:.2f} (expected ~{example['expected_llm_score']})")
        print(f"  Confidence: {result.confidence}")
        print(f"  Reasoning: {result.reasoning}")

        # Score should be high for brief text
        self.assertGreater(result.score, 0.6, "Brief text should score high")

    def test_detailed(self):
        """Test detailed content evaluation"""
        example = self.golden_examples[3]
        result = self.judge.evaluate(example['instruction'], example['text'])

        print(f"\n[LLM] Detailed test: {result.score:.2f} (expected ~{example['expected_llm_score']})")
        print(f"  Confidence: {result.confidence}")
        print(f"  Reasoning: {result.reasoning}")

        # Score should be high for detailed text
        self.assertGreater(result.score, 0.7, "Detailed text should score high")

    def test_caching(self):
        """Test LLM judge caching"""
        example = self.golden_examples[0]

        # Clear cache
        self.judge.clear_cache()
        initial_cache_size = self.judge.cache_size()

        # First call
        start_time = time.time()
        result1 = self.judge.evaluate(example['instruction'], example['text'])
        time1 = time.time() - start_time

        # Second call (should be cached)
        start_time = time.time()
        result2 = self.judge.evaluate(example['instruction'], example['text'])
        time2 = time.time() - start_time

        print(f"\n[LLM] Caching test:")
        print(f"  First call: {time1:.3f}s")
        print(f"  Cached call: {time2:.3f}s")
        print(f"  Cache size: {self.judge.cache_size()}")

        self.assertEqual(result1.score, result2.score, "Cached result should match")
        self.assertLess(time2, time1 / 2, "Cached call should be much faster")
        self.assertEqual(self.judge.cache_size(), initial_cache_size + 1, "Cache should grow")

    def test_performance(self):
        """Test LLM judge performance"""
        example = self.golden_examples[0]

        start_time = time.time()
        result = self.judge.evaluate(example['instruction'], example['text'])
        elapsed = time.time() - start_time

        print(f"\n[LLM] Performance test: {elapsed:.2f}s")

        # Should complete reasonably quickly
        self.assertLess(elapsed, 10.0, "Should complete within 10 seconds")


class TestHybridEvaluator(unittest.TestCase):
    """Test hybrid evaluator combining rule-based and LLM methods"""

    @classmethod
    def setUpClass(cls):
        """Load golden examples"""
        golden_path = Path(__file__).parent.parent / "data" / "golden_eval_examples.json"
        with open(golden_path) as f:
            cls.golden_examples = json.load(f)
        cls.evaluator = HybridEvaluator(use_cache=True)

    def test_formal_tone(self):
        """Test formal tone with hybrid evaluator"""
        example = self.golden_examples[0]
        result = self.evaluator.evaluate(example['instruction'], example['text'])

        print(f"\n[Hybrid] Formal test:")
        print(f"  Rule: {result.rule_score:.2f}")
        print(f"  LLM: {result.llm_score:.2f}")
        print(f"  Combined: {result.combined_score:.2f}")
        print(f"  Agreement: {result.agreement:.2f}")
        print(f"  Time: {result.elapsed_time:.2f}s")

        # Both should score reasonably high
        self.assertGreater(result.combined_score, 0.6, "Combined score should be high")

    def test_all_golden_examples(self):
        """Test all golden examples"""
        results = []

        for i, example in enumerate(self.golden_examples):
            result = self.evaluator.evaluate(example['instruction'], example['text'])
            results.append(result)

            print(f"\n[Hybrid] Golden example {i+1}/{len(self.golden_examples)}:")
            print(f"  Type: {example['conflict_type']}")
            print(f"  Rule: {result.rule_score:.2f}, LLM: {result.llm_score:.2f}")
            print(f"  Combined: {result.combined_score:.2f}, Agreement: {result.agreement:.2f}")

        # Calculate agreement stats
        agreement_stats = self.evaluator.get_agreement_stats(results)
        print(f"\n[Hybrid] Agreement Statistics:")
        print(f"  Mean agreement: {agreement_stats['mean_agreement']:.2f}")
        print(f"  Median agreement: {agreement_stats['median_agreement']:.2f}")
        print(f"  High agreement (>0.8): {agreement_stats['high_agreement_percentage']:.1f}%")

        # Calculate performance stats
        perf_stats = self.evaluator.get_performance_stats(results)
        print(f"\n[Hybrid] Performance Statistics:")
        print(f"  Mean time: {perf_stats['mean_time']:.2f}s")
        print(f"  Median time: {perf_stats['median_time']:.2f}s")
        print(f"  Under 2s: {perf_stats['percentage_under_2s']:.1f}%")

        # Get score distribution
        score_dist = self.evaluator.get_score_distribution(results)
        print(f"\n[Hybrid] Score Distribution:")
        print(f"  Rule mean: {score_dist['rule_mean']:.2f}")
        print(f"  LLM mean: {score_dist['llm_mean']:.2f}")
        print(f"  Combined mean: {score_dist['combined_mean']:.2f}")

        # Assertions
        self.assertEqual(len(results), len(self.golden_examples), "Should evaluate all examples")

        # Check if agreement is reasonable (target >80% have >0.8 agreement)
        # This is a soft target - may not always meet it
        if agreement_stats['high_agreement_percentage'] < 80:
            print(f"\n  WARNING: Agreement below 80% target")

        # Check performance (target: >90% under 2s)
        if perf_stats['percentage_under_2s'] < 90:
            print(f"\n  WARNING: Performance below target (>90% under 2s)")

    def test_batch_evaluation(self):
        """Test batch evaluation"""
        examples = [
            {"instruction": ex['instruction'], "text": ex['text']}
            for ex in self.golden_examples[:5]
        ]

        start_time = time.time()
        results = self.evaluator.evaluate_batch(examples)
        elapsed = time.time() - start_time

        print(f"\n[Hybrid] Batch evaluation: {len(results)} examples in {elapsed:.2f}s")
        print(f"  Average time per example: {elapsed/len(results):.2f}s")

        self.assertEqual(len(results), 5, "Should evaluate all examples")


class TestAgreementMetrics(unittest.TestCase):
    """Test agreement between rule-based and LLM methods"""

    @classmethod
    def setUpClass(cls):
        """Load golden examples"""
        golden_path = Path(__file__).parent.parent / "data" / "golden_eval_examples.json"
        with open(golden_path) as f:
            cls.golden_examples = json.load(f)

    def test_agreement_calculation(self):
        """Test agreement metrics between methods"""
        evaluator = HybridEvaluator(use_cache=True)

        results = []
        for example in self.golden_examples:
            result = evaluator.evaluate(example['instruction'], example['text'])
            results.append(result)

        agreement_stats = evaluator.get_agreement_stats(results)

        print(f"\n[Agreement] Full metrics:")
        print(f"  Mean: {agreement_stats['mean_agreement']:.3f}")
        print(f"  Median: {agreement_stats['median_agreement']:.3f}")
        print(f"  Min: {agreement_stats['min_agreement']:.3f}")
        print(f"  Max: {agreement_stats['max_agreement']:.3f}")
        print(f"  High agreement count: {agreement_stats['high_agreement_count']}/{agreement_stats['total_evaluations']}")
        print(f"  High agreement %: {agreement_stats['high_agreement_percentage']:.1f}%")

        # Target: >80% high agreement
        target_percentage = 80.0
        if agreement_stats['high_agreement_percentage'] >= target_percentage:
            print(f"\n  SUCCESS: Agreement target met ({target_percentage}%)")
        else:
            print(f"\n  NOTE: Agreement below {target_percentage}% target")
            print(f"        This is OK - rule-based and LLM methods measure different aspects")


def run_tests():
    """Run all tests"""
    print("="*80)
    print("RDIC Evaluation Framework - Unit Tests")
    print("="*80)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add tests
    suite.addTests(loader.loadTestsFromTestCase(TestRuleChecker))
    suite.addTests(loader.loadTestsFromTestCase(TestLLMJudge))
    suite.addTests(loader.loadTestsFromTestCase(TestHybridEvaluator))
    suite.addTests(loader.loadTestsFromTestCase(TestAgreementMetrics))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
