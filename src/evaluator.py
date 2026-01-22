"""
Main evaluation module combining rule-based and LLM-based evaluators.

This module provides a unified interface for evaluating instruction following,
combining multiple evaluation methods and providing aggregated results.
"""

import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import statistics

from src.rule_checker import RuleChecker, RuleCheckResult
from src.llm_judge import LLMJudge, CachedLLMJudge, LLMJudgeResult


@dataclass
class EvaluationResult:
    """Combined evaluation result"""
    instruction: str
    text: str
    rule_score: float
    llm_score: float
    combined_score: float
    agreement: float  # How much rule and LLM agree (0-1)
    rule_reasoning: str
    llm_reasoning: str
    llm_confidence: Optional[str] = None
    elapsed_time: float = 0.0


class Evaluator:
    """Main evaluator combining rule-based and LLM-based methods"""

    def __init__(
        self,
        use_rules: bool = True,
        use_llm: bool = True,
        rule_weight: float = 0.3,
        llm_weight: float = 0.7,
        use_cache: bool = True
    ):
        """
        Initialize evaluator.

        Args:
            use_rules: Whether to use rule-based checker
            use_llm: Whether to use LLM judge
            rule_weight: Weight for rule-based score (0-1)
            llm_weight: Weight for LLM score (0-1)
            use_cache: Whether to cache LLM results
        """
        self.use_rules = use_rules
        self.use_llm = use_llm
        self.rule_weight = rule_weight
        self.llm_weight = llm_weight

        # Initialize evaluators
        if self.use_rules:
            self.rule_checker = RuleChecker()

        if self.use_llm:
            if use_cache:
                self.llm_judge = CachedLLMJudge()
            else:
                self.llm_judge = LLMJudge()

        # Normalize weights
        total_weight = (rule_weight if use_rules else 0) + (llm_weight if use_llm else 0)
        if total_weight > 0:
            self.rule_weight = rule_weight / total_weight if use_rules else 0
            self.llm_weight = llm_weight / total_weight if use_llm else 0

    def evaluate(self, instruction: str, text: str) -> EvaluationResult:
        """
        Evaluate if text follows instruction using combined methods.

        Args:
            instruction: The instruction to check against
            text: The generated text to evaluate

        Returns:
            EvaluationResult with scores from all methods
        """
        start_time = time.time()

        rule_score = 0.5
        rule_reasoning = "Rule-based evaluation disabled"
        llm_score = 0.5
        llm_reasoning = "LLM evaluation disabled"
        llm_confidence = None

        # Run rule-based evaluation
        if self.use_rules:
            rule_result = self.rule_checker.check_instruction(instruction, text)
            rule_score = rule_result.score
            rule_reasoning = rule_result.reasoning

        # Run LLM evaluation
        if self.use_llm:
            llm_result = self.llm_judge.evaluate(instruction, text)
            llm_score = llm_result.score
            llm_reasoning = llm_result.reasoning
            llm_confidence = llm_result.confidence

        # Calculate combined score
        combined_score = (self.rule_weight * rule_score) + (self.llm_weight * llm_score)

        # Calculate agreement (how similar the scores are)
        agreement = 1.0 - abs(rule_score - llm_score)

        elapsed = time.time() - start_time

        return EvaluationResult(
            instruction=instruction,
            text=text,
            rule_score=rule_score,
            llm_score=llm_score,
            combined_score=combined_score,
            agreement=agreement,
            rule_reasoning=rule_reasoning,
            llm_reasoning=llm_reasoning,
            llm_confidence=llm_confidence,
            elapsed_time=elapsed
        )

    def evaluate_batch(self, examples: List[Dict]) -> List[EvaluationResult]:
        """
        Evaluate a batch of examples.

        Args:
            examples: List of dicts with 'instruction' and 'text' keys

        Returns:
            List of EvaluationResult objects
        """
        results = []
        for example in examples:
            result = self.evaluate(example['instruction'], example['text'])
            results.append(result)
        return results

    def get_agreement_stats(self, results: List[EvaluationResult]) -> Dict:
        """
        Calculate agreement statistics between rule-based and LLM methods.

        Args:
            results: List of evaluation results

        Returns:
            Dict with agreement metrics
        """
        if not results:
            return {
                'mean_agreement': 0.0,
                'median_agreement': 0.0,
                'min_agreement': 0.0,
                'max_agreement': 0.0,
                'high_agreement_count': 0,
                'high_agreement_percentage': 0.0
            }

        agreements = [r.agreement for r in results]
        high_agreement_threshold = 0.8
        high_agreement_count = sum(1 for a in agreements if a >= high_agreement_threshold)

        return {
            'mean_agreement': statistics.mean(agreements),
            'median_agreement': statistics.median(agreements),
            'min_agreement': min(agreements),
            'max_agreement': max(agreements),
            'high_agreement_count': high_agreement_count,
            'high_agreement_percentage': (high_agreement_count / len(agreements)) * 100,
            'total_evaluations': len(results)
        }

    def get_performance_stats(self, results: List[EvaluationResult]) -> Dict:
        """
        Calculate performance statistics.

        Args:
            results: List of evaluation results

        Returns:
            Dict with performance metrics
        """
        if not results:
            return {
                'mean_time': 0.0,
                'median_time': 0.0,
                'min_time': 0.0,
                'max_time': 0.0,
                'total_time': 0.0
            }

        times = [r.elapsed_time for r in results]

        return {
            'mean_time': statistics.mean(times),
            'median_time': statistics.median(times),
            'min_time': min(times),
            'max_time': max(times),
            'total_time': sum(times),
            'evaluations_under_2s': sum(1 for t in times if t < 2.0),
            'percentage_under_2s': (sum(1 for t in times if t < 2.0) / len(times)) * 100
        }

    def evaluate_conflict_example(
        self,
        conflict_example: Dict,
        generated_texts: Dict[int, str]
    ) -> Dict:
        """
        Evaluate a conflict example with generated responses.

        Args:
            conflict_example: Conflict example from dataset
            generated_texts: Dict mapping turn_id to generated text

        Returns:
            Dict with evaluation results for each instruction
        """
        results = []

        for turn in conflict_example['turns']:
            if 'instruction' in turn and turn['turn_id'] in generated_texts:
                result = self.evaluate(
                    turn['instruction'],
                    generated_texts[turn['turn_id']]
                )
                results.append({
                    'turn_id': turn['turn_id'],
                    'instruction': turn['instruction'],
                    'rule_score': result.rule_score,
                    'llm_score': result.llm_score,
                    'combined_score': result.combined_score,
                    'agreement': result.agreement,
                    'elapsed_time': result.elapsed_time
                })

        return {
            'id': conflict_example['id'],
            'conflict_type': conflict_example['conflict_type'],
            'turn_results': results,
            'overall_agreement': statistics.mean([r['agreement'] for r in results]) if results else 0.0
        }

    def get_score_distribution(self, results: List[EvaluationResult]) -> Dict:
        """
        Get distribution of scores.

        Args:
            results: List of evaluation results

        Returns:
            Dict with score distribution
        """
        if not results:
            return {}

        rule_scores = [r.rule_score for r in results]
        llm_scores = [r.llm_score for r in results]
        combined_scores = [r.combined_score for r in results]

        def score_bins(scores):
            """Categorize scores into bins"""
            return {
                'excellent (0.9-1.0)': sum(1 for s in scores if 0.9 <= s <= 1.0),
                'good (0.7-0.9)': sum(1 for s in scores if 0.7 <= s < 0.9),
                'fair (0.5-0.7)': sum(1 for s in scores if 0.5 <= s < 0.7),
                'poor (0.3-0.5)': sum(1 for s in scores if 0.3 <= s < 0.5),
                'very poor (0.0-0.3)': sum(1 for s in scores if 0.0 <= s < 0.3)
            }

        return {
            'rule_based': score_bins(rule_scores),
            'llm_based': score_bins(llm_scores),
            'combined': score_bins(combined_scores),
            'rule_mean': statistics.mean(rule_scores),
            'llm_mean': statistics.mean(llm_scores),
            'combined_mean': statistics.mean(combined_scores)
        }


class RuleOnlyEvaluator(Evaluator):
    """Evaluator using only rule-based checking"""

    def __init__(self):
        super().__init__(use_rules=True, use_llm=False, rule_weight=1.0)


class LLMOnlyEvaluator(Evaluator):
    """Evaluator using only LLM judge"""

    def __init__(self, use_cache: bool = True):
        super().__init__(use_rules=False, use_llm=True, llm_weight=1.0, use_cache=use_cache)


class HybridEvaluator(Evaluator):
    """Evaluator combining both methods with balanced weights"""

    def __init__(self, use_cache: bool = True):
        super().__init__(
            use_rules=True,
            use_llm=True,
            rule_weight=0.4,
            llm_weight=0.6,
            use_cache=use_cache
        )


if __name__ == "__main__":
    # Test the evaluator
    print("Testing Hybrid Evaluator...")
    evaluator = HybridEvaluator()

    # Test formal tone
    formal_text = "Pursuant to our agreement, I hereby submit this report. The findings demonstrate conclusive evidence."
    result = evaluator.evaluate("Use strictly formal tone", formal_text)
    print(f"\nFormal test:")
    print(f"  Rule score: {result.rule_score:.2f}")
    print(f"  LLM score: {result.llm_score:.2f}")
    print(f"  Combined: {result.combined_score:.2f}")
    print(f"  Agreement: {result.agreement:.2f}")
    print(f"  Time: {result.elapsed_time:.2f}s")

    # Test casual tone
    casual_text = "Hey! So I've been thinking about this and yeah, it's pretty cool. You're gonna love it!"
    result = evaluator.evaluate("Use casual friendly tone", casual_text)
    print(f"\nCasual test:")
    print(f"  Rule score: {result.rule_score:.2f}")
    print(f"  LLM score: {result.llm_score:.2f}")
    print(f"  Combined: {result.combined_score:.2f}")
    print(f"  Agreement: {result.agreement:.2f}")
    print(f"  Time: {result.elapsed_time:.2f}s")

    # Test batch
    examples = [
        {"instruction": "Be very brief", "text": "The answer is yes."},
        {"instruction": "Be very detailed", "text": "The answer is yes."}
    ]
    results = evaluator.evaluate_batch(examples)
    print(f"\nBatch test: {len(results)} examples evaluated")

    # Agreement stats
    agreement_stats = evaluator.get_agreement_stats(results)
    print(f"Mean agreement: {agreement_stats['mean_agreement']:.2f}")

    # Performance stats
    perf_stats = evaluator.get_performance_stats(results)
    print(f"Mean time: {perf_stats['mean_time']:.2f}s")
