"""
LLM-as-judge evaluator using Claude Haiku for RDIC evaluation.

This module uses Claude Haiku to evaluate whether generated text follows
specific instructions. It provides semantic understanding beyond rule-based
pattern matching.
"""

import anthropic
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from src.config import get_config


@dataclass
class LLMJudgeResult:
    """Result from LLM judge evaluation"""
    score: float  # 0-1 score
    reasoning: str  # LLM's explanation
    confidence: Optional[str] = None  # high/medium/low


class LLMJudge:
    """LLM-based instruction following judge using Claude Haiku"""

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        """
        Initialize LLM judge.

        Args:
            model: Model to use (default: claude-haiku-4-5-20251001)
        """
        self.model = model
        config = get_config()
        self.client = anthropic.Anthropic(api_key=config.claude_api_key)
        self.system_prompt = """You are an expert evaluator assessing how well a text follows specific instructions.

Your task is to evaluate whether the given text adheres to the instruction provided. Consider:
- Tone (formal, casual, professional, friendly, serious, humorous, etc.)
- Detail level (brief, detailed, comprehensive, minimal, etc.)
- Style (technical, layperson, academic, conversational, etc.)
- Content (examples, citations, opinions, facts, etc.)
- Format (bullets, paragraphs, sections, structure, etc.)

Provide:
1. A score from 0.0 to 1.0 where:
   - 1.0 = Perfect adherence to instruction
   - 0.7-0.9 = Good adherence with minor issues
   - 0.5-0.6 = Partial adherence
   - 0.3-0.4 = Poor adherence
   - 0.0-0.2 = Does not follow instruction

2. Brief reasoning (1-2 sentences) explaining your score

3. Confidence level: high, medium, or low

Respond in this exact format:
SCORE: [number between 0.0 and 1.0]
CONFIDENCE: [high/medium/low]
REASONING: [your explanation]"""

    def evaluate(self, instruction: str, text: str, timeout: int = 10) -> LLMJudgeResult:
        """
        Evaluate if text follows instruction using LLM.

        Args:
            instruction: The instruction to check against
            text: The generated text to evaluate
            timeout: Timeout in seconds (default: 10)

        Returns:
            LLMJudgeResult with score, reasoning, and confidence
        """
        prompt = f"""Instruction: {instruction}

Text to evaluate:
{text}

Evaluate how well the text follows the instruction."""

        try:
            start_time = time.time()

            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                system=self.system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            elapsed = time.time() - start_time

            # Parse response
            response_text = response.content[0].text
            return self._parse_response(response_text, elapsed)

        except Exception as e:
            # Return error result
            return LLMJudgeResult(
                score=0.5,
                reasoning=f"Error during evaluation: {str(e)}",
                confidence="low"
            )

    def _parse_response(self, response_text: str, elapsed: float) -> LLMJudgeResult:
        """Parse LLM response into structured result"""
        try:
            score = 0.5
            reasoning = "Could not parse response"
            confidence = "low"

            # Parse SCORE
            if "SCORE:" in response_text:
                score_line = [line for line in response_text.split('\n') if 'SCORE:' in line][0]
                score_str = score_line.split('SCORE:')[1].strip()
                # Extract just the number
                import re
                score_match = re.search(r'(\d+\.?\d*)', score_str)
                if score_match:
                    score = float(score_match.group(1))
                    score = max(0.0, min(1.0, score))  # Clamp to 0-1

            # Parse CONFIDENCE
            if "CONFIDENCE:" in response_text:
                confidence_line = [line for line in response_text.split('\n') if 'CONFIDENCE:' in line][0]
                confidence = confidence_line.split('CONFIDENCE:')[1].strip().lower()
                if confidence not in ['high', 'medium', 'low']:
                    confidence = 'medium'

            # Parse REASONING
            if "REASONING:" in response_text:
                reasoning_lines = response_text.split('REASONING:')[1].strip()
                reasoning = reasoning_lines.split('\n')[0].strip()

            # Add timing info to reasoning
            reasoning = f"{reasoning} (evaluated in {elapsed:.2f}s)"

            return LLMJudgeResult(
                score=score,
                reasoning=reasoning,
                confidence=confidence
            )

        except Exception as e:
            return LLMJudgeResult(
                score=0.5,
                reasoning=f"Error parsing response: {str(e)}",
                confidence="low"
            )

    def evaluate_batch(self, examples: List[Dict], max_concurrent: int = 5) -> List[Dict]:
        """
        Evaluate a batch of examples.

        Args:
            examples: List of dicts with 'instruction' and 'text' keys
            max_concurrent: Maximum concurrent API calls (rate limiting)

        Returns:
            List of dicts with results
        """
        results = []
        for i, example in enumerate(examples):
            # Simple rate limiting
            if i > 0 and i % max_concurrent == 0:
                time.sleep(1)  # Brief pause every N requests

            result = self.evaluate(example['instruction'], example['text'])
            results.append({
                'instruction': example['instruction'],
                'text': example['text'],
                'score': result.score,
                'reasoning': result.reasoning,
                'confidence': result.confidence
            })

        return results

    def evaluate_conflict_example(self, conflict_example: Dict) -> Dict:
        """
        Evaluate a full conflict example with multiple turns.

        Args:
            conflict_example: Conflict example from dataset

        Returns:
            Dict with evaluation results for each instruction
        """
        results = []

        for turn in conflict_example['turns']:
            if 'instruction' in turn:
                # We would need generated text for this instruction
                # For now, return placeholder
                results.append({
                    'turn_id': turn['turn_id'],
                    'instruction': turn['instruction'],
                    'note': 'Requires generated text to evaluate'
                })

        return {
            'id': conflict_example['id'],
            'conflict_type': conflict_example['conflict_type'],
            'turn_results': results
        }


class CachedLLMJudge(LLMJudge):
    """LLM Judge with simple caching to avoid redundant API calls"""

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        super().__init__(model)
        self._cache = {}

    def evaluate(self, instruction: str, text: str, timeout: int = 10) -> LLMJudgeResult:
        """Evaluate with caching"""
        cache_key = f"{instruction}||{text}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        result = super().evaluate(instruction, text, timeout)
        self._cache[cache_key] = result
        return result

    def clear_cache(self):
        """Clear evaluation cache"""
        self._cache.clear()

    def cache_size(self) -> int:
        """Get current cache size"""
        return len(self._cache)


if __name__ == "__main__":
    # Test the LLM judge
    judge = LLMJudge()

    # Test formal tone
    formal_text = "Pursuant to our agreement, I hereby submit this report. The findings demonstrate conclusive evidence."
    result = judge.evaluate("Use strictly formal tone", formal_text)
    print(f"Formal test - Score: {result.score:.2f}")
    print(f"Confidence: {result.confidence}")
    print(f"Reasoning: {result.reasoning}")
    print()

    # Test casual tone
    casual_text = "Hey! So I've been thinking about this and yeah, it's pretty cool. You're gonna love it!"
    result = judge.evaluate("Use casual friendly tone", casual_text)
    print(f"Casual test - Score: {result.score:.2f}")
    print(f"Confidence: {result.confidence}")
    print(f"Reasoning: {result.reasoning}")
    print()

    # Test brief
    brief_text = "The answer is yes."
    result = judge.evaluate("Be very brief", brief_text)
    print(f"Brief test - Score: {result.score:.2f}")
    print(f"Confidence: {result.confidence}")
    print(f"Reasoning: {result.reasoning}")
