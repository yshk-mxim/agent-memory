"""
Rule-based instruction checker for RDIC evaluation.

This module provides a rule-based approach to evaluating whether generated text
follows specific instructions across different dimensions (tone, detail, style, etc.)
"""

import re
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class RuleCheckResult:
    """Result from rule-based checking"""
    score: float  # 0-1 score
    reasoning: str  # Explanation of the score
    matched_patterns: List[str]  # Patterns that matched


class RuleChecker:
    """Rule-based instruction following checker"""

    def __init__(self):
        """Initialize rule checker with pattern definitions"""
        self._init_patterns()

    def _init_patterns(self):
        """Initialize detection patterns for different instruction types"""

        # TONE PATTERNS
        self.formal_indicators = [
            r'\b(hereby|aforementioned|pursuant|therefore|thus|furthermore|moreover|consequently)\b',
            r'\b(shall|must|ought|require|necessitate)\b',
            r'\b(regarding|concerning|pertaining to)\b',
            r'\b(Dear Sir|Dear Madam|To Whom It May Concern)\b',
            r'\b(Sincerely|Respectfully|Yours faithfully)\b',
        ]

        self.casual_indicators = [
            r'\b(hey|hi|hello|yeah|yep|nope|gonna|wanna|kinda|sorta)\b',
            r'\b(cool|awesome|great|nice|sweet)\b',
            r'[!]{2,}',  # Multiple exclamation marks
            r'\b(lol|haha|btw|fyi)\b',
            r'\b(you\'re|we\'re|it\'s|that\'s|don\'t|won\'t)\b',
        ]

        self.professional_indicators = [
            r'\b(professional|business|corporate|organizational)\b',
            r'\b(strategic|systematic|comprehensive|thorough)\b',
            r'\b(please find attached|as requested|per our discussion)\b',
            r'\b(best regards|kind regards|yours truly)\b',
        ]

        self.friendly_indicators = [
            r'\b(friend|buddy|pal|folks)\b',
            r'\b(happy to|glad to|pleased to|excited to)\b',
            r'\b(hope you|wish you|looking forward)\b',
            r'\b(thanks|thank you|appreciate)\b',
        ]

        self.serious_indicators = [
            r'\b(critical|important|essential|vital|crucial)\b',
            r'\b(urgent|immediate|pressing|significant)\b',
            r'\b(warning|caution|alert|notice)\b',
            r'\b(must|required|mandatory|obligatory)\b',
        ]

        self.humorous_indicators = [
            r'\b(joke|funny|laugh|humor|amusing|hilarious)\b',
            r'\b(pun|wit|irony|sarcasm)\b',
            r'[😂😄😊😉🤣]',  # Emoji
            r'\(.*?wink.*?\)',
            r'[!]{2,}.*[?]|[?].*[!]{2,}',  # Playful punctuation
        ]

        self.empathetic_indicators = [
            r'\b(understand|feel|empathize|sympathize)\b',
            r'\b(sorry|apologize|regret)\b',
            r'\b(difficult|challenging|hard|tough)\b',
            r'\b(support|help|assist|care)\b',
        ]

        self.objective_indicators = [
            r'\b(data|evidence|fact|study|research|analysis)\b',
            r'\b(according to|based on|shows that|indicates)\b',
            r'\b(measure|metric|statistic|percentage)\b',
            r'\b(objectively|impartially|neutrally)\b',
        ]

        # STYLE PATTERNS
        self.technical_indicators = [
            r'\b(algorithm|implementation|optimization|architecture)\b',
            r'\b(parameter|variable|function|method|class)\b',
            r'\b(API|SDK|framework|library|module)\b',
            r'\b(compile|execute|deploy|debug)\b',
            r'\b(bandwidth|latency|throughput|scalability)\b',
        ]

        self.layperson_indicators = [
            r'\b(basically|simply put|in other words|to put it simply)\b',
            r'\b(easy|simple|straightforward|basic)\b',
            r'\b(think of it as|imagine|like|similar to)\b',
            r'\b(everyday|common|normal|regular)\b',
        ]

        self.academic_indicators = [
            r'\b(research|study|analysis|investigation|examination)\b',
            r'\b(hypothesis|theory|methodology|findings)\b',
            r'\b(literature|scholars|researchers|authors)\b',
            r'\b(et al\.|ibid\.|cf\.|viz\.)\b',
            r'\b(significant|statistical|empirical)\b',
        ]

        self.conversational_indicators = [
            r'\b(well|you see|actually|you know|I mean|like)\b',
            r'\b(right|okay|so|now|then)\b',
            r'\?(?!\s*$)',  # Questions in middle of text
            r'\b(let\'s|we can|you can|you might)\b',
        ]

        self.jargon_indicators = [
            r'\b(leverage|synergy|paradigm|ecosystem|bandwidth)\b',
            r'\b(utilize|facilitate|implement|optimize)\b',
            r'\b(stakeholder|deliverable|actionable|scalable)\b',
            r'\b(ROI|KPI|B2B|SaaS|IoT)\b',
        ]

        # CONTENT PATTERNS
        self.example_indicators = [
            r'\b(for example|for instance|such as|e\.g\.)\b',
            r'\b(imagine|consider|suppose|let\'s say)\b',
            r'\b(example|illustration|case|scenario)\b',
            r':\s*["\']',  # Quote after colon
        ]

        self.citation_indicators = [
            r'\[[\d]+\]',  # [1], [2], etc.
            r'\([\w\s]+,?\s+\d{4}\)',  # (Author, 2024)
            r'\b(according to|as stated by|references|sources)\b',
            r'\b(cited|quoted|attributed to)\b',
        ]

        self.opinion_indicators = [
            r'\b(I think|I believe|in my opinion|personally)\b',
            r'\b(should|ought|recommend|suggest)\b',
            r'\b(feel|seems|appears|likely)\b',
            r'\b(arguably|presumably|possibly)\b',
        ]

        self.fact_indicators = [
            r'\b(is|are|was|were|has been|have been)\b(?!\s+should|\s+could)',
            r'\b(\d+%|\d+ percent)\b',
            r'\b(study|research|data|evidence)\b',
            r'\b(proven|demonstrated|established|confirmed)\b',
        ]

        # FORMAT PATTERNS
        self.bullet_indicators = [
            r'^\s*[-*•]\s+',  # Bullet points
            r'^\s*\d+\.\s+',  # Numbered lists
        ]

        self.paragraph_indicators = [
            r'\n\n',  # Paragraph breaks
            r'^[A-Z][^.!?]*[.!?]\s+[A-Z]',  # Sentence flow
        ]

        self.section_indicators = [
            r'^#{1,6}\s+',  # Markdown headers
            r'^[A-Z][^:]*:\s*$',  # Section headers with colons
            r'\b(Section|Chapter|Part)\s+\d+',
        ]

    def check_instruction(self, instruction: str, text: str) -> RuleCheckResult:
        """
        Check if text follows the given instruction using rule-based patterns.

        Args:
            instruction: The instruction to check against
            text: The generated text to evaluate

        Returns:
            RuleCheckResult with score, reasoning, and matched patterns
        """
        instruction_lower = instruction.lower()
        text_lower = text.lower()

        # Determine instruction type and check accordingly
        if any(word in instruction_lower for word in ['formal', 'professional', 'serious']):
            return self._check_formal_tone(instruction_lower, text, text_lower)
        elif any(word in instruction_lower for word in ['casual', 'friendly', 'informal']):
            return self._check_casual_tone(instruction_lower, text, text_lower)
        elif any(word in instruction_lower for word in ['humorous', 'humor', 'funny']):
            return self._check_humorous_tone(instruction_lower, text, text_lower)
        elif any(word in instruction_lower for word in ['empathetic', 'empathy', 'compassion']):
            return self._check_empathetic_tone(instruction_lower, text, text_lower)
        elif any(word in instruction_lower for word in ['objective', 'neutral', 'impartial']):
            return self._check_objective_tone(instruction_lower, text, text_lower)
        elif any(word in instruction_lower for word in ['technical', 'jargon']):
            return self._check_technical_style(instruction_lower, text, text_lower)
        elif any(word in instruction_lower for word in ['layperson', 'simple', 'plain language']):
            return self._check_layperson_style(instruction_lower, text, text_lower)
        elif any(word in instruction_lower for word in ['academic', 'scholarly']):
            return self._check_academic_style(instruction_lower, text, text_lower)
        elif any(word in instruction_lower for word in ['conversational', 'chatty']):
            return self._check_conversational_style(instruction_lower, text, text_lower)
        elif any(word in instruction_lower for word in ['brief', 'concise', 'short', 'minimal']):
            return self._check_brief_detail(instruction_lower, text, text_lower)
        elif any(word in instruction_lower for word in ['detailed', 'comprehensive', 'verbose', 'exhaustive']):
            return self._check_detailed(instruction_lower, text, text_lower)
        elif any(word in instruction_lower for word in ['example', 'examples']):
            return self._check_examples(instruction_lower, text, text_lower)
        elif any(word in instruction_lower for word in ['citation', 'citations', 'references']):
            return self._check_citations(instruction_lower, text, text_lower)
        elif any(word in instruction_lower for word in ['opinion', 'perspective', 'view']):
            return self._check_opinions(instruction_lower, text, text_lower)
        elif any(word in instruction_lower for word in ['fact', 'facts only', 'objective information']):
            return self._check_facts_only(instruction_lower, text, text_lower)
        elif any(word in instruction_lower for word in ['bullet', 'bullets', 'list']):
            return self._check_bullets(instruction_lower, text, text_lower)
        elif any(word in instruction_lower for word in ['paragraph', 'prose', 'continuous']):
            return self._check_paragraphs(instruction_lower, text, text_lower)
        elif any(word in instruction_lower for word in ['section', 'sections', 'structured']):
            return self._check_sections(instruction_lower, text, text_lower)
        else:
            # Generic check
            return RuleCheckResult(
                score=0.5,
                reasoning="Could not determine specific instruction type",
                matched_patterns=[]
            )

    def _count_pattern_matches(self, patterns: List[str], text: str) -> Tuple[int, List[str]]:
        """Count matches for a list of patterns"""
        total_matches = 0
        matched = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            if matches:
                total_matches += len(matches)
                matched.append(pattern)
        return total_matches, matched

    def _check_formal_tone(self, instruction_lower: str, text: str, text_lower: str) -> RuleCheckResult:
        """Check for formal/professional tone"""
        formal_matches, formal_patterns = self._count_pattern_matches(self.formal_indicators, text_lower)
        casual_matches, _ = self._count_pattern_matches(self.casual_indicators, text_lower)

        # Score based on formal/casual ratio
        total = formal_matches + casual_matches
        if total == 0:
            score = 0.5  # Neutral
        else:
            score = formal_matches / total

        # Penalty for contractions in very formal contexts
        if 'strictly' in instruction_lower or 'very formal' in instruction_lower:
            contraction_count = len(re.findall(r"\w+[''](?:re|ve|ll|d|s|t)\b", text))
            if contraction_count > 2:
                score *= 0.7

        reasoning = f"Formal indicators: {formal_matches}, Casual indicators: {casual_matches}"
        return RuleCheckResult(score=score, reasoning=reasoning, matched_patterns=formal_patterns)

    def _check_casual_tone(self, instruction_lower: str, text: str, text_lower: str) -> RuleCheckResult:
        """Check for casual/friendly tone"""
        casual_matches, casual_patterns = self._count_pattern_matches(self.casual_indicators, text_lower)
        formal_matches, _ = self._count_pattern_matches(self.formal_indicators, text_lower)

        total = casual_matches + formal_matches
        if total == 0:
            score = 0.5
        else:
            score = casual_matches / total

        # Bonus for contractions in casual contexts
        contraction_count = len(re.findall(r"\w+[''](?:re|ve|ll|d|s|t)\b", text))
        if contraction_count > 3:
            score = min(1.0, score * 1.2)

        reasoning = f"Casual indicators: {casual_matches}, Formal indicators: {formal_matches}, Contractions: {contraction_count}"
        return RuleCheckResult(score=score, reasoning=reasoning, matched_patterns=casual_patterns)

    def _check_humorous_tone(self, instruction_lower: str, text: str, text_lower: str) -> RuleCheckResult:
        """Check for humorous tone"""
        humor_matches, humor_patterns = self._count_pattern_matches(self.humorous_indicators, text_lower)
        serious_matches, _ = self._count_pattern_matches(self.serious_indicators, text_lower)

        # Check for punctuation patterns
        exclamation_count = text.count('!')
        if exclamation_count > 2:
            humor_matches += 1

        score = min(1.0, humor_matches / 3.0) if humor_matches > 0 else 0.3

        reasoning = f"Humor indicators: {humor_matches}, Serious indicators: {serious_matches}"
        return RuleCheckResult(score=score, reasoning=reasoning, matched_patterns=humor_patterns)

    def _check_empathetic_tone(self, instruction_lower: str, text: str, text_lower: str) -> RuleCheckResult:
        """Check for empathetic tone"""
        empathy_matches, empathy_patterns = self._count_pattern_matches(self.empathetic_indicators, text_lower)
        objective_matches, _ = self._count_pattern_matches(self.objective_indicators, text_lower)

        total = empathy_matches + objective_matches
        score = empathy_matches / total if total > 0 else 0.3

        reasoning = f"Empathetic indicators: {empathy_matches}, Objective indicators: {objective_matches}"
        return RuleCheckResult(score=score, reasoning=reasoning, matched_patterns=empathy_patterns)

    def _check_objective_tone(self, instruction_lower: str, text: str, text_lower: str) -> RuleCheckResult:
        """Check for objective tone"""
        objective_matches, objective_patterns = self._count_pattern_matches(self.objective_indicators, text_lower)
        empathy_matches, _ = self._count_pattern_matches(self.empathetic_indicators, text_lower)
        opinion_matches, _ = self._count_pattern_matches(self.opinion_indicators, text_lower)

        subjective_matches = empathy_matches + opinion_matches
        total = objective_matches + subjective_matches
        score = objective_matches / total if total > 0 else 0.5

        reasoning = f"Objective indicators: {objective_matches}, Subjective indicators: {subjective_matches}"
        return RuleCheckResult(score=score, reasoning=reasoning, matched_patterns=objective_patterns)

    def _check_technical_style(self, instruction_lower: str, text: str, text_lower: str) -> RuleCheckResult:
        """Check for technical style"""
        technical_matches, technical_patterns = self._count_pattern_matches(self.technical_indicators, text_lower)
        jargon_matches, jargon_patterns = self._count_pattern_matches(self.jargon_indicators, text_lower)
        layperson_matches, _ = self._count_pattern_matches(self.layperson_indicators, text_lower)

        technical_total = technical_matches + jargon_matches
        total = technical_total + layperson_matches
        score = technical_total / total if total > 0 else 0.3

        reasoning = f"Technical/Jargon indicators: {technical_total}, Layperson indicators: {layperson_matches}"
        return RuleCheckResult(score=score, reasoning=reasoning, matched_patterns=technical_patterns + jargon_patterns)

    def _check_layperson_style(self, instruction_lower: str, text: str, text_lower: str) -> RuleCheckResult:
        """Check for layperson-friendly style"""
        layperson_matches, layperson_patterns = self._count_pattern_matches(self.layperson_indicators, text_lower)
        technical_matches, _ = self._count_pattern_matches(self.technical_indicators, text_lower)
        jargon_matches, _ = self._count_pattern_matches(self.jargon_indicators, text_lower)

        technical_total = technical_matches + jargon_matches
        total = layperson_matches + technical_total
        score = layperson_matches / total if total > 0 else 0.5

        reasoning = f"Layperson indicators: {layperson_matches}, Technical/Jargon indicators: {technical_total}"
        return RuleCheckResult(score=score, reasoning=reasoning, matched_patterns=layperson_patterns)

    def _check_academic_style(self, instruction_lower: str, text: str, text_lower: str) -> RuleCheckResult:
        """Check for academic style"""
        academic_matches, academic_patterns = self._count_pattern_matches(self.academic_indicators, text_lower)
        conversational_matches, _ = self._count_pattern_matches(self.conversational_indicators, text_lower)

        total = academic_matches + conversational_matches
        score = academic_matches / total if total > 0 else 0.3

        reasoning = f"Academic indicators: {academic_matches}, Conversational indicators: {conversational_matches}"
        return RuleCheckResult(score=score, reasoning=reasoning, matched_patterns=academic_patterns)

    def _check_conversational_style(self, instruction_lower: str, text: str, text_lower: str) -> RuleCheckResult:
        """Check for conversational style"""
        conversational_matches, conversational_patterns = self._count_pattern_matches(self.conversational_indicators, text_lower)
        academic_matches, _ = self._count_pattern_matches(self.academic_indicators, text_lower)

        total = conversational_matches + academic_matches
        score = conversational_matches / total if total > 0 else 0.5

        # Bonus for questions
        question_count = text.count('?')
        if question_count > 1:
            score = min(1.0, score * 1.15)

        reasoning = f"Conversational indicators: {conversational_matches}, Academic indicators: {academic_matches}, Questions: {question_count}"
        return RuleCheckResult(score=score, reasoning=reasoning, matched_patterns=conversational_patterns)

    def _check_brief_detail(self, instruction_lower: str, text: str, text_lower: str) -> RuleCheckResult:
        """Check for brief/concise detail level"""
        word_count = len(text.split())

        # Thresholds for brief responses
        if 'minimal' in instruction_lower:
            threshold = 50
        elif 'brief' in instruction_lower or 'concise' in instruction_lower:
            threshold = 100
        else:
            threshold = 150

        if word_count <= threshold:
            score = 1.0
        elif word_count <= threshold * 1.5:
            score = 0.7
        elif word_count <= threshold * 2:
            score = 0.5
        else:
            score = 0.3

        reasoning = f"Word count: {word_count}, Threshold: {threshold}"
        return RuleCheckResult(score=score, reasoning=reasoning, matched_patterns=[])

    def _check_detailed(self, instruction_lower: str, text: str, text_lower: str) -> RuleCheckResult:
        """Check for detailed/comprehensive detail level"""
        word_count = len(text.split())

        # Thresholds for detailed responses
        if 'exhaustive' in instruction_lower or 'comprehensive' in instruction_lower:
            threshold = 300
        elif 'detailed' in instruction_lower or 'verbose' in instruction_lower:
            threshold = 200
        else:
            threshold = 150

        if word_count >= threshold:
            score = 1.0
        elif word_count >= threshold * 0.7:
            score = 0.7
        elif word_count >= threshold * 0.5:
            score = 0.5
        else:
            score = 0.3

        reasoning = f"Word count: {word_count}, Threshold: {threshold}"
        return RuleCheckResult(score=score, reasoning=reasoning, matched_patterns=[])

    def _check_examples(self, instruction_lower: str, text: str, text_lower: str) -> RuleCheckResult:
        """Check for presence of examples"""
        example_matches, example_patterns = self._count_pattern_matches(self.example_indicators, text_lower)

        # Look for "no examples" instruction
        if 'no example' in instruction_lower or 'without example' in instruction_lower:
            score = 1.0 - min(1.0, example_matches / 3.0)
            reasoning = f"Should avoid examples. Found: {example_matches}"
        else:
            score = min(1.0, example_matches / 2.0) if example_matches > 0 else 0.2
            reasoning = f"Example indicators: {example_matches}"

        return RuleCheckResult(score=score, reasoning=reasoning, matched_patterns=example_patterns)

    def _check_citations(self, instruction_lower: str, text: str, text_lower: str) -> RuleCheckResult:
        """Check for citations/references"""
        citation_matches, citation_patterns = self._count_pattern_matches(self.citation_indicators, text_lower)

        # Look for "no citations" instruction
        if 'no citation' in instruction_lower or 'without citation' in instruction_lower or 'no reference' in instruction_lower:
            score = 1.0 - min(1.0, citation_matches / 3.0)
            reasoning = f"Should avoid citations. Found: {citation_matches}"
        else:
            score = min(1.0, citation_matches / 2.0) if citation_matches > 0 else 0.2
            reasoning = f"Citation indicators: {citation_matches}"

        return RuleCheckResult(score=score, reasoning=reasoning, matched_patterns=citation_patterns)

    def _check_opinions(self, instruction_lower: str, text: str, text_lower: str) -> RuleCheckResult:
        """Check for opinions/perspectives"""
        opinion_matches, opinion_patterns = self._count_pattern_matches(self.opinion_indicators, text_lower)
        fact_matches, _ = self._count_pattern_matches(self.fact_indicators, text_lower)

        total = opinion_matches + fact_matches
        score = opinion_matches / total if total > 0 else 0.3

        reasoning = f"Opinion indicators: {opinion_matches}, Fact indicators: {fact_matches}"
        return RuleCheckResult(score=score, reasoning=reasoning, matched_patterns=opinion_patterns)

    def _check_facts_only(self, instruction_lower: str, text: str, text_lower: str) -> RuleCheckResult:
        """Check for facts-only content"""
        fact_matches, fact_patterns = self._count_pattern_matches(self.fact_indicators, text_lower)
        opinion_matches, _ = self._count_pattern_matches(self.opinion_indicators, text_lower)

        total = fact_matches + opinion_matches
        score = fact_matches / total if total > 0 else 0.5

        reasoning = f"Fact indicators: {fact_matches}, Opinion indicators: {opinion_matches}"
        return RuleCheckResult(score=score, reasoning=reasoning, matched_patterns=fact_patterns)

    def _check_bullets(self, instruction_lower: str, text: str, text_lower: str) -> RuleCheckResult:
        """Check for bullet points/lists"""
        bullet_matches, bullet_patterns = self._count_pattern_matches(self.bullet_indicators, text)

        # Count lines with bullets
        lines = text.split('\n')
        bullet_lines = sum(1 for line in lines if re.match(r'^\s*[-*•]\s+', line) or re.match(r'^\s*\d+\.\s+', line))

        if bullet_lines >= 3:
            score = 1.0
        elif bullet_lines >= 2:
            score = 0.7
        elif bullet_lines >= 1:
            score = 0.5
        else:
            score = 0.2

        reasoning = f"Bullet/list lines: {bullet_lines}"
        return RuleCheckResult(score=score, reasoning=reasoning, matched_patterns=bullet_patterns)

    def _check_paragraphs(self, instruction_lower: str, text: str, text_lower: str) -> RuleCheckResult:
        """Check for paragraph format"""
        paragraph_breaks = text.count('\n\n')
        bullet_matches, _ = self._count_pattern_matches(self.bullet_indicators, text)

        # Paragraphs should have breaks and no bullets
        if paragraph_breaks >= 2 and bullet_matches == 0:
            score = 1.0
        elif paragraph_breaks >= 1 and bullet_matches <= 1:
            score = 0.7
        elif bullet_matches > 3:
            score = 0.3
        else:
            score = 0.5

        reasoning = f"Paragraph breaks: {paragraph_breaks}, Bullet indicators: {bullet_matches}"
        return RuleCheckResult(score=score, reasoning=reasoning, matched_patterns=[])

    def _check_sections(self, instruction_lower: str, text: str, text_lower: str) -> RuleCheckResult:
        """Check for section structure"""
        section_matches, section_patterns = self._count_pattern_matches(self.section_indicators, text)

        # Count headers
        header_lines = sum(1 for line in text.split('\n') if re.match(r'^#{1,6}\s+', line) or re.match(r'^[A-Z][^:]*:\s*$', line))

        if header_lines >= 3:
            score = 1.0
        elif header_lines >= 2:
            score = 0.7
        elif header_lines >= 1:
            score = 0.5
        else:
            score = 0.2

        reasoning = f"Section headers: {header_lines}"
        return RuleCheckResult(score=score, reasoning=reasoning, matched_patterns=section_patterns)

    def evaluate_batch(self, examples: List[Dict]) -> List[Dict]:
        """
        Evaluate a batch of examples.

        Args:
            examples: List of dicts with 'instruction' and 'text' keys

        Returns:
            List of dicts with results
        """
        results = []
        for example in examples:
            result = self.check_instruction(example['instruction'], example['text'])
            results.append({
                'instruction': example['instruction'],
                'text': example['text'],
                'score': result.score,
                'reasoning': result.reasoning,
                'matched_patterns': result.matched_patterns
            })
        return results


if __name__ == "__main__":
    # Test the rule checker
    checker = RuleChecker()

    # Test formal tone
    formal_text = "Pursuant to our agreement, I hereby submit this report. The findings demonstrate conclusive evidence."
    result = checker.check_instruction("Use strictly formal tone", formal_text)
    print(f"Formal test - Score: {result.score:.2f}, Reasoning: {result.reasoning}")

    # Test casual tone
    casual_text = "Hey! So I've been thinking about this and yeah, it's pretty cool. You're gonna love it!"
    result = checker.check_instruction("Use casual friendly tone", casual_text)
    print(f"Casual test - Score: {result.score:.2f}, Reasoning: {result.reasoning}")

    # Test brief
    brief_text = "The answer is yes."
    result = checker.check_instruction("Be very brief", brief_text)
    print(f"Brief test - Score: {result.score:.2f}, Reasoning: {result.reasoning}")
