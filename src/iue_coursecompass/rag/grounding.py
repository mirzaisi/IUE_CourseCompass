"""
Grounding Module - Citation verification and hallucination detection.
=====================================================================

Provides tools to verify that generated answers are properly grounded
in source documents:
- Citation extraction and validation
- Claim extraction and verification
- Hallucination detection heuristics
- Grounding score calculation
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from iue_coursecompass.shared.logging import get_logger
from iue_coursecompass.shared.schemas import RetrievalHit

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Citation:
    """A citation reference in the answer."""

    chunk_id: str
    position: int  # Character position in answer
    valid: bool = False  # Whether it references a real source
    source: Optional[RetrievalHit] = None


@dataclass
class Claim:
    """A factual claim extracted from the answer."""

    text: str
    course_codes: list[str] = field(default_factory=list)
    numbers: list[str] = field(default_factory=list)  # ECTS, credits, etc.
    citation_ids: list[str] = field(default_factory=list)
    verified: bool = False
    verification_source: Optional[str] = None


@dataclass
class GroundingResult:
    """Result of grounding verification."""

    is_grounded: bool
    grounding_score: float  # 0.0 to 1.0
    total_citations: int
    valid_citations: int
    invalid_citations: list[str]
    claims_checked: int
    claims_verified: int
    unverified_claims: list[str]
    warnings: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Patterns
# ─────────────────────────────────────────────────────────────────────────────


# Course code pattern: XX(X) NNN or XXXX NNN
COURSE_CODE_PATTERN = re.compile(r"\b([A-Z]{2,4})\s*(\d{3})\b")

# Citation pattern: [chunk_id]
CITATION_PATTERN = re.compile(r"\[([^\]]+)\]")

# Number pattern for ECTS, credits, hours, etc.
NUMBER_PATTERN = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:ECTS|credits?|hours?|semesters?)\b", re.IGNORECASE)

# Hedging phrases that indicate uncertainty
HEDGING_PHRASES = [
    "i think",
    "i believe",
    "probably",
    "might be",
    "could be",
    "possibly",
    "it seems",
    "appears to",
    "likely",
    "perhaps",
    "maybe",
]

# Abstention phrases (good - shows the model knows its limits)
ABSTENTION_PHRASES = [
    "not found in",
    "could not find",
    "no information about",
    "don't have information",
    "not in the sources",
    "not mentioned in",
    "unable to find",
    "no data available",
    "sources do not contain",
]


# ─────────────────────────────────────────────────────────────────────────────
# Grounding Checker
# ─────────────────────────────────────────────────────────────────────────────


class GroundingChecker:
    """
    Verifies that generated answers are grounded in sources.

    Performs:
    - Citation validation
    - Claim extraction and verification
    - Course code existence checking
    - Hallucination heuristics

    Example:
        >>> checker = GroundingChecker()
        >>> result = checker.check(answer_text, source_hits)
        >>> print(f"Grounded: {result.is_grounded}, Score: {result.grounding_score}")
    """

    def __init__(
        self,
        min_grounding_score: float = 0.5,
        require_citations: bool = True,
    ):
        """
        Initialize the grounding checker.

        Args:
            min_grounding_score: Minimum score to consider grounded
            require_citations: Whether citations are required
        """
        self.min_grounding_score = min_grounding_score
        self.require_citations = require_citations

    def check(
        self,
        answer: str,
        sources: list[RetrievalHit],
    ) -> GroundingResult:
        """
        Check if an answer is properly grounded.

        Args:
            answer: Generated answer text
            sources: Source chunks used for generation

        Returns:
            GroundingResult with detailed verification info
        """
        warnings = []

        # Build source lookup
        source_ids = {s.chunk_id for s in sources}
        source_texts = {s.chunk_id: s.text.lower() for s in sources}
        all_source_text = " ".join(source_texts.values())

        # Extract and validate citations
        citations = self._extract_citations(answer, source_ids, sources)
        valid_citations = [c for c in citations if c.valid]
        invalid_citation_ids = [c.chunk_id for c in citations if not c.valid]

        if invalid_citation_ids:
            warnings.append(f"Invalid citations: {invalid_citation_ids}")

        # Extract and verify claims
        claims = self._extract_claims(answer)
        verified_claims = []
        unverified_claims = []

        for claim in claims:
            if self._verify_claim(claim, sources, all_source_text):
                claim.verified = True
                verified_claims.append(claim)
            else:
                unverified_claims.append(claim.text[:100])

        # Check for hedging (potential hallucination indicator)
        hedging_found = self._check_hedging(answer)
        if hedging_found:
            warnings.append(f"Hedging language detected: {hedging_found}")

        # Check for abstention (good behavior)
        is_abstention = self._check_abstention(answer)

        # Calculate grounding score
        grounding_score = self._calculate_score(
            citations=citations,
            valid_citations=valid_citations,
            claims=claims,
            verified_claims=verified_claims,
            is_abstention=is_abstention,
        )

        # Determine if grounded
        is_grounded = grounding_score >= self.min_grounding_score

        # If no citations but citations required, not grounded
        if self.require_citations and len(citations) == 0 and not is_abstention:
            is_grounded = False
            warnings.append("No citations found in answer")

        return GroundingResult(
            is_grounded=is_grounded,
            grounding_score=grounding_score,
            total_citations=len(citations),
            valid_citations=len(valid_citations),
            invalid_citations=invalid_citation_ids,
            claims_checked=len(claims),
            claims_verified=len(verified_claims),
            unverified_claims=unverified_claims,
            warnings=warnings,
        )

    def _extract_citations(
        self,
        answer: str,
        valid_ids: set[str],
        sources: list[RetrievalHit],
    ) -> list[Citation]:
        """Extract citation references from answer."""
        citations = []
        source_lookup = {s.chunk_id: s for s in sources}

        for match in CITATION_PATTERN.finditer(answer):
            chunk_id = match.group(1).strip()
            is_valid = chunk_id in valid_ids

            citations.append(
                Citation(
                    chunk_id=chunk_id,
                    position=match.start(),
                    valid=is_valid,
                    source=source_lookup.get(chunk_id),
                )
            )

        return citations

    def _extract_claims(self, answer: str) -> list[Claim]:
        """
        Extract factual claims from answer.

        Focuses on sentences containing:
        - Course codes
        - Numbers (ECTS, credits, etc.)
        - Citations
        """
        claims = []

        # Split into sentences
        sentences = re.split(r"[.!?]\s+", answer)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Find course codes
            course_codes = COURSE_CODE_PATTERN.findall(sentence)
            course_codes = [f"{c[0]} {c[1]}" for c in course_codes]

            # Find numbers
            numbers = NUMBER_PATTERN.findall(sentence)

            # Find citations
            citation_ids = CITATION_PATTERN.findall(sentence)

            # If it has specific claims, add it
            if course_codes or numbers:
                claims.append(
                    Claim(
                        text=sentence,
                        course_codes=course_codes,
                        numbers=numbers,
                        citation_ids=citation_ids,
                    )
                )

        return claims

    def _verify_claim(
        self,
        claim: Claim,
        sources: list[RetrievalHit],
        all_source_text: str,
    ) -> bool:
        """
        Verify a claim against sources.

        Returns True if the claim appears to be supported.
        """
        # If claim has citations, check if cited sources support it
        if claim.citation_ids:
            source_lookup = {s.chunk_id: s for s in sources}
            for cid in claim.citation_ids:
                source = source_lookup.get(cid)
                if source:
                    source_text = source.text.lower()

                    # Check course codes exist in source
                    for code in claim.course_codes:
                        code_parts = code.split()
                        if len(code_parts) == 2:
                            if code_parts[0].lower() in source_text and code_parts[1] in source_text:
                                claim.verification_source = cid
                                return True

                    # Check numbers exist in source
                    for num in claim.numbers:
                        if num in source.text:
                            claim.verification_source = cid
                            return True

        # Fallback: check if course codes appear anywhere in sources
        for code in claim.course_codes:
            code_lower = code.lower()
            if code_lower in all_source_text:
                return True

        # If no specific codes but has citation, consider weakly verified
        if claim.citation_ids and not claim.course_codes and not claim.numbers:
            return True

        return False

    def _check_hedging(self, answer: str) -> list[str]:
        """Check for hedging language that may indicate hallucination."""
        answer_lower = answer.lower()
        found = []

        for phrase in HEDGING_PHRASES:
            if phrase in answer_lower:
                found.append(phrase)

        return found

    def _check_abstention(self, answer: str) -> bool:
        """Check if the answer is an abstention (admitting lack of info)."""
        answer_lower = answer.lower()

        for phrase in ABSTENTION_PHRASES:
            if phrase in answer_lower:
                return True

        return False

    def _calculate_score(
        self,
        citations: list[Citation],
        valid_citations: list[Citation],
        claims: list[Claim],
        verified_claims: list[Claim],
        is_abstention: bool,
    ) -> float:
        """
        Calculate overall grounding score.

        Score components:
        - Claim verification ratio (60%) - PRIMARY: are course codes found in sources?
        - Citation validity ratio (20%) - SECONDARY: are citations properly formatted?
        - Base bonus (20%) - for having verified claims or valid citations
        
        The key insight: if the answer mentions course codes that exist in the 
        retrieved sources, the answer is grounded regardless of citation format.
        """
        # Abstention is always considered grounded
        if is_abstention:
            return 1.0

        score = 0.0

        # Claim verification score (60%) - PRIMARY metric
        # If course codes in the answer exist in sources, answer is grounded
        if claims:
            claim_ratio = len(verified_claims) / len(claims)
            score += 0.6 * claim_ratio
        else:
            # No specific claims - give benefit of doubt
            score += 0.3

        # Citation score (20%) - SECONDARY metric
        # Less important since LLM may use different citation formats
        if citations:
            citation_ratio = len(valid_citations) / len(citations)
            score += 0.2 * citation_ratio
        else:
            # No formal citations - small penalty only
            score += 0.1

        # Base score for having verified content (20%)
        if verified_claims or valid_citations:
            score += 0.2

        return min(1.0, score)


# ─────────────────────────────────────────────────────────────────────────────
# Course Code Verification
# ─────────────────────────────────────────────────────────────────────────────


def extract_course_codes(text: str) -> list[str]:
    """
    Extract course codes from text.

    Args:
        text: Text to search

    Returns:
        List of course codes (e.g., ["SE 301", "CE 100"])
    """
    matches = COURSE_CODE_PATTERN.findall(text)
    return [f"{m[0]} {m[1]}" for m in matches]


def verify_course_exists(
    course_code: str,
    sources: list[RetrievalHit],
) -> bool:
    """
    Verify a course code exists in sources.

    Args:
        course_code: Course code to verify
        sources: Source chunks

    Returns:
        True if course exists in sources
    """
    code_parts = course_code.upper().split()
    if len(code_parts) != 2:
        return False

    prefix, number = code_parts

    for source in sources:
        text = source.text.upper()
        if prefix in text and number in text:
            return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────────────────────────────────────


_checker: Optional[GroundingChecker] = None


def get_grounding_checker() -> GroundingChecker:
    """Get or create global grounding checker."""
    global _checker
    if _checker is None:
        _checker = GroundingChecker()
    return _checker


def check_grounding(
    answer: str,
    sources: list[RetrievalHit],
) -> GroundingResult:
    """
    Check if an answer is grounded in sources.

    Convenience function.

    Args:
        answer: Generated answer text
        sources: Source chunks

    Returns:
        GroundingResult
    """
    checker = get_grounding_checker()
    return checker.check(answer, sources)


def is_grounded(
    answer: str,
    sources: list[RetrievalHit],
    min_score: float = 0.5,
) -> bool:
    """
    Quick check if answer is grounded.

    Args:
        answer: Generated answer text
        sources: Source chunks
        min_score: Minimum grounding score

    Returns:
        True if grounded
    """
    result = check_grounding(answer, sources)
    return result.grounding_score >= min_score


def get_grounding_feedback(result: GroundingResult) -> str:
    """
    Generate human-readable feedback about grounding.

    Args:
        result: GroundingResult from check_grounding

    Returns:
        Feedback string
    """
    lines = []

    status = "✓ Grounded" if result.is_grounded else "✗ Not Grounded"
    lines.append(f"{status} (Score: {result.grounding_score:.2f})")

    lines.append(f"Citations: {result.valid_citations}/{result.total_citations} valid")

    if result.invalid_citations:
        lines.append(f"  Invalid: {result.invalid_citations}")

    lines.append(f"Claims: {result.claims_verified}/{result.claims_checked} verified")

    if result.unverified_claims:
        lines.append("  Unverified claims:")
        for claim in result.unverified_claims[:3]:  # Show first 3
            lines.append(f"    - {claim}...")

    if result.warnings:
        lines.append("Warnings:")
        for warning in result.warnings:
            lines.append(f"  ⚠ {warning}")

    return "\n".join(lines)
