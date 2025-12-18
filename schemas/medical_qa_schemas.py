"""
Medical QA Schema Constraint Evaluation Strategy

This module defines a progression of Pydantic schemas from most to least constraining,
designed to evaluate how schema constraints affect LLM performance on medical QA tasks.

Each schema level provides different insights:
- Level 1-2: Test strict structural compliance
- Level 3-4: Test semantic constraints and reasoning
- Level 5-6: Test flexibility and natural language handling
"""

from typing import List, Optional, Literal, Dict
from pydantic import BaseModel, Field, field_validator, constr


# ============================================================================
# LEVEL 1: Strict Answer Only
# ============================================================================
# Insight: Baseline test for pure structural compliance. Measures if model can output
#          valid JSON with exact enum values without additional fields.

class StrictAnswerOnly(BaseModel):
    """Level 1: Minimal constraint - only the answer, strict enum."""
    answer: Literal["yes", "no", "maybe"] = Field(
        description="The answer to the question"
    )
    
    class Config:
        extra = "forbid"  # No additional fields allowed


# ============================================================================
# LEVEL 2: Answer with confidence score 
# ============================================================================
# Insight: Tests impact of adding numeric type constraints. Measures whether requiring
#          a confidence score (0.0-1.0) affects model performance and calibration.

class StructuredAnswer(BaseModel):
    """Level 2: Answer with confidence score. """
    answer: Literal["yes", "no", "maybe"] = Field(
        description="The answer to the question"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0"
    )
    
    class Config:
        extra = "forbid"


# ============================================================================
# LEVEL 3: Answer with required brief reasoning
# ============================================================================
# Insight: Tests impact of requiring text reasoning (10-200 chars). Measures whether
#          forcing models to explain their answers improves accuracy or changes behavior.

class AnswerWithReasoning(BaseModel):
    """Level 3 (PubMedQA): Answer with required brief reasoning."""
    answer: Literal["yes", "no", "maybe"] = Field(
        description="The answer to the question"
    )
    reasoning: constr(min_length=10, max_length=200) = Field(
        description="Brief reasoning explaining why this answer is correct."
    )

    class Config:
        extra = "forbid"


# ============================================================================
# LEVEL_3_INVERTED - INVERTED CHAIN OF THOUGHT: Reasoning first, then answer
# ============================================================================
# Insight: Tests inverted chain-of-thought: requiring reasoning before the answer.
#          Measures if forcing models to think through problems first improves accuracy.

class InvertedCoTAnswer(BaseModel):
    """Inverted Chain of Thought for PubMedQA: Reasoning first, then answer."""
    reasoning: constr(min_length=10, max_length=200) = Field(
        description="Brief reasoning explaining the thought process before arriving at the answer."
    )
    answer: Literal["yes", "no", "maybe"] = Field(
        description="The answer to the question, determined after reasoning."
    )

    class Config:
        extra = "forbid"


# ============================================================================
# LEVEL 4: Answer with reasoning and confidence score
# ============================================================================
# Insight: Combines reasoning with confidence calibration. Tests whether requiring both
#          structured explanations and explicit confidence scores improves performance.

class AnswerWithReasoningAndConfidence(BaseModel):
    """Level 4 (PubMedQA): Answer with reasoning and confidence."""
    answer: Literal["yes", "no", "maybe"] = Field(
        description="The answer to the question"
    )
    reasoning: constr(min_length=10, max_length=200) = Field(
        description="Brief reasoning explaining why this answer is correct."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0"
    )

    class Config:
        extra = "forbid"




# ============================================================================
# LEVEL 5: Answer grounded in key evidence from the abstract
# ============================================================================
# Insight: Requires evidence extraction (1-5 key pieces) plus extended reasoning (10-500 chars).
#          Tests if grounding answers in explicit evidence improves accuracy and trustworthiness.

class GroundedAnswer(BaseModel):
    """Level 5 (PubMedQA): Answer grounded in key evidence from the abstract."""
    answer: Literal["yes", "no", "maybe"] = Field(
        description="The answer to the question"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0"
    )
    reasoning: constr(min_length=10, max_length=500) = Field(
        description="Reasoning that synthesizes how the evidence supports the answer."
    )
    key_evidence: List[str] = Field(
        min_length=1,
        max_length=5,
        description="List of 1-5 key pieces of evidence from the abstract (paraphrased or lightly quoted)."
    )

    class Config:
        extra = "forbid"


# ============================================================================
# LEVEL 6: Flexible Answer with optional extras
# ============================================================================
# Insight: Tests flexibility: all fields optional except answer, and extra fields allowed.
#          Measures whether reduced constraints improve or degrade model performance.

class FlexibleGroundedAnswer(BaseModel):
    """Level 6 (PubMedQA): Core fields plus optional extras allowed."""
    answer: Literal["yes", "no", "maybe"] = Field(
        description="The answer to the question"
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional confidence score between 0.0 and 1.0"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Optional reasoning for the answer."
    )
    key_evidence: Optional[List[str]] = Field(
        default=None,
        description="Optional list of key evidence from the abstract."
    )

    class Config:
        extra = "allow"




# ============================================================================
# MULTIPLE CHOICE SCHEMAS for MedQA and MedMCQA (one valid answer only)
# ============================================================================

# ============================================================================
# LEVEL 1: Strict Multiple Choice
# ============================================================================
# Insight: Baseline for multiple choice: strict enum (A/B/C/D) only. Tests pure structural
#          compliance and model's ability to produce valid JSON with exact enum values.
class StrictMultipleChoice(BaseModel):
    """Level 1 for MC: Strict enum for multiple choice answers."""
    answer: Literal["A", "B", "C", "D"] = Field(
        description="The multiple choice answer"
    )
    
    class Config:
        extra = "forbid"


# ============================================================================
# LEVEL 2: Answer with confidence score
# ============================================================================
# Insight: Adds confidence score (0.0-1.0) to strict answer. Tests whether requiring
#          probabilistic beliefs improves answer calibration and error diagnosis.

class MCQAnswerWithConfidence(BaseModel):
    """Level 2 for MC: Answer with confidence score."""
    answer: Literal["A", "B", "C", "D"]= Field(
        description="The multiple choice answer"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="confidence score"
    )
    
    class Config:
        extra = "forbid"



# ============================================================================
# LEVEL 3: Answer with justification
# ============================================================================
# Insight: Adds justification field (max 200 chars) to explain the answer. Tests whether
#          requiring explanations improves model reasoning quality and answer accuracy.
class MCQAnswerWithJustification(BaseModel):
    """Level 3 for MC: Answer with justification."""
    answer: Literal["A", "B", "C", "D"] = Field(
        description="The multiple choice answer"
    )
    justification: constr(max_length=200)
    
    class Config:
        extra = "forbid"

# ============================================================================
# LEVEL_3_INVERTED - INVERTED CHAIN OF THOUGHT: Reasoning first, then answer
# ============================================================================
# Insight: Inverted CoT for MCQ: reasoning before answer. Tests if requiring models to
#          think through problems first improves multiple choice accuracy.

class MCQInvertedCoTAnswer(BaseModel):
    """Inverted Chain of Thought for MCQ: Reasoning first, then answer."""
    reasoning: constr(min_length=20, max_length=200) = Field(
        description="Brief reasoning explaining the thought process before arriving at the answer."
    )
    answer: Literal["A", "B", "C", "D"] = Field(
        description="The multiple choice answer, determined after reasoning."
    )

    class Config:
        extra = "forbid"

# ============================================================================
# LEVEL 4: Answer with reasoning and confidence score
# ============================================================================
# Insight: Combines reasoning (min 20 chars) with confidence score. Tests whether requiring
#          both substantive explanations and explicit confidence calibration improves performance.
class MultipleChoiceWithReasoning(BaseModel):
    """Level 4 for MC: Answer with required reasoning and confidence score."""
    answer: Literal["A", "B", "C", "D"] = Field(
        description="The multiple choice answer"
    )
    reasoning: constr(min_length=20, max_length=200) = Field(
        description="Brief reasoning explaining why this answer is correct"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score"
    )
    
    class Config:
        extra = "forbid"



# ============================================================================
# LEVEL 5: Option Elimination
# ============================================================================
# Insight: Requires elimination of 3 incorrect options with structured reasoning for each.
#          Tests whether forcing models to explicitly rule out wrong answers improves accuracy.

class OptionElimination(BaseModel):
    option: Literal["A", "B", "C", "D"]
    reason: constr(min_length=5, max_length=300) = Field(
        description="Short clinical reasoning why this option is incorrect, tied to the question stem."
    )

class MCQWithFullElimination(BaseModel):
    """Level 5 for MC: Full structured response for medical diagnosis with option elimination."""
    answer: Literal["A", "B", "C", "D"] = Field(
        description="The single best answer among A, B, C, D."
    )
    eliminated: List[OptionElimination] = Field(
        description="Exactly three entries: three distinct options among A, B, C, D, each with a reason why it is incorrect."
    )
    key_evidence: constr(min_length=5, max_length=300) = Field(
        description="The main piece of clinical or factual evidence that supports the chosen answer."
    )

    class Config:
        extra = "forbid"

    @field_validator("eliminated")
    @classmethod
    def check_three_unique_eliminations(cls, eliminated: List[OptionElimination]):
        """
        Ensure:
        - exactly 3 eliminations
        - options are unique
        - options are in {A, B, C, D}
        (No coupling with the chosen 'answer'; correctness is handled by the eval pipeline.)
        """
        if len(eliminated) != 3:
            raise ValueError("You must provide exactly three eliminated options.")

        options = [e.option for e in eliminated]
        if len(set(options)) != 3:
            raise ValueError("Eliminated options must be three distinct choices among A, B, C, and D.")

        # MCQOption already restricts to A/B/C/D, so this is technically redundant,
        # but it makes the intent explicit and protects you if MCQOption changes later.
        allowed = {"A", "B", "C", "D"}
        if not set(options).issubset(allowed):
            raise ValueError("Eliminated options must be in {'A', 'B', 'C', 'D'}.")

        return eliminated

# ============================================================================
# LEVEL 6: Comprehensive Multiple Choice
# ============================================================================
# Insight: Maximum complexity: reasoning, confidence, key concepts, and differential diagnosis.
#          Tests whether comprehensive structured outputs improve medical diagnosis accuracy.

class ComprehensiveMultipleChoice(BaseModel):
    """Level 6 for MC: Full structured response for medical diagnosis."""
    answer: Literal["A", "B", "C", "D"] = Field(
        description="The multiple choice answer"
    )
    reasoning: constr(min_length=20, max_length=500) = Field(
        description="Reasoning explaining why this answer is correct."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score"
    )
    key_concepts: List[str] = Field(
        min_length=2,
        max_length=5,
        description="Key medical concepts relevant to this question"
    )
    differential_diagnosis: Optional[List[str]] = Field(
        default=None,
        max_length=3,
        description="Other possible answers considered (differential diagnosis)"
    )
    
    class Config:
        extra = "forbid"



# ============================================================================
# SCHEMA REGISTRY - Easy access for evaluation
# ============================================================================

PUBMEDQA_SCHEMAS = {
    "level1_strict": StrictAnswerOnly,
    "level2_structured": StructuredAnswer,
    "level3_reasoning": AnswerWithReasoning,
    "level4_reasoning_confidence": AnswerWithReasoningAndConfidence,
    "level5_grounded": GroundedAnswer,
    "level6_flexible": FlexibleGroundedAnswer,
    "inverted_cot": InvertedCoTAnswer,
}

MULTIPLE_CHOICE_SCHEMAS = {
    "level1_strict": StrictMultipleChoice,
    "level2_confidence": MCQAnswerWithConfidence,
    "level3_justification": MCQAnswerWithJustification,
    "level4_reasoning": MultipleChoiceWithReasoning,
    "level5_elimination": MCQWithFullElimination,
    "level6_comprehensive": ComprehensiveMultipleChoice,
    "inverted_cot": MCQInvertedCoTAnswer,
}

# Schema metadata for evaluation tracking
SCHEMA_METADATA = {
    # PubMedQA schemas
    "level1_strict": {
        "constraint_level": 1,
        "description": "Strict enum, no extras",
        "expected_insight": "Baseline structural compliance",
        "schema_class": "StrictAnswerOnly",
    },
    "level2_structured": {
        "constraint_level": 2,
        "description": "Enum + required numeric field (confidence)",
        "expected_insight": "Impact of type constraints",
        "schema_class": "StructuredAnswer",
    },
    "level3_reasoning": {
        "constraint_level": 3,
        "description": "Enum + required reasoning field",
        "expected_insight": "Impact of required text constraints",
        "schema_class": "AnswerWithReasoning",
    },
    "level4_reasoning_confidence": {
        "constraint_level": 4,
        "description": "Enum + reasoning + confidence",
        "expected_insight": "Multiple required fields with validation",
        "schema_class": "AnswerWithReasoningAndConfidence",
    },
    "level5_grounded": {
        "constraint_level": 5,
        "description": "Answer grounded in key evidence from abstract",
        "expected_insight": "Complex structured output with evidence",
        "schema_class": "GroundedAnswer",
    },
    "level6_flexible": {
        "constraint_level": 6,
        "description": "Core fields plus optional extras allowed",
        "expected_insight": "Flexibility benefit analysis",
        "schema_class": "FlexibleGroundedAnswer",
    },
    # Multiple Choice schemas
    "mc_level1_strict": {
        "constraint_level": 1,
        "description": "Strict enum for multiple choice (A/B/C/D)",
        "expected_insight": "Baseline structural compliance",
        "schema_class": "StrictMultipleChoice",
    },
    "mc_level2_confidence": {
        "constraint_level": 2,
        "description": "Answer with confidence score",
        "expected_insight": "Impact of numeric constraints",
        "schema_class": "MCQAnswerWithConfidence",
    },
    "mc_level2_confidence_new": {
        "constraint_level": 2,
        "description": "Answer with confidence score",
        "expected_insight": "Impact of numeric constraints",
        "schema_class": "MCQAnswerWithConfidenceNew",
    },
    "mc_level3_justification": {
        "constraint_level": 3,
        "description": "Answer with justification",
        "expected_insight": "Impact of required text field",
        "schema_class": "MCQAnswerWithJustification",
    },
    "mc_level4_reasoning": {
        "constraint_level": 4,
        "description": "Answer with reasoning and confidence",
        "expected_insight": "Multiple required fields impact",
        "schema_class": "MultipleChoiceWithReasoning",
    },
    "mc_level5_elimination": {
        "constraint_level": 5,
        "description": "Full structured response with option elimination",
        "expected_insight": "Complex structured output with elimination reasoning",
        "schema_class": "MCQWithFullElimination",
    },
    "mc_level6_comprehensive": {
        "constraint_level": 6,
        "description": "Comprehensive multiple choice with all fields",
        "expected_insight": "Maximum structured output complexity",
        "schema_class": "ComprehensiveMultipleChoice",
    },
}

