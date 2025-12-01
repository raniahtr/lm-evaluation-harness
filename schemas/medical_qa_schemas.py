"""
Medical QA Schema Constraint Evaluation Strategy

This module defines a progression of Pydantic schemas from most to least constraining,
designed to evaluate how schema constraints affect LLM performance on medical QA tasks.

Each schema level provides different insights:
- Level 1-2: Test strict structural compliance
- Level 3-4: Test semantic constraints and reasoning
- Level 5-6: Test flexibility and natural language handling
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator, constr


# ============================================================================
# LEVEL 1: Strict Answer Only
# ============================================================================
# Insight: Tests pure structural compliance - can the model output valid JSON
#          with exact enum values? This is the baseline constraint test.

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
# Insight: Tests if adding required fields and type constraints (numbers, arrays)
#          affects performance. Measures impact of structural complexity.

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
# Insight: 

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
# LEVEL 4: Answer with reasoning and confidence score
# ============================================================================
# Insight: 

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
# Insight: 

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
# Insight: 

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
# LEVEL 7: Very Low Constraint - Minimal Structure
# ============================================================================
# Insight: Tests if minimal JSON structure (just an object) helps vs. vanilla.
#          Measures baseline benefit of structured output format.

class MinimalStructure(BaseModel):
    """Level 7: Minimal structure - just requires valid JSON object."""
    response: str = Field(
        description="The model's response in any format"
    )
    
    class Config:
        extra = "allow"


# ============================================================================
# MULTIPLE CHOICE SCHEMAS (for MedQA, MedMCQA)
# ============================================================================

# ============================================================================
# LEVEL 1: Strict Multiple Choice
# ============================================================================
# Insight: Provides a baseline by restricting output to a fixed set of valid multiple choice answers (A/B/C/D) only; tests pure structural compliance and model's ability to produce valid JSON.
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
# Insight: Adds numeric "confidence" to strict answer for richer, quantitative outputs and error 
#diagnosis. Useful to measure whether LMs that can express probabilistic beliefs produce 
#better-calibrated answers.

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
# Insight: Adds "justification" field to provide detailed explanation for the answer.
#          Useful to measure whether LMs can produce well-reasoned answers.
class MCQAnswerWithJustification(BaseModel):
    """Level 3 for MC: Answer with justification."""
    answer: Literal["A", "B", "C", "D"] = Field(
        description="The multiple choice answer"
    )
    justification: constr(max_length=200)
    
    class Config:
        extra = "forbid"


# ============================================================================
# LEVEL 4: Answer with reasoning and confidence score
# ============================================================================
# Insight: Adds "reasoning" and 'confidence' field.
#          Useful to measure whether the two options combine well.
class MultipleChoiceWithReasoning(BaseModel):
    """Level 4 for MC: Answer with required reasonin and confidence score."""
    answer: Literal["A", "B", "C", "D"] = Field(
        description="The multiple choice answer"
    )
    reasoning: str = Field(
        min_length=20,
        max_length=200,
        description="Detailed reasoning explaining why this answer is correct"
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
# Insight: Adds "eliminated" field to provide detailed explanation for the answer.
#          schema is per-option elimination : for every eliminated option, we have structured reasoning.

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
# Insight: Very dense structured output for medical diagnosis.

class ComprehensiveMultipleChoice(BaseModel):
    """Level 6 for MC: Full structured response for medical diagnosis."""
    answer: Literal["A", "B", "C", "D"] = Field(
        description="The multiple choice answer"
    )
    reasoning: str = Field(
        min_length=20,
        max_length=500,
        description="Detailed reasoning"
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
    "level7_minimal": MinimalStructure,
}

MULTIPLE_CHOICE_SCHEMAS = {
    "level1_strict": StrictMultipleChoice,
    "level2_confidence": MCQAnswerWithConfidence,
    "level3_justification": MCQAnswerWithJustification,
    "level4_reasoning": MultipleChoiceWithReasoning,
    "level5_elimination": MCQWithFullElimination,
    "level6_comprehensive": ComprehensiveMultipleChoice,
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
    "level7_minimal": {
        "constraint_level": 7,
        "description": "Minimal JSON structure",
        "expected_insight": "Baseline structured output benefit",
        "schema_class": "MinimalStructure",
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

