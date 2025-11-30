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
from pydantic import BaseModel, Field, field_validator


# ============================================================================
# LEVEL 1: MOST CONSTRAINING - Strict Enum with No Extras
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
# LEVEL 2: High Constraint - Enum + Required Fields + Type Constraints
# ============================================================================
# Insight: Tests if adding required fields and type constraints (numbers, arrays)
#          affects performance. Measures impact of structural complexity.

class StructuredAnswer(BaseModel):
    """Level 2: Answer with confidence score and strict typing."""
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
# LEVEL 3: Medium-High Constraint - Enum + Optional Reasoning
# ============================================================================
# Insight: Tests if allowing optional free-text reasoning helps or hurts.
#          Measures trade-off between structure and flexibility.

class AnswerWithOptionalReasoning(BaseModel):
    """Level 3: Answer with optional reasoning field."""
    answer: Literal["yes", "no", "maybe"] = Field(
        description="The answer to the question"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Optional brief reasoning for the answer"
    )
    
    class Config:
        extra = "forbid"


# ============================================================================
# LEVEL 4: Medium Constraint - Multiple Required Fields with Validation
# ============================================================================
# Insight: Tests complex structured output with multiple required fields.
#          Measures impact of requiring the model to provide reasoning,
#          confidence, and structured metadata.

class ComprehensiveAnswer(BaseModel):
    """Level 4: Full structured response with reasoning and confidence."""
    answer: Literal["yes", "no", "maybe"] = Field(
        description="The answer to the question"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0"
    )
    reasoning: str = Field(
        min_length=10,
        max_length=500,
        description="Brief reasoning explaining the answer (10-500 characters)"
    )
    key_evidence: List[str] = Field(
        min_length=1,
        max_length=5,
        description="List of 1-5 key pieces of evidence from the abstract"
    )
    
    class Config:
        extra = "forbid"


# ============================================================================
# LEVEL 5: Medium-Low Constraint - Flexible Structure with Validation
# ============================================================================
# Insight: Tests if allowing additional fields (additionalProperties: true)
#          while maintaining core structure improves performance.
#          Measures impact of flexibility vs. strictness.

class FlexibleAnswer(BaseModel):
    """Level 5: Core structure required, but allows additional fields."""
    answer: Literal["yes", "no", "maybe"] = Field(
        description="The answer to the question"
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional confidence score"
    )
    
    class Config:
        extra = "allow"  # Allows additional fields


# ============================================================================
# LEVEL 6: Low Constraint - String Answer with Minimal Validation
# ============================================================================
# Insight: Tests if constraining to JSON structure but allowing free-form
#          answer text (not enum) helps. Measures impact of format vs. content.

class FreeFormAnswer(BaseModel):
    """Level 6: Structured format but free-form answer text."""
    answer: str = Field(
        min_length=1,
        max_length=100,
        description="The answer to the question (free-form text)"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Optional reasoning"
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

class StrictMultipleChoice(BaseModel):
    """Level 1 for MC: Strict enum for multiple choice answers."""
    answer: Literal["A", "B", "C", "D"] = Field(
        description="The multiple choice answer"
    )
    
    class Config:
        extra = "forbid"


class MultipleChoiceWithReasoning(BaseModel):
    """Level 3 for MC: Answer with required reasoning."""
    answer: Literal["A", "B", "C", "D"] = Field(
        description="The multiple choice answer"
    )
    reasoning: str = Field(
        min_length=20,
        max_length=500,
        description="Detailed reasoning explaining why this answer is correct"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score"
    )
    
    class Config:
        extra = "forbid"


class ComprehensiveMultipleChoice(BaseModel):
    """Level 4 for MC: Full structured response for medical diagnosis."""
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
    "level3_optional_reasoning": AnswerWithOptionalReasoning,
    "level4_comprehensive": ComprehensiveAnswer,
    "level5_flexible": FlexibleAnswer,
    "level6_freeform": FreeFormAnswer,
    "level7_minimal": MinimalStructure,
}

MULTIPLE_CHOICE_SCHEMAS = {
    "level1_strict": StrictMultipleChoice,
    "level3_reasoning": MultipleChoiceWithReasoning,
    "level4_comprehensive": ComprehensiveMultipleChoice,
}

# Schema metadata for evaluation tracking
SCHEMA_METADATA = {
    "level1_strict": {
        "constraint_level": 1,
        "description": "Strict enum, no extras",
        "expected_insight": "Baseline structural compliance",
    },
    "level2_structured": {
        "constraint_level": 2,
        "description": "Enum + required numeric field",
        "expected_insight": "Impact of type constraints",
    },
    "level3_optional_reasoning": {
        "constraint_level": 3,
        "description": "Enum + optional text field",
        "expected_insight": "Trade-off: structure vs flexibility",
    },
    "level4_comprehensive": {
        "constraint_level": 4,
        "description": "Multiple required fields with validation",
        "expected_insight": "Complex structured output impact",
    },
    "level5_flexible": {
        "constraint_level": 5,
        "description": "Core structure + allow extras",
        "expected_insight": "Flexibility benefit analysis",
    },
    "level6_freeform": {
        "constraint_level": 6,
        "description": "Structure with free-form answer",
        "expected_insight": "Format vs content constraint impact",
    },
    "level7_minimal": {
        "constraint_level": 7,
        "description": "Minimal JSON structure",
        "expected_insight": "Baseline structured output benefit",
    },
}

