"""
Medical QA Schema Definitions

This module provides Pydantic schemas for evaluating LLM performance
with different constraint levels on medical QA tasks.
"""

from .medical_qa_schemas import (
    # PubMedQA schemas (Yes/No/Maybe)
    StrictAnswerOnly,
    StructuredAnswer,
    AnswerWithReasoning,
    AnswerWithReasoningAndConfidence,
    GroundedAnswer,
    FlexibleGroundedAnswer,
    MinimalStructure,
    
    # Multiple Choice schemas (A/B/C/D)
    StrictMultipleChoice,
    MCQAnswerWithConfidence,
    MCQAnswerWithJustification,
    MultipleChoiceWithReasoning,
    MCQWithFullElimination,
    ComprehensiveMultipleChoice,
    
    # Registry
    PUBMEDQA_SCHEMAS,
    MULTIPLE_CHOICE_SCHEMAS,
    SCHEMA_METADATA,
)

__all__ = [
    "StrictAnswerOnly",
    "StructuredAnswer",
    "AnswerWithReasoning",
    "AnswerWithReasoningAndConfidence",
    "GroundedAnswer",
    "FlexibleGroundedAnswer",
    "MinimalStructure",
    "StrictMultipleChoice",
    "MCQAnswerWithConfidence",
    "MCQAnswerWithJustification",
    "MultipleChoiceWithReasoning",
    "MCQWithFullElimination",
    "ComprehensiveMultipleChoice",
    "PUBMEDQA_SCHEMAS",
    "MULTIPLE_CHOICE_SCHEMAS",
    "SCHEMA_METADATA",
]

