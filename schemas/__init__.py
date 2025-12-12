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
    InvertedCoTAnswer,
    
    # Multiple Choice schemas (A/B/C/D)
    StrictMultipleChoice,
    MCQAnswerWithConfidence,
    MCQAnswerWithConfidenceNew,
    MCQAnswerWithJustification,
    MultipleChoiceWithReasoning,
    MCQWithFullElimination,
    ComprehensiveMultipleChoice,
    MCQInvertedCoTAnswer,
    
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
    "InvertedCoTAnswer",
    "StrictMultipleChoice",
    "MCQAnswerWithConfidence",
    "MCQAnswerWithConfidenceNew",
    "MCQAnswerWithJustification",
    "MultipleChoiceWithReasoning",
    "MCQWithFullElimination",
    "ComprehensiveMultipleChoice",
    "MCQInvertedCoTAnswer",
    "PUBMEDQA_SCHEMAS",
    "MULTIPLE_CHOICE_SCHEMAS",
    "SCHEMA_METADATA",
]

