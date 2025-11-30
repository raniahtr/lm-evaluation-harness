"""
Medical QA Schema Definitions

This module provides Pydantic schemas for evaluating LLM performance
with different constraint levels on medical QA tasks.
"""

from .medical_qa_schemas import (
    # PubMedQA schemas (Yes/No/Maybe)
    StrictAnswerOnly,
    StructuredAnswer,
    AnswerWithOptionalReasoning,
    ComprehensiveAnswer,
    FlexibleAnswer,
    FreeFormAnswer,
    MinimalStructure,
    
    # Multiple Choice schemas (A/B/C/D)
    StrictMultipleChoice,
    MultipleChoiceWithReasoning,
    ComprehensiveMultipleChoice,
    
    # Registry
    PUBMEDQA_SCHEMAS,
    MULTIPLE_CHOICE_SCHEMAS,
    SCHEMA_METADATA,
)

__all__ = [
    "StrictAnswerOnly",
    "StructuredAnswer",
    "AnswerWithOptionalReasoning",
    "ComprehensiveAnswer",
    "FlexibleAnswer",
    "FreeFormAnswer",
    "MinimalStructure",
    "StrictMultipleChoice",
    "MultipleChoiceWithReasoning",
    "ComprehensiveMultipleChoice",
    "PUBMEDQA_SCHEMAS",
    "MULTIPLE_CHOICE_SCHEMAS",
    "SCHEMA_METADATA",
]

