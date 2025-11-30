# Schema Quick Reference

## PubMedQA Schemas (Yes/No/Maybe)

| Level | Schema Name | Constraint | Key Fields | Expected Insight |
|-------|-------------|------------|------------|------------------|
| **1** | `StrictAnswerOnly` | ⭐⭐⭐⭐⭐ | `answer: Literal["yes","no","maybe"]` | Pure structural compliance baseline |
| **2** | `StructuredAnswer` | ⭐⭐⭐⭐ | `answer` + `confidence: float` | Impact of numeric type constraints |
| **3** | `AnswerWithOptionalReasoning` | ⭐⭐⭐ | `answer` + `reasoning: Optional[str]` | - structure + flexibility |
| **4** | `ComprehensiveAnswer` | ⭐⭐ | `answer` + `confidence` + `reasoning` + `key_evidence[]` | Complex structured output impact |
| **5** | `FlexibleAnswer` | ⭐ | `answer` + `confidence: Optional` + allow extras | Flexibility benefit analysis |
| **6** | `FreeFormAnswer` | ⭐ | `answer: str` (free-form) + `reasoning` | Format vs content constraint |
| **7** | `MinimalStructure` | ⭐ | `response: str` (any text) | Minimal structure benefit |

## Multiple Choice Schemas (A/B/C/D)

| Level | Schema Name | Constraint | Key Fields | Use Case |
|-------|-------------|------------|------------|----------|
| **1** | `StrictMultipleChoice` | ⭐⭐⭐⭐⭐ | `answer: Literal["A","B","C","D"]` | MedQA, MedMCQA baseline |
| **3** | `MultipleChoiceWithReasoning` | ⭐⭐⭐ | `answer` + `reasoning` + `confidence` | Reasoning impact test |
| **4** | `ComprehensiveMultipleChoice` | ⭐⭐ | `answer` + `reasoning` + `confidence` + `key_concepts[]` + `differential_diagnosis[]` | Full medical diagnosis structure |

## Usage Examples

### Using Pydantic Model (Recommended)

```bash
# Level 1: Strict
python3 -m lm_eval \
  --model sglang-schema \
  --model_args pretrained=OpenMeditron/Meditron3-8B,base_url=http://localhost:31000,schema_model=schemas.medical_qa_schemas.StrictAnswerOnly \
  --tasks pubmedqa_generation \
  --output_path ./results/pubmedqa_level1.json

# Level 3: Optional Reasoning 
python3 -m lm_eval \
  --model sglang-schema \
  --model_args pretrained=OpenMeditron/Meditron3-8B,base_url=http://localhost:31000,schema_model=schemas.medical_qa_schemas.AnswerWithOptionalReasoning \
  --tasks pubmedqa_generation \
  --output_path ./results/pubmedqa_level3.json
```

### Using JSON Schema File ( Not yet used)

First, export schemas:
```bash
cd schemas
python3 export_schemas.py
```

Then use:
```bash
--model_args schema_file=schemas/json_schemas/pubmedqa_level1_strict.json
```


