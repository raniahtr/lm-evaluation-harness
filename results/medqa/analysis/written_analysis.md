# MedQA 4-Options Generation Task: Comprehensive Analysis Report

## Executive Summary

This report summarizes results from evaluating the Meditron3-8B model on the MedQA 4-options generation task using schema-constrained generation. We tested six schema levels (L1–L6) against a baseline, analyzed confidence calibration, and investigated why Level 5 underperformed.

## 1. Methodology and Experimental Setup

### 1.1 Baseline Prompt Optimization

We changed the generation prompt from an instruction-based format to a neutral question format to improve baseline performance.

**Original Prompt:**
```
Q: {question}
(A) {option A} (B) {option B} (C) {option C} (D) {option D}
A: Provide the final answer enclosed in boxed{the_answer}.
```
**Result:** 47.13% accuracy

**Neutral Prompt (Final):**
```
Q: {question}
(A) {option A}
(B) {option B}
(C) {option C}
(D) {option D}

What is the correct answer?
```
**Result:** 49.88% accuracy (+2.75 percentage points)

The neutral prompt removed explicit formatting instructions and improved answer extraction, establishing a better baseline for schema comparisons.

### 1.2 Schema Constraint Levels

We evaluated six schema levels with increasing complexity:

- **L1 (Strict Answer Only):** Requires only `answer` field with strict enum `["A", "B", "C", "D"]`
- **L2 (Answer + Confidence):** Adds required `confidence` field (0.0-1.0)
- **L3 (Answer + Justification):** Adds required `justification` text field
- **L4 (Answer + Reasoning + Confidence):** Combines reasoning and confidence
- **L5 (Option Elimination):** Requires `answer`, `eliminated` array (3 options with reasons), and `key_evidence`
- **L6 (Comprehensive):** Includes answer, reasoning, confidence, key_concepts, and optional differential_diagnosis

All schemas use Pydantic validation with `extra="forbid"` to enforce strict compliance.

## 2. Overall Performance Results

### 2.1 Performance Comparison Table

| Level | Exact Match | Std Error | Percentage | Improvement vs Baseline |
|-------|-------------|-----------|------------|------------------------|
| Baseline | 0.4988 | 0.0140 | 49.88% | - |
| L1 | 0.6214 | 0.0136 | 62.14% | +12.25 (+24.6%) |
| L2 | 0.6214 | 0.0136 | 62.14% | +12.25 (+24.6%) |
| L3 | 0.6127 | 0.0137 | 61.27% | +11.39 (+22.8%) |
| L4 | 0.6229 | 0.0136 | 62.29% | +12.41 (+24.9%) |
| **L5** | **0.5852** | **0.0138** | **58.52%** | **+8.64 (+17.3%)** |
| L6 | 0.6229 | 0.0136 | 62.29% | +12.41 (+24.9%) |

### 2.2 Key Findings

Schema-constrained generation improved accuracy by ~12 percentage points (24.6% relative) over the baseline. L1, L2, L4, and L6 perform similarly (~62%), while L5 underperforms at 58.52% (-3.77 percentage points vs the average of other schemas).

## 3. Error Analysis

### 3.1 Error Breakdown by Level

| Level | Correct | Wrong Answer | Invalid | JSON Failures | Accuracy |
|-------|---------|--------------|---------|---------------|----------|
| Baseline | 635 | 351 | 287 | - | 49.88% |
| L1 | 791 | 460 | 22 | 22 | 62.14% |
| L2 | 791 | 460 | 22 | 22 | 62.14% |
| L3 | 780 | 453 | 40 | 76 | 61.27% |
| L4 | 793 | 458 | 22 | 398 | 62.29% |
| L5 | 745 | 427 | 101 | 118 | 58.52% |
| L6 | 793 | 458 | 22 | 418 | 62.29% |

**Observations:**
- Invalid answers dropped from 287 (22.5%) in baseline to ~22 (1.7%) in most schema levels
- L5 has 101 invalid answers (7.9%), significantly higher than other schemas
- L5 has 118 JSON failures (9.3%), indicating schema compliance issues

### 3.2 Answer Distribution Changes

Schemas shift the answer distribution compared to baseline:

**Baseline Distribution:**
- A: 29.6%, B: 24.6%, C: 21.6%, D: 20.2%, Invalid: 22.5%

**Schema Levels (L1-L6, excluding L5):**
- A: ~27-28%, B: ~24-25%, C: ~26-27%, D: ~20-21%, Invalid: ~1.7%

Schemas reduce invalid outputs and slightly favor option C (which had higher baseline accuracy: 73.7% vs 56.3% for D).

## 4. Level 5 Underperformance Analysis

### 4.1 Root Cause Investigation

L5 requires:
1. A chosen answer
2. An `eliminated` array with exactly 3 options, each with a reason
3. A `key_evidence` field

**Performance Gap:** L5 achieves 58.52% vs 62.03% average for other schemas (gap: -3.51 percentage points).

### 4.2 Contributing Factors

1. **Schema Complexity:** L5 requires structured elimination reasoning, which increases cognitive load and generation complexity.

2. **Validation Failures:** 
   - 118 JSON failures (9.3% vs ~1.7% for L1-L2)
   - 101 invalid answers (7.9% vs ~1.7% for L1-L2)
   - The model struggles to consistently produce valid elimination arrays

3. **Elimination Accuracy:** Analysis shows the model sometimes eliminates correct options or provides incorrect reasoning, leading to wrong final answers.

4. **Answer Distribution:** L5 shows lower accuracy across all answer types (A: 62.04%, B: 59.22%, C: 60.40%, D: 50.57%) compared to other schemas (A: ~65%, B: ~63%, C: ~63%, D: ~55%).

**Conclusion:** The elimination requirement adds complexity that degrades performance. The model struggles to reliably identify and reason about incorrect options, leading to more errors.

## 5. Confidence Calibration Analysis

### 5.1 Calibration Metrics by Level

We evaluated Expected Calibration Error (ECE) and correlation between confidence and correctness for levels with confidence scores (L2, L4, L6):

| Level | ECE | Correlation | Mean Conf (Correct) | Mean Conf (Incorrect) | Confidence Gap |
|-------|-----|-------------|---------------------|----------------------|----------------|
| L2 | 0.3332 | 0.0055 | 0.9524 | 0.9511 | 0.0013 |
| L4 | 0.2774 | 0.0123 | 0.9123 | 0.9012 | 0.0111 |
| L6 | 0.2498 | 0.0187 | 0.8891 | 0.8715 | 0.0176 |

**Key Findings:**
- All levels show poor calibration (ECE > 0.2; ideal < 0.1)
- Correlation between confidence and correctness is near zero (0.0055-0.0187)
- Mean confidence is high (~0.85-0.95) regardless of correctness
- Confidence gap between correct and incorrect answers is minimal (0.0013-0.0176)

**Interpretation:** The model is overconfident and poorly calibrated. Confidence scores do not reliably indicate correctness, limiting their use for uncertainty estimation or filtering.

### 5.2 Probability-Based Confidence Schema (L2 Variant)

We tested a variant of L2 (`MCQAnswerWithConfidenceNew`) that requires:
- Full probability distribution over all options (must sum to 1.0)
- Confidence must equal the probability of the chosen answer

**Results Comparison:**

| Metric | Old L2 | New L2 | Change |
|--------|--------|--------|--------|
| Accuracy | 62.14% | 62.22% | +0.08% |
| Invalid Samples | 22 (1.7%) | 36 (2.8%) | +14 |
| ECE | 0.3332 | 0.3865 | +0.0533 (worse) |
| Correlation | 0.0055 | 0.0263 | +0.0208 |
| Confidence Gap | 0.0013 | 0.0174 | +0.0161 |

**Findings:**
- Minimal accuracy improvement (+0.08%)
- Increased validation failures (+14 samples, 1.1%)
- Calibration worsened (ECE increased)
- Slight improvement in correlation and confidence gap, but still very weak

**Conclusion:** The stricter probability constraint does not improve calibration. The model produces extreme probability distributions (88.2% have chosen probability ≥ 0.95, median = 1.0), indicating overconfidence rather than calibrated probabilities.

## 6. Answer Selection Changes: Baseline vs Schemas

When comparing answer selection between baseline and schemas:

- **Same answer:** 87.2% (5,008/5,741 comparisons)
- **Different answer:** 12.8% (733 comparisons)

**When answers differ:**
- Baseline correct → Schema wrong: 221 (30.2%)
- Baseline wrong → Schema correct: 321 (43.8%)

**Most common answer changes:**
- D → C: 144 (19.6%)
- D → B: 107 (14.6%)
- D → A: 89 (12.1%)

Schemas correct more baseline errors than they introduce, with a net improvement. The shift away from option D (which had lower baseline accuracy) contributes to better performance.

## 7. Performance by Answer Type

Analysis across all levels shows systematic difficulty differences:

| Target Answer | Overall Accuracy | Total Questions |
|---------------|------------------|-----------------|
| A | ~63% | 2,471 |
| B | ~62% | 2,163 |
| C | ~62% | 2,422 |
| D | ~54% | 1,855 |

Option D is consistently the hardest across all levels, suggesting either inherent difficulty or systematic bias in the model's training or evaluation.

## 8. Conclusions and Recommendations

### 8.1 Key Takeaways

1. **Schema constraints significantly improve performance:** +12 percentage points (24.6% relative) over baseline, primarily by reducing invalid outputs and improving answer extraction.

2. **Optimal schema levels:** L1, L2, L4, and L6 perform similarly (~62%). L1 (simplest) is recommended for maximum performance with minimal complexity.

3. **Level 5 underperforms:** The elimination requirement adds complexity that degrades performance. Avoid L5 for production use.

4. **Confidence calibration is poor:** Confidence scores are not reliable indicators of correctness. The model is overconfident regardless of accuracy.

5. **Neutral prompts improve baseline:** Removing explicit formatting instructions improved baseline performance by 2.75 percentage points.

### 8.2 Recommendations

**For Production Use:**
- Use **L1** (strict answer only) for maximum performance and simplicity
- Avoid **L5** due to underperformance
- Do not rely on confidence scores for uncertainty estimation or filtering

**For Research:**
- Investigate why option D is consistently harder
- Explore alternative calibration methods (temperature scaling, Platt scaling)
- Consider simplifying L5 schema or removing the elimination requirement
- Study why probability-based constraints don't improve calibration

**For Schema Design:**
- Simpler schemas (L1-L2) perform as well as complex ones (L4, L6)
- Additional fields (reasoning, justification) don't improve accuracy
- Schema complexity should be justified by downstream use cases, not assumed to improve performance

This analysis demonstrates that schema-constrained generation improves answer extraction and accuracy, but more complex schemas do not necessarily improve performance, and confidence calibration remains a challenge.