# MedQA Generation Task Performance Investigation Report

## Executive Summary

This report investigates the performance drop in `medqa_4options_generation` task (0.4713) compared to the multiple-choice baseline (0.6316), representing a **25.4% relative decrease** in performance.

**Key Finding**: The performance drop is primarily due to **model knowledge issues** (35.5% wrong answers) rather than extraction failures (17.4% invalid fallback). However, extraction issues still contribute significantly to the overall performance gap.

## 1. Performance Overview

| Metric | Value |
|--------|-------|
| Generation Task Performance | 0.4713 (47.13%) |
| Multiple-Choice Baseline | 0.6316 (63.16%) |
| Performance Drop | -0.1603 (-25.4% relative) |
| Total Samples Analyzed | 1,273 |

## 2. Root Cause Analysis

### 2.1 Failure Mode Breakdown

| Failure Mode | Count | Percentage |
|--------------|-------|------------|
| **Wrong Answer Extracted** | 452 | 35.5% |
| **Extraction Failure** | 221 | 17.4% |
| **Correct** | 600 | 47.1% |

**Key Insight**: 
- **35.5%** of failures are due to the model generating the wrong answer (extraction works correctly)
- **17.4%** of failures are due to extraction issues (regex cannot find valid answer)
- This suggests the performance drop is **primarily a knowledge/understanding issue** rather than purely an extraction problem

### 2.2 Extraction Analysis

#### Current Regex Pattern
```regex
(?i)answer\W(?:is)*\W*([A-D])(?:\W|$)|(?:answer|boxed)?{\W*([A-D])\W+
```

**Issues Identified**:
1. **Two capture groups**: The pattern has two `([A-D])` capture groups in an OR pattern, which can cause confusion in group selection
2. **Invalid fallback rate**: 17.4% of samples result in `[invalid]` fallback
3. **Pattern complexity**: The pattern may not match all valid output formats

#### Extraction Statistics
- Valid extractions: 1,052 (82.6%)
- Invalid fallback: 221 (17.4%)
- Extraction matches target: 600 (47.1%)

### 2.3 Model Output Format Analysis

| Output Format | Count | Percentage | Success Rate |
|---------------|-------|------------|--------------|
| standalone_letter | 723 | 56.8% | ~47% |
| answer_colon | 468 | 36.8% | ~47% |
| other | 51 | 4.0% | Variable |
| answer_is | 21 | 1.6% | Variable |
| empty | 10 | 0.8% | 0% |
| boxed_format | Rare | <1% | Variable |

**Key Observations**:
1. **Model rarely follows instructions**: Despite prompt asking for `boxed{the_answer}`, model rarely uses this format
2. **Most common formats**: Standalone letters (56.8%) and "Answer: X" format (36.8%)
3. **Format consistency**: Model output is relatively consistent, but doesn't match prompt instructions


## 4. Detailed Findings

### 4.1 Regex Pattern Testing

Tested alternative patterns on the sample set:

| Pattern | Success Rate | Invalid Fallback |
|---------|--------------|------------------|
| current | 47.1% | 17.4% |
| simple_letter | 47.1% | 17.4% |
| flexible | 24.8% | 17.4% |
| answer_colon | 21.4% | 17.4% |
| answer_is | 2.7% | 17.4% |
| boxed_only | 0.0% | 17.4% |

**Finding**: The `simple_letter` pattern (`\b([A-D])\b`) performs identically to the current complex pattern, suggesting the complexity doesn't add value.

### 4.2 Sample Failure Cases

#### Invalid Fallback Examples
- **Case 1**: Model outputs reasoning starting with "A:" but no explicit answer letter
- **Case 2**: Model outputs empty or truncated responses
- **Case 3**: Model outputs repetitive format (A: B: C: D: repeated)

#### Wrong Answer Examples
- **Case 1**: Model outputs "Answer: A" but correct answer is "B"
- **Case 2**: Model outputs "{B}" but correct answer is "D"
- **Case 3**: Model includes reasoning but gives wrong final answer


## Appendix: Files Created

1. **Test Script**: `scripts/test_medqa_extraction.py`
   - Automated extraction testing
   - Pattern comparison
   - Failure reporting

2. **Analysis Notebook**: `results/medqa/medqa_generation_analysis.ipynb`
   - Comprehensive analysis
   - Visualizations
   - Sample failure cases

3. **Extraction Report**: `results/medqa/extraction_analysis_report.txt`
   - Detailed statistics
   - Pattern comparison
   - Failure examples

4. **Investigation Report**: `results/medqa/INVESTIGATION_REPORT.md` (this file)
   - Executive summary
   - Findings and recommendations

