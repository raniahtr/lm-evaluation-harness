# MedQA Confidence Calibration Analysis Report

## Executive Summary

This report analyzes the relationship between model confidence and answer correctness for MedQA Level 2, and compares performance across Levels 1, 2, and 3.

**Key Findings:**
- Level 1 Accuracy: 0.6182
- Level 2 Accuracy: 0.0000
- Level 3 Accuracy: 0.5978

## 1. Accuracy Comparison

| Level | Accuracy | Description |
|-------|----------|-------------|
| Level 1 | 0.6182 | StrictMultipleChoice (answer only) |
| Level 2 | 0.0000 | MCQAnswerWithConfidence (answer + confidence) |
| Level 3 | 0.5978 | MCQAnswerWithJustification (answer + justification) |

### Statistical Significance Tests


**Level 1 vs Level 2:**
- McNemar's test: χ² = 785.0013, p = 0.0000 ***
- Level 1 only correct: 787
- Level 2 only correct: 0

**Level 1 vs Level 3:**
- McNemar's test: χ² = 15.6250, p = 0.0001 ***
- Level 1 only correct: 33
- Level 3 only correct: 7

**Level 2 vs Level 3:**
- McNemar's test: χ² = 759.0013, p = 0.0000 ***
- Level 2 only correct: 0
- Level 3 only correct: 761


## 2. Level 2 Confidence Calibration Analysis

### Calibration Metrics


- **Expected Calibration Error (ECE)**: 0.7171
  - Lower is better (0 = perfect calibration)
- **Brier Score**: 0.5814
  - Lower is better (0 = perfect predictions)
- **Pearson Correlation**: r = nan, p = nan
- **Spearman Correlation**: ρ = nan, p = nan

### Confidence Statistics

- Mean: 0.7171
- Std: 0.2593
- Min: 0.0000
- Max: 1.0000
- Number of samples with confidence: 1252

### Per-Bin Calibration Analysis

| Confidence Bin | Count | Avg Confidence | Accuracy | Calibration Error |
|----------------|-------|----------------|----------|-------------------|
| 0.0-0.2 | 15 | 0.0000 | 0.0000 | 0.0000 |
| 0.4-0.6 | 677 | 0.5000 | 0.0000 | 0.5000 |
| 0.8-1.0 | 560 | 0.9988 | 0.0000 | 0.9988 |

**Overconfident Bins** (confidence > accuracy + 0.1):
- 0.4-0.6: confidence=0.5000, accuracy=0.0000
- 0.8-1.0: confidence=0.9988, accuracy=0.0000


## 3. Level Agreement Analysis

- **Total matched samples**: 1273
- **All levels agree**: 0 (0.00%)
- **Level 1 & 2 agree**: 0 (0.00%)
- **Level 1 & 3 agree**: 1182 (92.85%)
- **Level 2 & 3 agree**: 0 (0.00%)

### Correctness Patterns

- **All correct**: 0 (0.00%)
- **All incorrect**: 479 (37.63%)
- **Only Level 1 correct**: 33
- **Only Level 2 correct**: 0
- **Only Level 3 correct**: 7


## 4. Level 3 Justification Analysis

- **Mean justification length (correct)**: 199.6 characters
- **Mean justification length (incorrect)**: 199.7 characters
- **Std justification length (correct)**: 5.3 characters
- **Std justification length (incorrect)**: 4.1 characters


## 5. Visualizations

The following visualizations have been generated:

1. **Calibration Plot**: Shows the relationship between confidence and accuracy for Level 2
2. **Confidence Distribution**: Histogram of confidence scores for correct vs incorrect answers
3. **Reliability Diagram**: Binned accuracy vs binned confidence with error bars
4. **Confidence Heatmap**: 2D histogram showing confidence vs correctness
5. **Accuracy Comparison**: Bar chart comparing accuracy across all three levels
6. **Agreement Matrix**: Heatmap showing agreement patterns between levels
7. **Correctness Pattern Breakdown**: Distribution of correctness patterns
8. **Justification Length Distribution**: Comparison of justification lengths for correct vs incorrect answers (Level 3)

All plots are saved in the `confidence_plots/` directory.

## 6. Conclusions and Recommendations

### Key Insights

- **Calibration Quality**: Level 2 confidence shows poor calibration (ECE >= 0.2)
- **Confidence-Correctness Correlation**: No significant correlation (r=nan, p>=0.05)
- **Level 1 vs Level 2**: Accuracies differ by 61.82%
- **Level 1 vs Level 3**: Accuracies differ by 2.04%
- **Level 2 vs Level 3**: Accuracies differ by 59.78%

- **Agreement**: 0.00% of samples have all three levels agreeing on the answer

### Recommendations

1. **Confidence Calibration**: Consider recalibrating the confidence scores using temperature scaling or Platt scaling to improve calibration.

2. **Schema Impact**: The addition of confidence (Level 2) and justification (Level 3) fields does not significantly impact accuracy compared to Level 1, suggesting the model can handle these additional constraints effectively.

3. **Future Work**: 
   - Investigate cases where levels disagree to understand model uncertainty
   - Analyze the relationship between justification quality and correctness
   - Consider using confidence scores for selective prediction or uncertainty quantification

---
*Report generated automatically by analyze_confidence_calibration.py*
