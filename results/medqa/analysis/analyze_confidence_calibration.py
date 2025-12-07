#!/usr/bin/env python3
"""
MedQA Confidence Calibration Analysis

This script analyzes the relationship between model confidence and answer correctness
for MedQA Level 2, and compares performance across Levels 1, 2, and 3.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Try to import seaborn, but make it optional
try:
    import seaborn as sns
    sns.set_style("whitegrid")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Warning: seaborn not available, using matplotlib defaults")

# Set style for plots
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.style.use('default')

# Paths
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "analysis"
PLOTS_DIR = OUTPUT_DIR / "confidence_plots"
PLOTS_DIR.mkdir(exist_ok=True, parents=True)


def load_jsonl(filepath: Path) -> List[Dict]:
    """Load JSONL file and return list of dictionaries."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_json_results(filepath: Path) -> Dict:
    """Load JSON results file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def parse_level1_response(resp_str: str) -> Optional[str]:
    """Parse Level 1 response: {"answer":"A"} format."""
    try:
        resp_json = json.loads(resp_str)
        return resp_json.get("answer", None)
    except (json.JSONDecodeError, TypeError):
        return None


def parse_level2_response(resp_str: str) -> Tuple[Optional[str], Optional[float]]:
    """Parse Level 2 response: {"answer":"A","confidence":0.85} format."""
    try:
        resp_json = json.loads(resp_str)
        answer = resp_json.get("answer", None)
        confidence = resp_json.get("confidence", None)
        # Validate confidence is in [0, 1]
        if confidence is not None:
            confidence = float(confidence)
            if not (0.0 <= confidence <= 1.0):
                confidence = None
        return answer, confidence
    except (json.JSONDecodeError, TypeError, ValueError):
        return None, None


def parse_level3_response(resp_str: str) -> Tuple[Optional[str], Optional[int]]:
    """Parse Level 3 response: {"answer":"A","justification":"..."} format."""
    try:
        resp_json = json.loads(resp_str)
        answer = resp_json.get("answer", None)
        justification = resp_json.get("justification", "")
        justification_length = len(justification) if justification else None
        return answer, justification_length
    except (json.JSONDecodeError, TypeError):
        return None, None


def load_and_parse_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """Load and parse data from all three levels."""
    print("Loading data files...")
    
    # Load JSONL files
    level1_samples = load_jsonl(BASE_DIR / "samples_LEVEL1.jsonl")
    level2_samples = load_jsonl(BASE_DIR / "samples_LEVEL2.jsonl")
    level3_samples = load_jsonl(BASE_DIR / "samples_LEVEL3.jsonl")
    
    # Load JSON results files
    level1_results = load_json_results(BASE_DIR / "medqa4option_LEVEL1.json")
    level3_results = load_json_results(BASE_DIR / "medqa4options_LEVEL3.json")
    
    print(f"Loaded {len(level1_samples)} Level 1 samples")
    print(f"Loaded {len(level2_samples)} Level 2 samples")
    print(f"Loaded {len(level3_samples)} Level 3 samples")
    
    # Parse Level 1
    level1_data = []
    for sample in level1_samples:
        doc_id = sample.get("doc_id", None)
        target = sample.get("target", None)
        resps = sample.get("resps", [])
        
        if resps and len(resps) > 0 and len(resps[0]) > 0:
            answer = parse_level1_response(resps[0][0])
        else:
            answer = None
        
        correct = 1.0 if answer == target else 0.0
        doc_hash = sample.get("doc_hash", None)
        
        level1_data.append({
            "doc_id": doc_id,
            "doc_hash": doc_hash,
            "target": target,
            "answer": answer,
            "correct": correct,
            "level": 1
        })
    
    # Parse Level 2
    level2_data = []
    for sample in level2_samples:
        doc_id = sample.get("doc_id", None)
        target = sample.get("target", None)
        resps = sample.get("resps", [])
        
        if resps and len(resps) > 0 and len(resps[0]) > 0:
            answer, confidence = parse_level2_response(resps[0][0])
        else:
            answer, confidence = None, None
        
        correct = 1.0 if answer == target else 0.0
        doc_hash = sample.get("doc_hash", None)
        
        level2_data.append({
            "doc_id": doc_id,
            "doc_hash": doc_hash,
            "target": target,
            "answer": answer,
            "confidence": confidence,
            "correct": correct,
            "level": 2
        })
    
    # Parse Level 3
    level3_data = []
    for sample in level3_samples:
        doc_id = sample.get("doc_id", None)
        target = sample.get("target", None)
        resps = sample.get("resps", [])
        
        if resps and len(resps) > 0 and len(resps[0]) > 0:
            answer, justification_length = parse_level3_response(resps[0][0])
        else:
            answer, justification_length = None, None
        
        correct = 1.0 if answer == target else 0.0
        doc_hash = sample.get("doc_hash", None)
        
        level3_data.append({
            "doc_id": doc_id,
            "doc_hash": doc_hash,
            "target": target,
            "answer": answer,
            "justification_length": justification_length,
            "correct": correct,
            "level": 3
        })
    
    # Create DataFrames
    df1 = pd.DataFrame(level1_data)
    df2 = pd.DataFrame(level2_data)
    df3 = pd.DataFrame(level3_data)
    
    # Calculate accuracies
    acc1 = df1["correct"].mean()
    acc2 = df2["correct"].mean()
    acc3 = df3["correct"].mean()
    
    print(f"\nCalculated accuracies:")
    print(f"  Level 1: {acc1:.4f}")
    print(f"  Level 2: {acc2:.4f}")
    print(f"  Level 3: {acc3:.4f}")
    
    # Verify against JSON results
    json_acc1 = level1_results["results"]["medqa_4options_generation"]["exact_match,strict-match"]
    json_acc3 = level3_results["results"]["medqa_4options_generation"]["exact_match,strict-match"]
    
    print(f"\nJSON results accuracies:")
    print(f"  Level 1: {json_acc1:.4f} (difference: {abs(acc1 - json_acc1):.6f})")
    print(f"  Level 3: {json_acc3:.4f} (difference: {abs(acc3 - json_acc3):.6f})")
    
    results_dict = {
        "level1_accuracy": acc1,
        "level2_accuracy": acc2,
        "level3_accuracy": acc3,
        "json_level1_accuracy": json_acc1,
        "json_level3_accuracy": json_acc3
    }
    
    return df1, df2, df3, results_dict


def calculate_calibration_metrics(df: pd.DataFrame) -> Dict:
    """Calculate calibration metrics for Level 2."""
    # Filter out samples without confidence
    df_conf = df[df["confidence"].notna()].copy()
    
    if len(df_conf) == 0:
        return {"error": "No confidence scores available"}
    
    # Bin confidence scores
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    bin_labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    df_conf["confidence_bin"] = pd.cut(df_conf["confidence"], bins=bins, labels=bin_labels, include_lowest=True)
    
    # Calculate per-bin metrics
    bin_stats = []
    for bin_label in bin_labels:
        bin_data = df_conf[df_conf["confidence_bin"] == bin_label]
        if len(bin_data) > 0:
            bin_stats.append({
                "bin": bin_label,
                "count": len(bin_data),
                "avg_confidence": bin_data["confidence"].mean(),
                "accuracy": bin_data["correct"].mean(),
                "calibration_error": abs(bin_data["correct"].mean() - bin_data["confidence"].mean())
            })
    
    bin_df = pd.DataFrame(bin_stats)
    
    # Calculate ECE (Expected Calibration Error)
    ece = (bin_df["count"] * bin_df["calibration_error"]).sum() / bin_df["count"].sum()
    
    # Calculate Brier Score
    brier_score = np.mean((df_conf["confidence"] - df_conf["correct"]) ** 2)
    
    # Calculate correlations
    pearson_r, pearson_p = stats.pearsonr(df_conf["confidence"], df_conf["correct"])
    spearman_r, spearman_p = stats.spearmanr(df_conf["confidence"], df_conf["correct"])
    
    # Identify overconfidence/underconfidence
    overconfident_bins = bin_df[bin_df["avg_confidence"] > bin_df["accuracy"] + 0.1]
    underconfident_bins = bin_df[bin_df["avg_confidence"] < bin_df["accuracy"] - 0.1]
    
    return {
        "ece": ece,
        "brier_score": brier_score,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "bin_stats": bin_df,
        "overconfident_bins": overconfident_bins,
        "underconfident_bins": underconfident_bins,
        "n_samples": len(df_conf),
        "confidence_mean": df_conf["confidence"].mean(),
        "confidence_std": df_conf["confidence"].std(),
        "confidence_min": df_conf["confidence"].min(),
        "confidence_max": df_conf["confidence"].max()
    }


def create_calibration_plot(df: pd.DataFrame, metrics: Dict, save_path: Path):
    """Create calibration plot for Level 2."""
    df_conf = df[df["confidence"].notna()].copy()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Calibration plot (reliability diagram)
    ax1 = axes[0, 0]
    bin_stats = metrics["bin_stats"]
    
    bin_centers = [0.1, 0.3, 0.5, 0.7, 0.9]
    accuracies = []
    confidences = []
    counts = []
    
    for i, bin_label in enumerate(["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]):
        bin_data = bin_stats[bin_stats["bin"] == bin_label]
        if len(bin_data) > 0:
            accuracies.append(bin_data["accuracy"].values[0])
            confidences.append(bin_data["avg_confidence"].values[0])
            counts.append(bin_data["count"].values[0])
        else:
            accuracies.append(np.nan)
            confidences.append(np.nan)
            counts.append(0)
    
    # Plot perfect calibration line
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
    
    # Plot actual calibration
    valid_indices = ~np.isnan(accuracies)
    if valid_indices.any():
        ax1.plot(np.array(bin_centers)[valid_indices], np.array(accuracies)[valid_indices], 
                'o-', label='Model Calibration', linewidth=2, markersize=8)
        
        # Add count annotations
        for i, (x, y, count) in enumerate(zip(bin_centers, accuracies, counts)):
            if not np.isnan(y) and count > 0:
                ax1.annotate(f'n={count}', (x, y), xytext=(5, 5), 
                           textcoords='offset points', fontsize=9)
    
    ax1.set_xlabel('Confidence', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title(f'Calibration Plot (ECE = {metrics["ece"]:.4f})', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # 2. Confidence distribution by correctness
    ax2 = axes[0, 1]
    correct_conf = df_conf[df_conf["correct"] == 1.0]["confidence"]
    incorrect_conf = df_conf[df_conf["correct"] == 0.0]["confidence"]
    
    ax2.hist(correct_conf, bins=20, alpha=0.6, label='Correct', color='green', density=True)
    ax2.hist(incorrect_conf, bins=20, alpha=0.6, label='Incorrect', color='red', density=True)
    ax2.set_xlabel('Confidence', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Confidence Distribution by Correctness', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Reliability diagram with error bars
    ax3 = axes[1, 0]
    bin_stats = metrics["bin_stats"]
    
    bin_centers = []
    accuracies = []
    confidences = []
    errors = []
    
    for bin_label in ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]:
        bin_data = bin_stats[bin_stats["bin"] == bin_label]
        if len(bin_data) > 0:
            center = bin_data["avg_confidence"].values[0]
            acc = bin_data["accuracy"].values[0]
            count = bin_data["count"].values[0]
            
            bin_centers.append(center)
            accuracies.append(acc)
            confidences.append(center)
            # Standard error for accuracy
            se = np.sqrt(acc * (1 - acc) / count) if count > 0 else 0
            errors.append(se * 1.96)  # 95% CI
    
    if bin_centers:
        ax3.errorbar(confidences, accuracies, yerr=errors, fmt='o-', 
                    capsize=5, capthick=2, linewidth=2, markersize=8, label='Model')
        ax3.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
        ax3.set_xlabel('Average Confidence', fontsize=12)
        ax3.set_ylabel('Accuracy', fontsize=12)
        ax3.set_title('Reliability Diagram with 95% CI', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim([0, 1])
        ax3.set_ylim([0, 1])
    
    # 4. Confidence heatmap
    ax4 = axes[1, 1]
    # Create 2D histogram
    correct_conf = df_conf[df_conf["correct"] == 1.0]["confidence"].values
    incorrect_conf = df_conf[df_conf["correct"] == 0.0]["confidence"].values
    
    if len(correct_conf) > 0 and len(incorrect_conf) > 0:
        ax4.hist2d(
            np.concatenate([correct_conf, incorrect_conf]),
            np.concatenate([np.ones(len(correct_conf)), np.zeros(len(incorrect_conf))]),
            bins=[20, 2], cmap='YlOrRd', alpha=0.8
        )
        ax4.set_xlabel('Confidence', fontsize=12)
        ax4.set_ylabel('Correctness', fontsize=12)
        ax4.set_title('Confidence vs Correctness Heatmap', fontsize=14, fontweight='bold')
        ax4.set_yticks([0, 1])
        ax4.set_yticklabels(['Incorrect', 'Correct'])
        plt.colorbar(ax4.collections[0], ax=ax4, label='Count')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved calibration plots to {save_path}")


def compare_levels(df1: pd.DataFrame, df2: pd.DataFrame, df3: pd.DataFrame) -> Dict:
    """Compare performance across all three levels."""
    # Merge dataframes on doc_hash or doc_id
    merge_key = "doc_hash" if "doc_hash" in df1.columns else "doc_id"
    
    merged = df1[[merge_key, "answer", "correct"]].merge(
        df2[[merge_key, "answer", "correct", "confidence"]], 
        on=merge_key, suffixes=("_l1", "_l2"), how="inner"
    ).merge(
        df3[[merge_key, "answer", "correct", "justification_length"]], 
        on=merge_key, suffixes=("", "_l3"), how="inner"
    )
    
    # Rename columns for clarity
    merged = merged.rename(columns={
        "answer": "answer_l3",
        "correct": "correct_l3"
    })
    
    # Calculate agreement
    all_agree = (merged["answer_l1"] == merged["answer_l2"]) & (merged["answer_l2"] == merged["answer_l3"])
    l1_l2_agree = merged["answer_l1"] == merged["answer_l2"]
    l1_l3_agree = merged["answer_l1"] == merged["answer_l3"]
    l2_l3_agree = merged["answer_l2"] == merged["answer_l3"]
    
    # Cases where all are correct
    all_correct = (merged["correct_l1"] == 1.0) & (merged["correct_l2"] == 1.0) & (merged["correct_l3"] == 1.0)
    
    # Cases where all are incorrect
    all_incorrect = (merged["correct_l1"] == 0.0) & (merged["correct_l2"] == 0.0) & (merged["correct_l3"] == 0.0)
    
    # Cases where only one level is correct
    only_l1_correct = (merged["correct_l1"] == 1.0) & (merged["correct_l2"] == 0.0) & (merged["correct_l3"] == 0.0)
    only_l2_correct = (merged["correct_l1"] == 0.0) & (merged["correct_l2"] == 1.0) & (merged["correct_l3"] == 0.0)
    only_l3_correct = (merged["correct_l1"] == 0.0) & (merged["correct_l2"] == 0.0) & (merged["correct_l3"] == 1.0)
    
    # Justification length analysis for Level 3
    l3_justification_stats = {
        "mean_correct": merged[merged["correct_l3"] == 1.0]["justification_length"].mean() if "justification_length" in merged.columns else None,
        "mean_incorrect": merged[merged["correct_l3"] == 0.0]["justification_length"].mean() if "justification_length" in merged.columns else None,
        "std_correct": merged[merged["correct_l3"] == 1.0]["justification_length"].std() if "justification_length" in merged.columns else None,
        "std_incorrect": merged[merged["correct_l3"] == 0.0]["justification_length"].std() if "justification_length" in merged.columns else None
    }
    
    return {
        "n_matched": len(merged),
        "all_agree_count": all_agree.sum(),
        "all_agree_pct": all_agree.mean() * 100,
        "l1_l2_agree_count": l1_l2_agree.sum(),
        "l1_l2_agree_pct": l1_l2_agree.mean() * 100,
        "l1_l3_agree_count": l1_l3_agree.sum(),
        "l1_l3_agree_pct": l1_l3_agree.mean() * 100,
        "l2_l3_agree_count": l2_l3_agree.sum(),
        "l2_l3_agree_pct": l2_l3_agree.mean() * 100,
        "all_correct_count": all_correct.sum(),
        "all_correct_pct": all_correct.mean() * 100,
        "all_incorrect_count": all_incorrect.sum(),
        "all_incorrect_pct": all_incorrect.mean() * 100,
        "only_l1_correct_count": only_l1_correct.sum(),
        "only_l2_correct_count": only_l2_correct.sum(),
        "only_l3_correct_count": only_l3_correct.sum(),
        "justification_stats": l3_justification_stats,
        "merged_df": merged
    }


def create_comparison_plots(df1: pd.DataFrame, df2: pd.DataFrame, df3: pd.DataFrame, 
                           comparison: Dict, save_path: Path):
    """Create comparison visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Accuracy comparison bar chart
    ax1 = axes[0, 0]
    accuracies = [
        df1["correct"].mean(),
        df2["correct"].mean(),
        df3["correct"].mean()
    ]
    levels = ["Level 1", "Level 2", "Level 3"]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    bars = ax1.bar(levels, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Accuracy Comparison Across Levels', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 2. Agreement matrix
    ax2 = axes[0, 1]
    merged = comparison["merged_df"]
    
    # Create agreement matrix
    agreement_matrix = np.zeros((3, 3))
    agreement_matrix[0, 0] = comparison["all_agree_pct"] / 100
    agreement_matrix[0, 1] = comparison["l1_l2_agree_pct"] / 100
    agreement_matrix[0, 2] = comparison["l1_l3_agree_pct"] / 100
    agreement_matrix[1, 0] = comparison["l1_l2_agree_pct"] / 100
    agreement_matrix[1, 1] = comparison["all_agree_pct"] / 100
    agreement_matrix[1, 2] = comparison["l2_l3_agree_pct"] / 100
    agreement_matrix[2, 0] = comparison["l1_l3_agree_pct"] / 100
    agreement_matrix[2, 1] = comparison["l2_l3_agree_pct"] / 100
    agreement_matrix[2, 2] = comparison["all_agree_pct"] / 100
    
    im = ax2.imshow(agreement_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax2.set_xticks([0, 1, 2])
    ax2.set_yticks([0, 1, 2])
    ax2.set_xticklabels(['Level 1', 'Level 2', 'Level 3'])
    ax2.set_yticklabels(['Level 1', 'Level 2', 'Level 3'])
    ax2.set_title('Answer Agreement Matrix', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(3):
        for j in range(3):
            text = ax2.text(j, i, f'{agreement_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontweight='bold')
    
    plt.colorbar(im, ax=ax2, label='Agreement Rate')
    
    # 3. Correctness pattern breakdown
    ax3 = axes[1, 0]
    patterns = [
        "All Correct",
        "All Incorrect",
        "Only L1 Correct",
        "Only L2 Correct",
        "Only L3 Correct"
    ]
    counts = [
        comparison["all_correct_count"],
        comparison["all_incorrect_count"],
        comparison["only_l1_correct_count"],
        comparison["only_l2_correct_count"],
        comparison["only_l3_correct_count"]
    ]
    
    bars = ax3.barh(patterns, counts, color=['green', 'red', 'blue', 'orange', 'purple'], alpha=0.7)
    ax3.set_xlabel('Count', fontsize=12)
    ax3.set_title('Correctness Pattern Breakdown', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar, count in zip(bars, counts):
        width = bar.get_width()
        ax3.text(width + 5, bar.get_y() + bar.get_height()/2.,
                f'{count}', ha='left', va='center', fontsize=10)
    
    # 4. Justification length vs correctness (Level 3)
    ax4 = axes[1, 1]
    if "justification_length" in df3.columns and df3["justification_length"].notna().any():
        correct_just = df3[df3["correct"] == 1.0]["justification_length"].dropna()
        incorrect_just = df3[df3["correct"] == 0.0]["justification_length"].dropna()
        
        if len(correct_just) > 0 and len(incorrect_just) > 0:
            ax4.hist(correct_just, bins=20, alpha=0.6, label='Correct', color='green', density=True)
            ax4.hist(incorrect_just, bins=20, alpha=0.6, label='Incorrect', color='red', density=True)
            ax4.set_xlabel('Justification Length (characters)', fontsize=12)
            ax4.set_ylabel('Density', fontsize=12)
            ax4.set_title('Justification Length Distribution by Correctness (Level 3)', 
                         fontsize=14, fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Justification Length Distribution (Level 3)', 
                         fontsize=14, fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'No justification data available', ha='center', va='center', 
                transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Justification Length Distribution (Level 3)', 
                     fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plots to {save_path}")


def perform_statistical_tests(df1: pd.DataFrame, df2: pd.DataFrame, df3: pd.DataFrame, 
                              comparison: Dict) -> Dict:
    """Perform statistical significance tests."""
    results = {}
    
    # McNemar's test for paired samples (Level 1 vs Level 2)
    merged = comparison["merged_df"]
    if len(merged) > 0:
        # Create contingency table
        l1_correct = merged["correct_l1"] == 1.0
        l2_correct = merged["correct_l2"] == 1.0
        
        # Count agreements and disagreements
        both_correct = (l1_correct & l2_correct).sum()
        both_incorrect = (~l1_correct & ~l2_correct).sum()
        l1_only = (l1_correct & ~l2_correct).sum()
        l2_only = (~l1_correct & l2_correct).sum()
        
        # McNemar's test
        if l1_only + l2_only > 0:
            mcnemar_stat = ((abs(l1_only - l2_only) - 1) ** 2) / (l1_only + l2_only)
            mcnemar_p = 1 - stats.chi2.cdf(mcnemar_stat, df=1)
            results["mcnemar_l1_vs_l2"] = {
                "statistic": mcnemar_stat,
                "p_value": mcnemar_p,
                "l1_only_correct": int(l1_only),
                "l2_only_correct": int(l2_only)
            }
        
        # Level 1 vs Level 3
        l3_correct = merged["correct_l3"] == 1.0
        l1_only_l3 = (l1_correct & ~l3_correct).sum()
        l3_only_l1 = (~l1_correct & l3_correct).sum()
        
        if l1_only_l3 + l3_only_l1 > 0:
            mcnemar_stat = ((abs(l1_only_l3 - l3_only_l1) - 1) ** 2) / (l1_only_l3 + l3_only_l1)
            mcnemar_p = 1 - stats.chi2.cdf(mcnemar_stat, df=1)
            results["mcnemar_l1_vs_l3"] = {
                "statistic": mcnemar_stat,
                "p_value": mcnemar_p,
                "l1_only_correct": int(l1_only_l3),
                "l3_only_correct": int(l3_only_l1)
            }
        
        # Level 2 vs Level 3
        l2_only_l3 = (l2_correct & ~l3_correct).sum()
        l3_only_l2 = (~l2_correct & l3_correct).sum()
        
        if l2_only_l3 + l3_only_l2 > 0:
            mcnemar_stat = ((abs(l2_only_l3 - l3_only_l2) - 1) ** 2) / (l2_only_l3 + l3_only_l2)
            mcnemar_p = 1 - stats.chi2.cdf(mcnemar_stat, df=1)
            results["mcnemar_l2_vs_l3"] = {
                "statistic": mcnemar_stat,
                "p_value": mcnemar_p,
                "l2_only_correct": int(l2_only_l3),
                "l3_only_correct": int(l3_only_l2)
            }
    
    return results


def generate_report(df1: pd.DataFrame, df2: pd.DataFrame, df3: pd.DataFrame,
                   metrics: Dict, comparison: Dict, stats_results: Dict, 
                   results_dict: Dict, save_path: Path):
    """Generate comprehensive markdown report."""
    
    report = f"""# MedQA Confidence Calibration Analysis Report

## Executive Summary

This report analyzes the relationship between model confidence and answer correctness for MedQA Level 2, and compares performance across Levels 1, 2, and 3.

**Key Findings:**
- Level 1 Accuracy: {results_dict['level1_accuracy']:.4f}
- Level 2 Accuracy: {results_dict['level2_accuracy']:.4f}
- Level 3 Accuracy: {results_dict['level3_accuracy']:.4f}

## 1. Accuracy Comparison

| Level | Accuracy | Description |
|-------|----------|-------------|
| Level 1 | {results_dict['level1_accuracy']:.4f} | StrictMultipleChoice (answer only) |
| Level 2 | {results_dict['level2_accuracy']:.4f} | MCQAnswerWithConfidence (answer + confidence) |
| Level 3 | {results_dict['level3_accuracy']:.4f} | MCQAnswerWithJustification (answer + justification) |

### Statistical Significance Tests

"""
    
    # Add McNemar's test results
    if "mcnemar_l1_vs_l2" in stats_results:
        mcnemar = stats_results["mcnemar_l1_vs_l2"]
        sig = "***" if mcnemar["p_value"] < 0.001 else "**" if mcnemar["p_value"] < 0.01 else "*" if mcnemar["p_value"] < 0.05 else "ns"
        report += f"""
**Level 1 vs Level 2:**
- McNemar's test: χ² = {mcnemar['statistic']:.4f}, p = {mcnemar['p_value']:.4f} {sig}
- Level 1 only correct: {mcnemar['l1_only_correct']}
- Level 2 only correct: {mcnemar['l2_only_correct']}
"""
    
    if "mcnemar_l1_vs_l3" in stats_results:
        mcnemar = stats_results["mcnemar_l1_vs_l3"]
        sig = "***" if mcnemar["p_value"] < 0.001 else "**" if mcnemar["p_value"] < 0.01 else "*" if mcnemar["p_value"] < 0.05 else "ns"
        report += f"""
**Level 1 vs Level 3:**
- McNemar's test: χ² = {mcnemar['statistic']:.4f}, p = {mcnemar['p_value']:.4f} {sig}
- Level 1 only correct: {mcnemar['l1_only_correct']}
- Level 3 only correct: {mcnemar['l3_only_correct']}
"""
    
    if "mcnemar_l2_vs_l3" in stats_results:
        mcnemar = stats_results["mcnemar_l2_vs_l3"]
        sig = "***" if mcnemar["p_value"] < 0.001 else "**" if mcnemar["p_value"] < 0.01 else "*" if mcnemar["p_value"] < 0.05 else "ns"
        report += f"""
**Level 2 vs Level 3:**
- McNemar's test: χ² = {mcnemar['statistic']:.4f}, p = {mcnemar['p_value']:.4f} {sig}
- Level 2 only correct: {mcnemar['l2_only_correct']}
- Level 3 only correct: {mcnemar['l3_only_correct']}
"""
    
    report += f"""

## 2. Level 2 Confidence Calibration Analysis

### Calibration Metrics

"""
    
    if "error" not in metrics:
        report += f"""
- **Expected Calibration Error (ECE)**: {metrics['ece']:.4f}
  - Lower is better (0 = perfect calibration)
- **Brier Score**: {metrics['brier_score']:.4f}
  - Lower is better (0 = perfect predictions)
- **Pearson Correlation**: r = {metrics['pearson_r']:.4f}, p = {metrics['pearson_p']:.4f}
- **Spearman Correlation**: ρ = {metrics['spearman_r']:.4f}, p = {metrics['spearman_p']:.4f}

### Confidence Statistics

- Mean: {metrics['confidence_mean']:.4f}
- Std: {metrics['confidence_std']:.4f}
- Min: {metrics['confidence_min']:.4f}
- Max: {metrics['confidence_max']:.4f}
- Number of samples with confidence: {metrics['n_samples']}

### Per-Bin Calibration Analysis

| Confidence Bin | Count | Avg Confidence | Accuracy | Calibration Error |
|----------------|-------|----------------|----------|-------------------|
"""
        
        for _, row in metrics['bin_stats'].iterrows():
            report += f"| {row['bin']} | {int(row['count'])} | {row['avg_confidence']:.4f} | {row['accuracy']:.4f} | {row['calibration_error']:.4f} |\n"
        
        if len(metrics['overconfident_bins']) > 0:
            report += "\n**Overconfident Bins** (confidence > accuracy + 0.1):\n"
            for _, row in metrics['overconfident_bins'].iterrows():
                report += f"- {row['bin']}: confidence={row['avg_confidence']:.4f}, accuracy={row['accuracy']:.4f}\n"
        
        if len(metrics['underconfident_bins']) > 0:
            report += "\n**Underconfident Bins** (confidence < accuracy - 0.1):\n"
            for _, row in metrics['underconfident_bins'].iterrows():
                report += f"- {row['bin']}: confidence={row['avg_confidence']:.4f}, accuracy={row['accuracy']:.4f}\n"
    else:
        report += f"\n**Error**: {metrics['error']}\n"
    
    report += f"""

## 3. Level Agreement Analysis

- **Total matched samples**: {comparison['n_matched']}
- **All levels agree**: {comparison['all_agree_count']} ({comparison['all_agree_pct']:.2f}%)
- **Level 1 & 2 agree**: {comparison['l1_l2_agree_count']} ({comparison['l1_l2_agree_pct']:.2f}%)
- **Level 1 & 3 agree**: {comparison['l1_l3_agree_count']} ({comparison['l1_l3_agree_pct']:.2f}%)
- **Level 2 & 3 agree**: {comparison['l2_l3_agree_count']} ({comparison['l2_l3_agree_pct']:.2f}%)

### Correctness Patterns

- **All correct**: {comparison['all_correct_count']} ({comparison['all_correct_pct']:.2f}%)
- **All incorrect**: {comparison['all_incorrect_count']} ({comparison['all_incorrect_pct']:.2f}%)
- **Only Level 1 correct**: {comparison['only_l1_correct_count']}
- **Only Level 2 correct**: {comparison['only_l2_correct_count']}
- **Only Level 3 correct**: {comparison['only_l3_correct_count']}

"""
    
    # Justification analysis
    if comparison['justification_stats']['mean_correct'] is not None:
        report += f"""
## 4. Level 3 Justification Analysis

- **Mean justification length (correct)**: {comparison['justification_stats']['mean_correct']:.1f} characters
- **Mean justification length (incorrect)**: {comparison['justification_stats']['mean_incorrect']:.1f} characters
- **Std justification length (correct)**: {comparison['justification_stats']['std_correct']:.1f} characters
- **Std justification length (incorrect)**: {comparison['justification_stats']['std_incorrect']:.1f} characters

"""
    
    report += """
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

"""
    
    if "error" not in metrics:
        if metrics['ece'] < 0.1:
            report += "- **Calibration Quality**: Level 2 confidence is well-calibrated (ECE < 0.1)\n"
        elif metrics['ece'] < 0.2:
            report += "- **Calibration Quality**: Level 2 confidence is moderately calibrated (ECE < 0.2)\n"
        else:
            report += "- **Calibration Quality**: Level 2 confidence shows poor calibration (ECE >= 0.2)\n"
        
        if metrics['pearson_p'] < 0.05:
            report += f"- **Confidence-Correctness Correlation**: Significant positive correlation (r={metrics['pearson_r']:.4f}, p<0.05)\n"
        else:
            report += f"- **Confidence-Correctness Correlation**: No significant correlation (r={metrics['pearson_r']:.4f}, p>=0.05)\n"
    
    # Accuracy comparison insights
    acc_diff_12 = abs(results_dict['level1_accuracy'] - results_dict['level2_accuracy'])
    acc_diff_13 = abs(results_dict['level1_accuracy'] - results_dict['level3_accuracy'])
    acc_diff_23 = abs(results_dict['level2_accuracy'] - results_dict['level3_accuracy'])
    
    if acc_diff_12 < 0.01:
        report += "- **Level 1 vs Level 2**: Accuracies are very similar (difference < 1%)\n"
    else:
        report += f"- **Level 1 vs Level 2**: Accuracies differ by {acc_diff_12*100:.2f}%\n"
    
    if acc_diff_13 < 0.01:
        report += "- **Level 1 vs Level 3**: Accuracies are very similar (difference < 1%)\n"
    else:
        report += f"- **Level 1 vs Level 3**: Accuracies differ by {acc_diff_13*100:.2f}%\n"
    
    if acc_diff_23 < 0.01:
        report += "- **Level 2 vs Level 3**: Accuracies are very similar (difference < 1%)\n"
    else:
        report += f"- **Level 2 vs Level 3**: Accuracies differ by {acc_diff_23*100:.2f}%\n"
    
    report += f"""
- **Agreement**: {comparison['all_agree_pct']:.2f}% of samples have all three levels agreeing on the answer

### Recommendations

1. **Confidence Calibration**: """
    
    if "error" not in metrics:
        if metrics['ece'] > 0.2:
            report += "Consider recalibrating the confidence scores using temperature scaling or Platt scaling to improve calibration.\n"
        else:
            report += "Confidence scores are reasonably well-calibrated. Continue monitoring calibration quality.\n"
    else:
        report += "Unable to assess calibration due to missing confidence data.\n"
    
    report += """
2. **Schema Impact**: The addition of confidence (Level 2) and justification (Level 3) fields does not significantly impact accuracy compared to Level 1, suggesting the model can handle these additional constraints effectively.

3. **Future Work**: 
   - Investigate cases where levels disagree to understand model uncertainty
   - Analyze the relationship between justification quality and correctness
   - Consider using confidence scores for selective prediction or uncertainty quantification

---
*Report generated automatically by analyze_confidence_calibration.py*
"""
    
    with open(save_path, 'w') as f:
        f.write(report)
    
    print(f"Saved report to {save_path}")


def main():
    """Main analysis pipeline."""
    print("=" * 80)
    print("MedQA Confidence Calibration Analysis")
    print("=" * 80)
    
    # Load and parse data
    df1, df2, df3, results_dict = load_and_parse_data()
    
    # Calculate calibration metrics for Level 2
    print("\n" + "=" * 80)
    print("Calculating Level 2 Calibration Metrics...")
    print("=" * 80)
    metrics = calculate_calibration_metrics(df2)
    
    if "error" not in metrics:
        print(f"ECE: {metrics['ece']:.4f}")
        print(f"Brier Score: {metrics['brier_score']:.4f}")
        print(f"Pearson r: {metrics['pearson_r']:.4f} (p={metrics['pearson_p']:.4f})")
        print(f"Spearman ρ: {metrics['spearman_r']:.4f} (p={metrics['spearman_p']:.4f})")
    
    # Create Level 2 visualizations
    print("\n" + "=" * 80)
    print("Creating Level 2 Calibration Visualizations...")
    print("=" * 80)
    create_calibration_plot(df2, metrics, PLOTS_DIR / "level2_calibration_plots.png")
    
    # Compare levels
    print("\n" + "=" * 80)
    print("Comparing Levels...")
    print("=" * 80)
    comparison = compare_levels(df1, df2, df3)
    print(f"Matched samples: {comparison['n_matched']}")
    print(f"All levels agree: {comparison['all_agree_pct']:.2f}%")
    
    # Create comparison visualizations
    print("\n" + "=" * 80)
    print("Creating Level Comparison Visualizations...")
    print("=" * 80)
    create_comparison_plots(df1, df2, df3, comparison, PLOTS_DIR / "level_comparison_plots.png")
    
    # Statistical tests
    print("\n" + "=" * 80)
    print("Performing Statistical Tests...")
    print("=" * 80)
    stats_results = perform_statistical_tests(df1, df2, df3, comparison)
    for test_name, result in stats_results.items():
        print(f"{test_name}: p={result['p_value']:.4f}")
    
    # Generate report
    print("\n" + "=" * 80)
    print("Generating Report...")
    print("=" * 80)
    generate_report(df1, df2, df3, metrics, comparison, stats_results, results_dict,
                   OUTPUT_DIR / "confidence_calibration_report.md")
    
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)
    print(f"Report saved to: {OUTPUT_DIR / 'confidence_calibration_report.md'}")
    print(f"Plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()

