"""
Analysis utilities for global benchmark analysis.

This module contains reusable functions for data loading, parsing, metrics computation,
and statistical analysis extracted from global_analysis.ipynb.
"""

import json
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from sklearn.calibration import calibration_curve


# ============================================================================
# File Mappings Configuration
# ============================================================================

PUBMEDQA_RESULT_FILES = {
    'baseline': 'pubmedqa_baseline_2025-12-02T17-10-15.514633.json',
    'L1': 'pubmedqa_level1_2025-12-08T14-28-16.793274.json',
    'L2': 'pubmedqa_level2_2025-12-08T14-33-38.031833.json',
    'L3': 'pubmedqa_level3_2025-12-08T16-37-05.472225.json',
    'L3_inverted': 'pubmedqa_level3_inverted_2025-12-12T13-14-54.900999.json',
    'L4': 'pubmedqa_level4_2025-12-08T17-53-41.436382.json',
    'L5': 'pubmedqa_level5_2025-12-08T19-23-52.009458.json',
    'L6': 'pubmedqa_level6_2025-12-08T19-45-46.527366.json',
}

PUBMEDQA_SAMPLE_FILES = {
    'baseline': 'samples_pubmedqa_generation_baseline.jsonl',
    'L1': 'samples_pubmedqa_generation_L1.jsonl',
    'L2': 'samples_pubmedqa_generation_L2.jsonl',
    'L3': 'samples_pubmedqa_generation_L3.jsonl',
    'L3_inverted': 'samples_pubmedqa_generation_level3Inverted.jsonl',
    'L4': 'samples_pubmedqa_generation_L4.jsonl',
    'L5': 'samples_pubmedqa_generation_L5.jsonl',
    'L6': 'samples_pubmedqa_generation_L6.jsonl',
}

MEDQA_RESULT_FILES = {
    'baseline': 'medqa4option_BASELINE.json',
    'L1': 'medqa4options_LEVEL1.json',
    'L2': 'medqa4options_LEVEL2.json',
    'L3': 'medqa4options_LEVEL3.json',
    'L3_inverted': 'medqa_inverted_cot_2025-12-12T15-31-19.422661.json',
    'L4': 'medqa4options_LEVEL4.json',
    'L5': 'medqa4options_LEVEL5.json',
    'L6': 'medqa4options_LEVEL6.json',
}

MEDQA_SAMPLE_FILES = {
    'baseline': 'samples_BASELINE.jsonl',
    'L1': 'samples_LEVEL1.jsonl',
    'L2': 'samples_LEVEL2.jsonl',
    'L3': 'samples_LEVEL3.jsonl',
    'L3_inverted': 'samples_INVERTED.jsonl',
    'L4': 'samples_LEVEL4.jsonl',
    'L5': 'samples_LEVEL5.jsonl',
    'L6': 'samples_LEVEL6.jsonl',
}

MEDMCQA_RESULT_FILES = {
    'baseline': 'False_medmcqa_generation_baseline_2025-12-11T08-21-06.604776.json',
    'L1': 'medmcqa_level1_2025-12-08T18-49-34.939793.json',
    'L2': 'medmcqa_level2_2025-12-08T22-15-47.472224.json',
    'L3': 'medmcqa_level3_2025-12-10T02-51-09.353135.json',
    'L3_inverted': 'medmcqa_MCQInvertedCoTAnswer_2025-12-12T19-12-02.556477.json',
    'L4': 'medmcqa_level4_2025-12-10T17-18-56.377289.json',
    'L5': 'medmcqa_level5_2025-12-11T08-06-31.565470.json',
    'L6': 'medmcqa_level6_2025-12-11T21-30-51.473619.json',
}

MEDMCQA_SAMPLE_FILES = {
    'baseline': 'samples_medmcqa_generation_2025-12-11T08-21-06.604776.jsonl',
    'L1': 'samples_medmcqa_generation_2025-12-08T18-49-34.939793.jsonl',
    'L2': 'samples_medmcqa_generation_2025-12-08T22-15-47.472224.jsonl',
    'L3': 'samples_medmcqa_generation_2025-12-10T02-51-09.353135.jsonl',
    'L3_inverted': 'samples_medmcqa_generation_2025-12-12T19-12-02.556477.jsonl',
    'L4': 'samples_medmcqa_generation_2025-12-10T17-18-56.377289.jsonl',
    'L5': 'samples_medmcqa_generation_2025-12-11T08-06-31.565470.jsonl',
    'L6': 'samples_medmcqa_generation_2025-12-11T21-30-51.473619.jsonl',
}


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_results(result_file: str) -> Dict[str, Any]:
    """Load result JSON file and extract metrics."""
    with open(result_file, 'r') as f:
        data = json.load(f)
    return data


def load_samples(samples_file: str) -> List[Dict[str, Any]]:
    """Load sample JSONL file."""
    samples = []
    with open(samples_file, 'r') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def extract_json_from_text(text: str) -> Optional[Dict]:
    """Extract JSON object from text, handling incomplete JSON."""
    if not text or not isinstance(text, str):
        return None
    
    text_cleaned = text.strip()
    
    # Try direct parse first
    try:
        return json.loads(text_cleaned)
    except (json.JSONDecodeError, TypeError):
        pass
    
    # Try to find complete JSON object using balanced braces
    start_idx = text_cleaned.find('{')
    if start_idx == -1:
        return None
    
    brace_count = 0
    for i in range(start_idx, len(text_cleaned)):
        if text_cleaned[i] == '{':
            brace_count += 1
        elif text_cleaned[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                json_str = text_cleaned[start_idx:i+1]
                try:
                    return json.loads(json_str)
                except (json.JSONDecodeError, TypeError):
                    return None
    return None


def get_file_mappings(benchmark: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Get file mappings for a given benchmark.
    
    Args:
        benchmark: Benchmark name ('pubmedqa', 'medqa', or 'medmcqa')
        
    Returns:
        Tuple of (result_files_dict, sample_files_dict)
    """
    mappings = {
        'pubmedqa': (PUBMEDQA_RESULT_FILES, PUBMEDQA_SAMPLE_FILES),
        'medqa': (MEDQA_RESULT_FILES, MEDQA_SAMPLE_FILES),
        'medmcqa': (MEDMCQA_RESULT_FILES, MEDMCQA_SAMPLE_FILES),
    }
    return mappings.get(benchmark, ({}, {}))


def load_all_benchmark_data(base_dir: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, List[Dict]]]]:
    """Load all benchmark data (results and samples).
    
    Args:
        base_dir: Base directory containing benchmark subdirectories
        
    Returns:
        Tuple of (all_results, all_samples) dictionaries
    """
    all_results = {}
    all_samples = {}
    
    benchmarks = [
        ("pubmedqa", base_dir / "pubmedqa", "Loading PubMedQA..."),
        ("medqa", base_dir / "medqa", "Loading MedQA..."),
        ("medmcqa", base_dir / "medmcqa", "Loading MedMCQA..."),
    ]
    
    for key, bench_dir, header in benchmarks:
        print(header)
        
        result_files, sample_files = get_file_mappings(key)
        
        all_results[key] = {}
        all_samples[key] = {}
        
        # Load results
        for level, filename in result_files.items():
            filepath = bench_dir / filename
            if filepath.exists():
                all_results[key][level] = load_results(str(filepath))
                print(f"   {level}: {filename}")
            else:
                print(f"   Warning: {filename} not found")
        
        # Load samples
        for level, filename in sample_files.items():
            filepath = bench_dir / filename
            if filepath.exists():
                all_samples[key][level] = load_samples(str(filepath))
                print(f"   {level} samples: {len(all_samples[key][level])} samples")
            else:
                print(f"   Warning: {filename} not found")
    
    print("\nData loading complete")
    return all_results, all_samples


# ============================================================================
# Parsing Functions
# ============================================================================

def parse_sample(sample: Dict, level: str, benchmark: str) -> Dict[str, Any]:
    """Parse a sample based on schema level with comprehensive extraction.
    
    Args:
        sample: Sample dictionary from JSONL
        level: Schema level (baseline, L1-L6)
        benchmark: Benchmark name (pubmedqa, medqa, medmcqa)
        
    Returns:
        Parsed sample dictionary with extracted fields
    """
    raw_resp = sample.get('resps', [[None]])[0][0] if sample.get('resps') else None
    filtered_resp = sample.get('filtered_resps', ['[invalid]'])[0]
    target = sample.get('target', '')
    exact_match = sample.get('exact_match', 0)
    doc_id = sample.get('doc_id', -1)
    
    # Determine valid answers based on benchmark
    if benchmark == 'pubmedqa':
        valid_answers = ['yes', 'no', 'maybe']
        answer_normalize = lambda x: x.lower().strip() if x else '[invalid]'
    else:  # medqa, medmcqa
        valid_answers = ['A', 'B', 'C', 'D']
        answer_normalize = lambda x: x.upper().strip() if x else '[invalid]'
    
    parsed_data = {
        'benchmark': benchmark,
        'level': level,
        'doc_id': doc_id,
        'raw_response': raw_resp if raw_resp else '',
        'filtered_response': filtered_resp if filtered_resp else '[invalid]',
        'target': target,
        'exact_match': exact_match,
        'json_valid': False,
        'answer': None,
        'confidence': None,
        'probabilities': None,
        'reasoning': None,
        'justification': None,
        'eliminated': None,
        'key_evidence': None,
        'key_concepts': None,
        'has_answer': False,
        'has_confidence': False,
        'has_probabilities': False,
        'has_reasoning': False,
        'has_justification': False,
        'has_eliminated': False,
        'has_key_evidence': False,
        'has_key_concepts': False,
        'confidence_valid': False,
        'schema_compliant': False,
    }
    
    # Baseline: extract from free-form text
    if level == 'baseline':
        answer = answer_normalize(filtered_resp)
        if answer in valid_answers:
            parsed_data['answer'] = answer
            parsed_data['has_answer'] = True
        else:
            parsed_data['answer'] = '[invalid]'
        parsed_data['schema_compliant'] = parsed_data['has_answer']
        return parsed_data
    
    # Schema levels: parse JSON
    if raw_resp and isinstance(raw_resp, str):
        parsed_json = extract_json_from_text(raw_resp)
        if parsed_json and isinstance(parsed_json, dict):
            parsed_data['json_valid'] = True
            answer = answer_normalize(parsed_json.get('answer', ''))
            parsed_data['answer'] = answer
            parsed_data['has_answer'] = answer in valid_answers
            
            # Extract confidence
            if 'confidence' in parsed_json:
                try:
                    conf = float(parsed_json['confidence'])
                    parsed_data['confidence'] = conf
                    parsed_data['has_confidence'] = True
                    parsed_data['confidence_valid'] = (0.0 <= conf <= 1.0)
                except (ValueError, TypeError):
                    pass
            
            # Extract reasoning/justification
            parsed_data['reasoning'] = parsed_json.get('reasoning', '')
            parsed_data['justification'] = parsed_json.get('justification', '')
            parsed_data['has_reasoning'] = bool(parsed_data['reasoning'])
            parsed_data['has_justification'] = bool(parsed_data['justification'])
            
            # Extract eliminated options (Level 5)
            parsed_data['eliminated'] = parsed_json.get('eliminated', [])
            parsed_data['has_eliminated'] = isinstance(parsed_data['eliminated'], list) and len(parsed_data['eliminated']) > 0
            
            # Extract key evidence (Level 5)
            parsed_data['key_evidence'] = parsed_json.get('key_evidence', '')
            parsed_data['has_key_evidence'] = bool(parsed_data['key_evidence'])
            
            # Extract key concepts (Level 6)
            parsed_data['key_concepts'] = parsed_json.get('key_concepts', [])
            parsed_data['has_key_concepts'] = isinstance(parsed_data['key_concepts'], list) and len(parsed_data['key_concepts']) > 0
            
            # Check schema compliance based on level
            parsed_data['schema_compliant'] = _check_schema_compliance(
                parsed_data, level, benchmark
            )
        else:
            # Fallback to filtered response
            answer = answer_normalize(filtered_resp)
            if answer in valid_answers:
                parsed_data['answer'] = answer
                parsed_data['has_answer'] = True
            else:
                parsed_data['answer'] = '[invalid]'
    
    return parsed_data


def _check_schema_compliance(parsed_data: Dict, level: str, benchmark: str) -> bool:
    """Check if parsed data is schema compliant for given level and benchmark."""
    if level == 'L1':
        return parsed_data['json_valid'] and parsed_data['has_answer']
    elif level == 'L2':
        if benchmark == 'pubmedqa':
            return (parsed_data['json_valid'] and parsed_data['has_answer'] 
                   and parsed_data['has_confidence'] and parsed_data['confidence_valid'])
        else:  # medqa, medmcqa
            return (parsed_data['json_valid'] and parsed_data['has_answer'] 
                   and parsed_data['has_confidence'] and parsed_data['confidence_valid'])
    elif level == 'L3':
        if benchmark == 'pubmedqa':
            return (parsed_data['json_valid'] and parsed_data['has_answer'] 
                   and parsed_data['has_reasoning'])
        else:  # medqa, medmcqa
            return (parsed_data['json_valid'] and parsed_data['has_answer'] 
                   and parsed_data['has_justification'])
    elif level == 'L3_inverted':
        return (parsed_data['json_valid'] and parsed_data['has_answer'] 
               and parsed_data['has_reasoning'])
    elif level == 'L4':
        return (parsed_data['json_valid'] and parsed_data['has_answer'] 
               and parsed_data['has_reasoning'] and parsed_data['has_confidence'] 
               and parsed_data['confidence_valid'])
    elif level == 'L5':
        if benchmark == 'pubmedqa':
            return (parsed_data['json_valid'] and parsed_data['has_answer'] 
                   and parsed_data['has_confidence'] and parsed_data['confidence_valid']
                   and parsed_data['has_reasoning'] and parsed_data['has_key_evidence'])
        else:  # medqa, medmcqa
            return (parsed_data['json_valid'] and parsed_data['has_answer'] 
                   and parsed_data['has_eliminated'] and parsed_data['has_key_evidence'])
    elif level == 'L6':
        if benchmark == 'pubmedqa':
            return (parsed_data['json_valid'] and parsed_data['has_answer'])
        else:  # medqa, medmcqa
            return (parsed_data['json_valid'] and parsed_data['has_answer'] 
                   and parsed_data['has_reasoning'] and parsed_data['has_confidence'] 
                   and parsed_data['confidence_valid'] and parsed_data['has_key_concepts'])
    return False


def parse_all_samples(all_samples: Dict[str, Dict[str, List[Dict]]]) -> Dict[str, Dict[str, List[Dict]]]:
    """Parse all samples for all benchmarks and levels.
    
    Args:
        all_samples: Dictionary of benchmark -> level -> list of samples
        
    Returns:
        Dictionary of benchmark -> level -> list of parsed samples
    """
    all_parsed = {}
    for benchmark in ['pubmedqa', 'medqa', 'medmcqa']:
        all_parsed[benchmark] = {}
        for level in all_samples[benchmark].keys():
            all_parsed[benchmark][level] = [
                parse_sample(s, level, benchmark) for s in all_samples[benchmark][level]
            ]
            print(f"✓ Parsed {benchmark} {level}: {len(all_parsed[benchmark][level])} samples")
    
    print("\n All samples parsed")
    return all_parsed


# ============================================================================
# Metrics Computation Functions
# ============================================================================

def compute_accuracy_metrics(all_results: Dict, all_parsed: Dict) -> pd.DataFrame:
    """Extract accuracy metrics from results and parsed samples.
    
    Args:
        all_results: Dictionary of benchmark -> level -> results
        all_parsed: Dictionary of benchmark -> level -> list of parsed samples
        
    Returns:
        DataFrame with accuracy metrics
    """
    accuracy_data = []
    
    for benchmark in ['pubmedqa', 'medqa', 'medmcqa']:
        for level in all_parsed[benchmark].keys():
            # Get accuracy from results file if available
            if level in all_results[benchmark]:
                result = all_results[benchmark][level]
                # Extract task name (varies by benchmark)
                task_key = None
                for key in result.get('results', {}).keys():
                    if 'generation' in key or 'pubmedqa' in key or 'medqa' in key or 'medmcqa' in key:
                        task_key = key
                        break
                
                if task_key:
                    exact_match = result['results'][task_key].get('exact_match,strict-match', 0)
                    exact_match_stderr = result['results'][task_key].get('exact_match_stderr,strict-match', 0)
                else:
                    exact_match = 0
                    exact_match_stderr = 0
            else:
                # Calculate from parsed samples
                parsed = all_parsed[benchmark][level]
                exact_match = np.mean([p['exact_match'] for p in parsed])
                exact_match_stderr = np.std([p['exact_match'] for p in parsed]) / np.sqrt(len(parsed))
            
            # Also calculate from parsed samples for consistency
            parsed = all_parsed[benchmark][level]
            accuracy_from_samples = np.mean([p['exact_match'] for p in parsed])
            
            accuracy_data.append({
                'benchmark': benchmark,
                'level': level,
                'accuracy': exact_match,
                'accuracy_stderr': exact_match_stderr,
                'accuracy_from_samples': accuracy_from_samples,
                'n_samples': len(parsed)
            })
    
    return pd.DataFrame(accuracy_data)


def compute_compliance_metrics(all_parsed: Dict) -> pd.DataFrame:
    """Compute compliance metrics for all benchmarks and levels.
    
    Args:
        all_parsed: Dictionary of benchmark -> level -> list of parsed samples
        
    Returns:
        DataFrame with compliance metrics
    """
    compliance_data = []
    
    for benchmark in ['pubmedqa', 'medqa', 'medmcqa']:
        for level in all_parsed[benchmark].keys():
            parsed = all_parsed[benchmark][level]
            
            n_total = len(parsed)
            n_json_valid = sum(1 for p in parsed if p['json_valid'])
            n_schema_compliant = sum(1 for p in parsed if p['schema_compliant'])
            n_has_answer = sum(1 for p in parsed if p['has_answer'])
            n_invalid_answer = sum(1 for p in parsed if p['answer'] == '[invalid]')
            n_has_confidence = sum(1 for p in parsed if p['has_confidence'])
            n_has_reasoning = sum(1 for p in parsed if p['has_reasoning'])
            n_has_justification = sum(1 for p in parsed if p['has_justification'])
            n_has_eliminated = sum(1 for p in parsed if p['has_eliminated'])
            n_has_key_evidence = sum(1 for p in parsed if p['has_key_evidence'])
            n_has_key_concepts = sum(1 for p in parsed if p['has_key_concepts'])
            
            # Compute parseable answer rate
            if level == 'baseline':
                n_parseable = n_has_answer
            else:
                n_parseable = n_json_valid
            
            # Compute conditional schema compliance rate (within valid JSON)
            if level != 'baseline' and n_json_valid > 0:
                schema_compliant_given_valid_json = n_schema_compliant / n_json_valid
            else:
                schema_compliant_given_valid_json = np.nan
            
            # Compute categories for stacked bar plot
            if level == 'baseline':
                n_valid_json_compliant = sum(1 for p in parsed if p['has_answer'] and p['schema_compliant'])
                n_valid_json_non_compliant = sum(1 for p in parsed if p['has_answer'] and not p['schema_compliant'])
                n_invalid_json = sum(1 for p in parsed if not p['has_answer'])
            else:
                n_valid_json_compliant = sum(1 for p in parsed if p['json_valid'] and p['schema_compliant'])
                n_valid_json_non_compliant = sum(1 for p in parsed if p['json_valid'] and not p['schema_compliant'])
                n_invalid_json = n_total - n_json_valid
            
            compliance_data.append({
                'benchmark': benchmark,
                'level': level,
                'n_total': n_total,
                'json_valid_rate': n_json_valid / n_total if n_total > 0 else 0,
                'schema_compliant_rate': n_schema_compliant / n_total if n_total > 0 else 0,
                'parseable_answer_rate': n_parseable / n_total if n_total > 0 else 0,
                'schema_compliant_rate_given_valid_json': schema_compliant_given_valid_json,
                'has_answer_rate': n_has_answer / n_total if n_total > 0 else 0,
                'invalid_answer_rate': n_invalid_answer / n_total if n_total > 0 else 0,
                'has_confidence_rate': n_has_confidence / n_total if n_total > 0 else 0,
                'has_reasoning_rate': n_has_reasoning / n_total if n_total > 0 else 0,
                'has_justification_rate': n_has_justification / n_total if n_total > 0 else 0,
                'has_eliminated_rate': n_has_eliminated / n_total if n_total > 0 else 0,
                'has_key_evidence_rate': n_has_key_evidence / n_total if n_total > 0 else 0,
                'has_key_concepts_rate': n_has_key_concepts / n_total if n_total > 0 else 0,
                'n_valid_json_compliant': n_valid_json_compliant,
                'n_valid_json_non_compliant': n_valid_json_non_compliant,
                'n_invalid_json': n_invalid_json,
                'rate_valid_json_compliant': n_valid_json_compliant / n_total if n_total > 0 else 0,
                'rate_valid_json_non_compliant': n_valid_json_non_compliant / n_total if n_total > 0 else 0,
                'rate_invalid_json': n_invalid_json / n_total if n_total > 0 else 0,
            })
    
    return pd.DataFrame(compliance_data)


# ============================================================================
# Calibration Functions
# ============================================================================

def compute_ece(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE).
    
    Args:
        confidences: Array of confidence scores
        correct: Array of correctness (0 or 1)
        n_bins: Number of bins for calibration
        
    Returns:
        Expected Calibration Error
    """
    if len(confidences) == 0 or len(correct) == 0:
        return np.nan
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = correct[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def compute_calibration_metrics(all_parsed: Dict) -> List[Dict]:
    """Extract confidence and correctness data for calibration analysis.
    
    Args:
        all_parsed: Dictionary of benchmark -> level -> list of parsed samples
        
    Returns:
        List of calibration data dictionaries
    """
    calibration_data = []
    
    for benchmark in ['pubmedqa', 'medqa', 'medmcqa']:
        for level in all_parsed[benchmark].keys():
            parsed = all_parsed[benchmark][level]
            
            # Filter to samples with confidence
            conf_samples = [p for p in parsed if p['has_confidence'] and p['confidence'] is not None]
            
            if len(conf_samples) == 0:
                continue
            
            confidences = np.array([p['confidence'] for p in conf_samples])
            correct = np.array([p['exact_match'] for p in conf_samples])
            
            ece = compute_ece(confidences, correct)
            mean_confidence = np.mean(confidences)
            mean_accuracy = np.mean(correct)
            correlation = np.corrcoef(confidences, correct)[0, 1] if len(confidences) > 1 else np.nan
            
            calibration_data.append({
                'benchmark': benchmark,
                'level': level,
                'n_with_confidence': len(conf_samples),
                'ece': ece,
                'mean_confidence': mean_confidence,
                'mean_accuracy': mean_accuracy,
                'correlation': correlation,
                'confidences': confidences,
                'correct': correct,
            })
    
    return calibration_data


# ============================================================================
# Distribution Distance Functions
# ============================================================================

def compute_kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute KL divergence D(P||Q)."""
    p = np.array(p)
    q = np.array(q)
    # Avoid division by zero
    p = p + 1e-10
    q = q + 1e-10
    p = p / p.sum()
    q = q / q.sum()
    return np.sum(p * np.log(p / q))


def compute_jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    """Compute Jensen-Shannon divergence (symmetric)."""
    p = np.array(p)
    q = np.array(q)
    p = p + 1e-10
    q = q + 1e-10
    p = p / p.sum()
    q = q / q.sum()
    return jensenshannon(p, q)


def compute_total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Compute Total Variation Distance (L1 norm)."""
    p = np.array(p)
    q = np.array(q)
    p = p + 1e-10
    q = q + 1e-10
    p = p / p.sum()
    q = q / q.sum()
    return 0.5 * np.sum(np.abs(p - q))


def compute_wasserstein_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Compute Wasserstein distance (Earth Mover's Distance)."""
    p = np.array(p)
    q = np.array(q)
    p = p + 1e-10
    q = q + 1e-10
    p = p / p.sum()
    q = q / q.sum()
    # For discrete distributions, use cumulative distribution
    return wasserstein_distance(np.arange(len(p)), np.arange(len(q)), p, q)


def compute_chi2_statistic(observed: np.ndarray, expected: np.ndarray) -> float:
    """Compute chi-square statistic."""
    observed = np.array(observed)
    expected = np.array(expected)
    # Avoid division by zero
    observed = observed + 1e-10
    expected = expected + 1e-10
    return np.sum((observed - expected)**2 / expected)


def compute_distribution_metrics(all_parsed: Dict) -> Tuple[List[Dict], pd.DataFrame]:
    """Compute comprehensive distribution metrics for all benchmarks and levels.
    
    Args:
        all_parsed: Dictionary of benchmark -> level -> list of parsed samples
        
    Returns:
        Tuple of (comprehensive_distribution_data list, DataFrame)
    """
    comprehensive_distribution_data = []
    
    for benchmark in ['pubmedqa', 'medqa', 'medmcqa']:
        # Determine valid answers
        if benchmark == 'pubmedqa':
            valid_answers = ['yes', 'no', 'maybe']
        else:
            valid_answers = ['A', 'B', 'C', 'D']
        
        # Get reference distribution (baseline or L1)
        reference_level = 'baseline' if 'baseline' in all_parsed[benchmark].keys() else 'L1'
        if reference_level not in all_parsed[benchmark]:
            continue
        
        ref_parsed = all_parsed[benchmark][reference_level]
        ref_answers = [p['answer'] for p in ref_parsed if p['answer'] != '[invalid]']
        ref_counts = Counter(ref_answers)
        ref_dist = np.array([ref_counts.get(a, 0) for a in valid_answers], dtype=float)
        ref_total = ref_dist.sum()
        ref_dist = ref_dist / ref_total if ref_total > 0 else ref_dist
        
        for level in all_parsed[benchmark].keys():
            parsed = all_parsed[benchmark][level]
            answers = [p['answer'] for p in parsed if p['answer'] != '[invalid]']
            counts = Counter(answers)
            dist = np.array([counts.get(a, 0) for a in valid_answers], dtype=float)
            total = dist.sum()
            dist = dist / total if total > 0 else dist
            
            # Compute all distance metrics from reference
            kl_div = compute_kl_divergence(dist, ref_dist) if total > 0 and ref_total > 0 else np.nan
            js_div = compute_jensen_shannon(dist, ref_dist) if total > 0 and ref_total > 0 else np.nan
            tv_dist = compute_total_variation_distance(dist, ref_dist) if total > 0 and ref_total > 0 else np.nan
            wass_dist = compute_wasserstein_distance(dist, ref_dist) if total > 0 and ref_total > 0 else np.nan
            
            # Chi-square test
            if total > 0 and ref_total > 0:
                observed = dist * len(answers) if len(answers) > 0 else dist
                expected = ref_dist * len(answers) if len(answers) > 0 else ref_dist
                chi2_stat = compute_chi2_statistic(observed, expected)
            else:
                chi2_stat = np.nan
            
            # Store distribution as percentages
            dist_percentages = {a: dist[i] * 100 for i, a in enumerate(valid_answers)}
            counts_dict = {a: counts.get(a, 0) for a in valid_answers}
            
            comprehensive_distribution_data.append({
                'benchmark': benchmark,
                'level': level,
                'distribution': dist_percentages,
                'counts': counts_dict,
                'kl_divergence': kl_div,
                'jensen_shannon': js_div,
                'total_variation': tv_dist,
                'wasserstein': wass_dist,
                'chi2_statistic': chi2_stat,
                'valid_answers': valid_answers,
                'n_samples': len(answers),
            })
    
    comprehensive_dist_df = pd.DataFrame([
        {k: v for k, v in d.items() if k not in ['distribution', 'counts', 'valid_answers']} 
        for d in comprehensive_distribution_data
    ])
    
    return comprehensive_distribution_data, comprehensive_dist_df


# ============================================================================
# Statistical Analysis Functions
# ============================================================================

def compute_statistical_significance(accuracy_df: pd.DataFrame, all_parsed: Dict) -> None:
    """Print statistical significance tests between levels.
    
    Args:
        accuracy_df: DataFrame with accuracy metrics
        all_parsed: Dictionary of benchmark -> level -> list of parsed samples
    """
    print("="*80)
    print("STATISTICAL SIGNIFICANCE TESTS: Accuracy Differences Between Levels")
    print("="*80)
    
    for benchmark in ['pubmedqa', 'medqa', 'medmcqa']:
        print(f"\n{benchmark.upper()}:")
        bench_data = accuracy_df[accuracy_df['benchmark'] == benchmark].copy()
        levels = sorted(bench_data['level'].unique())
        
        # Get sample-level accuracy data for statistical tests
        bench_parsed = {}
        for level in levels:
            parsed = all_parsed[benchmark][level]
            bench_parsed[level] = [p['exact_match'] for p in parsed]
        
        # Compare each level with baseline (or L1 for pubmedqa)
        reference = 'baseline' if 'baseline' in levels else 'L1'
        if reference not in bench_parsed:
            continue
        
        ref_acc = np.array(bench_parsed[reference])
        
        for level in levels:
            if level == reference:
                continue
            level_acc = np.array(bench_parsed[level])
            
            # Perform t-test
            t_stat, p_value = stats.ttest_rel(ref_acc, level_acc) if len(ref_acc) == len(level_acc) else stats.ttest_ind(ref_acc, level_acc)
            
            mean_diff = np.mean(level_acc) - np.mean(ref_acc)
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            
            print(f"  {reference} vs {level}: Δ = {mean_diff*100:+.2f}%, p = {p_value:.4f} {significance}")
    
    print("="*80)


def compare_l3_vs_l3_inverted(compliance_df: pd.DataFrame, accuracy_df: pd.DataFrame, 
                              all_parsed: Dict) -> pd.DataFrame:
    """Compare L3 vs L3_inverted across all benchmarks.
    
    Args:
        compliance_df: DataFrame with compliance metrics
        accuracy_df: DataFrame with accuracy metrics
        all_parsed: Dictionary of benchmark -> level -> list of parsed samples
        
    Returns:
        DataFrame with comparison metrics
    """
    l3_comparison_data = []
    
    for benchmark in ['pubmedqa', 'medqa', 'medmcqa']:
        # Get L3 and L3_inverted data
        l3_data = compliance_df[(compliance_df['benchmark'] == benchmark) & (compliance_df['level'] == 'L3')]
        l3_inv_data = compliance_df[(compliance_df['benchmark'] == benchmark) & (compliance_df['level'] == 'L3_inverted')]
        
        if len(l3_data) > 0 and len(l3_inv_data) > 0:
            l3_row = l3_data.iloc[0]
            l3_inv_row = l3_inv_data.iloc[0]
            
            # Get accuracy data
            l3_acc = accuracy_df[(accuracy_df['benchmark'] == benchmark) & (accuracy_df['level'] == 'L3')]
            l3_inv_acc = accuracy_df[(accuracy_df['benchmark'] == benchmark) & (accuracy_df['level'] == 'L3_inverted')]
            
            l3_accuracy = l3_acc['accuracy'].values[0] * 100 if len(l3_acc) > 0 else np.nan
            l3_inv_accuracy = l3_inv_acc['accuracy'].values[0] * 100 if len(l3_inv_acc) > 0 else np.nan
            
            # Get parsed samples for reasoning length analysis
            l3_parsed = all_parsed[benchmark].get('L3', [])
            l3_inv_parsed = all_parsed[benchmark].get('L3_inverted', [])
            
            # Calculate reasoning/justification lengths
            l3_reasoning_lengths = []
            l3_inv_reasoning_lengths = []
            
            for p in l3_parsed:
                if benchmark == 'pubmedqa':
                    reasoning = p.get('reasoning', '')
                else:  # medqa, medmcqa
                    reasoning = p.get('justification', '')
                if reasoning:
                    l3_reasoning_lengths.append(len(reasoning))
            
            for p in l3_inv_parsed:
                reasoning = p.get('reasoning', '')
                if reasoning:
                    l3_inv_reasoning_lengths.append(len(reasoning))
            
            l3_comparison_data.append({
                'benchmark': benchmark,
                'l3_accuracy': l3_accuracy,
                'l3_inv_accuracy': l3_inv_accuracy,
                'accuracy_diff': l3_inv_accuracy - l3_accuracy,
                'l3_json_valid': l3_row['json_valid_rate'] * 100,
                'l3_inv_json_valid': l3_inv_row['json_valid_rate'] * 100,
                'json_valid_diff': (l3_inv_row['json_valid_rate'] - l3_row['json_valid_rate']) * 100,
                'l3_schema_compliant': l3_row['schema_compliant_rate'] * 100,
                'l3_inv_schema_compliant': l3_inv_row['schema_compliant_rate'] * 100,
                'schema_compliant_diff': (l3_inv_row['schema_compliant_rate'] - l3_row['schema_compliant_rate']) * 100,
                'l3_has_reasoning': l3_row['has_reasoning_rate'] * 100 if 'has_reasoning_rate' in l3_row else l3_row.get('has_justification_rate', 0) * 100,
                'l3_inv_has_reasoning': l3_inv_row['has_reasoning_rate'] * 100,
                'l3_avg_reasoning_length': np.mean(l3_reasoning_lengths) if l3_reasoning_lengths else 0,
                'l3_inv_avg_reasoning_length': np.mean(l3_inv_reasoning_lengths) if l3_inv_reasoning_lengths else 0,
                'reasoning_length_diff': (np.mean(l3_inv_reasoning_lengths) if l3_inv_reasoning_lengths else 0) - 
                                         (np.mean(l3_reasoning_lengths) if l3_reasoning_lengths else 0),
            })
    
    return pd.DataFrame(l3_comparison_data)


def compute_shift_patterns(comprehensive_distribution_data: List[Dict], 
                           all_parsed: Dict) -> pd.DataFrame:
    """Analyze shift patterns - direction and magnitude.
    
    Args:
        comprehensive_distribution_data: List of distribution data dictionaries
        all_parsed: Dictionary of benchmark -> level -> list of parsed samples
        
    Returns:
        DataFrame with shift patterns
    """
    shift_patterns = []
    
    for benchmark in ['pubmedqa', 'medqa', 'medmcqa']:
        # Get reference distribution
        reference_level = 'baseline' if 'baseline' in all_parsed[benchmark].keys() else 'L1'
        ref_item = next((d for d in comprehensive_distribution_data 
                         if d['benchmark'] == benchmark and d['level'] == reference_level), None)
        if not ref_item:
            continue
        
        ref_dist = ref_item['distribution']
        valid_answers = ref_item['valid_answers']
        
        for level in all_parsed[benchmark].keys():
            if level == reference_level:
                continue
            
            level_item = next((d for d in comprehensive_distribution_data 
                              if d['benchmark'] == benchmark and d['level'] == level), None)
            if not level_item:
                continue
            
            level_dist = level_item['distribution']
            
            # Calculate shift for each answer choice
            shifts = {}
            for ans in valid_answers:
                ref_pct = ref_dist.get(ans, 0)
                level_pct = level_dist.get(ans, 0)
                shift = level_pct - ref_pct
                shifts[ans] = shift
            
            # Identify largest increases and decreases
            sorted_shifts = sorted(shifts.items(), key=lambda x: x[1], reverse=True)
            max_increase = sorted_shifts[0] if sorted_shifts else (None, 0)
            max_decrease = sorted_shifts[-1] if sorted_shifts else (None, 0)
            
            # Calculate overall shift magnitude (sum of absolute changes)
            shift_magnitude = sum(abs(s) for s in shifts.values())
            
            shift_patterns.append({
                'benchmark': benchmark,
                'level': level,
                'shifts': shifts,
                'max_increase_choice': max_increase[0],
                'max_increase_value': max_increase[1],
                'max_decrease_choice': max_decrease[0],
                'max_decrease_value': max_decrease[1],
                'shift_magnitude': shift_magnitude,
            })
    
    return pd.DataFrame(shift_patterns)

