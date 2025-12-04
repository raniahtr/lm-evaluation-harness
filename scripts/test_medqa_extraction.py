#!/usr/bin/env python3
"""
Test script for analyzing MedQA generation extraction failures.

This script loads sample outputs from JSONL files, applies regex extraction,
and generates a detailed failure report.
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional

# Note: We implement regex testing directly to avoid dependency issues


def load_samples(jsonl_path: str) -> List[Dict]:
    """Load samples from JSONL file."""
    samples = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def test_regex_pattern(text: str, pattern: str) -> Tuple[Optional[str], List[str]]:
    """
    Test a regex pattern on text and return extracted value and all matches.
    
    Returns:
        (extracted_value, all_matches)
    """
    regex = re.compile(pattern)
    matches = regex.findall(text)
    
    if not matches:
        return None, []
    
    # Handle tuple matches (multiple capture groups)
    first_match = matches[0]
    if isinstance(first_match, tuple):
        # Get first non-empty group
        extracted = [m for m in first_match if m]
        if extracted:
            extracted = extracted[0].strip()
        else:
            extracted = None
    else:
        extracted = first_match.strip() if first_match else None
    
    return extracted, matches


def analyze_extraction_failures(samples: List[Dict], regex_pattern: str) -> Dict:
    """
    Analyze extraction failures and categorize them.
    
    Returns:
        Dictionary with failure analysis statistics
    """
    stats = {
        'total': len(samples),
        'correct_extraction': 0,
        'wrong_extraction': 0,
        'no_match': 0,
        'invalid_fallback': 0,
        'correct_but_wrong_answer': 0,  # Extraction works but model gave wrong answer
        'extraction_failures': [],  # List of failure examples
        'output_formats': Counter(),  # Count different output formats
        'failure_modes': Counter(),  # Categorize failure modes
    }
    
    for sample in samples:
        target = sample.get('target', '')
        filtered_resp = sample.get('filtered_resps', [''])[0] if sample.get('filtered_resps') else ''
        raw_resp = sample.get('resps', [['']])[0][0] if sample.get('resps') else ''
        
        # Test regex extraction
        extracted, all_matches = test_regex_pattern(raw_resp, regex_pattern)
        
        # Categorize output format
        if not raw_resp or raw_resp.strip() == '':
            format_type = 'empty'
        elif 'boxed{' in raw_resp.lower():
            format_type = 'boxed_format'
        elif re.search(r'answer\s*[:\-]?\s*[A-D]', raw_resp, re.I):
            format_type = 'answer_colon'
        elif re.search(r'the\s+answer\s+is\s+[A-D]', raw_resp, re.I):
            format_type = 'answer_is'
        elif re.search(r'\b([A-D])\b', raw_resp):
            format_type = 'standalone_letter'
        else:
            format_type = 'other'
        
        stats['output_formats'][format_type] += 1
        
        # Analyze extraction result
        if filtered_resp == '[invalid]':
            stats['invalid_fallback'] += 1
            stats['failure_modes']['invalid_fallback'] += 1
            stats['extraction_failures'].append({
                'doc_id': sample.get('doc_id'),
                'target': target,
                'raw_output': raw_resp[:200],  # Truncate for readability
                'extracted': extracted,
                'all_matches': all_matches[:5],  # Limit matches shown
                'format_type': format_type,
                'failure_reason': 'no_match_found'
            })
        elif extracted is None:
            stats['no_match'] += 1
            stats['failure_modes']['no_match'] += 1
        elif filtered_resp == target:
            stats['correct_extraction'] += 1
            if sample.get('exact_match', 0) == 0:
                # Extraction correct but model gave wrong answer
                stats['correct_but_wrong_answer'] += 1
        else:
            stats['wrong_extraction'] += 1
            stats['failure_modes']['wrong_extraction'] += 1
            stats['extraction_failures'].append({
                'doc_id': sample.get('doc_id'),
                'target': target,
                'extracted': filtered_resp,
                'raw_output': raw_resp[:200],
                'format_type': format_type,
                'failure_reason': 'wrong_letter_extracted'
            })
    
    return stats


def test_alternative_patterns(samples: List[Dict]) -> Dict[str, Dict]:
    """
    Test alternative regex patterns on the samples.
    
    Returns:
        Dictionary mapping pattern names to their statistics
    """
    patterns = {
        'current': r'(?i)answer\W(?:is)*\W*([A-D])(?:\W|$)|(?:answer|boxed)?{\W*([A-D])\W+',
        'simple_letter': r'\b([A-D])\b',
        'answer_colon': r'(?i)answer\s*[:\-]?\s*([A-D])(?:\W|$)',
        'boxed_only': r'(?i)boxed\s*{\s*([A-D])\s*}',
        'answer_is': r'(?i)(?:the\s+)?answer\s+is\s+([A-D])(?:\W|$)',
        'flexible': r'(?i)(?:answer|boxed|final)\s*(?:is|:)?\s*[:\-]?\s*[{\[]?\s*([A-D])\s*[}\]]?',
    }
    
    results = {}
    for pattern_name, pattern in patterns.items():
        stats = analyze_extraction_failures(samples, pattern)
        results[pattern_name] = {
            'pattern': pattern,
            'stats': stats
        }
    
    return results


def generate_report(samples: List[Dict], regex_pattern: str, output_path: Optional[str] = None):
    """Generate a detailed failure analysis report."""
    
    # Analyze with current pattern
    current_stats = analyze_extraction_failures(samples, regex_pattern)
    
    # Test alternative patterns
    alt_results = test_alternative_patterns(samples)
    
    # Generate report
    report_lines = [
        "=" * 80,
        "MedQA Generation Extraction Failure Analysis",
        "=" * 80,
        "",
        f"Total samples analyzed: {current_stats['total']}",
        "",
        "--- Current Regex Pattern Statistics ---",
        f"Pattern: {regex_pattern}",
        f"Correct extractions: {current_stats['correct_extraction']} ({100*current_stats['correct_extraction']/current_stats['total']:.1f}%)",
        f"Wrong extractions: {current_stats['wrong_extraction']} ({100*current_stats['wrong_extraction']/current_stats['total']:.1f}%)",
        f"No match found: {current_stats['no_match']} ({100*current_stats['no_match']/current_stats['total']:.1f}%)",
        f"Invalid fallback: {current_stats['invalid_fallback']} ({100*current_stats['invalid_fallback']/current_stats['total']:.1f}%)",
        f"Correct extraction but wrong answer: {current_stats['correct_but_wrong_answer']}",
        "",
        "--- Output Format Distribution ---",
    ]
    
    for format_type, count in current_stats['output_formats'].most_common():
        report_lines.append(f"  {format_type}: {count} ({100*count/current_stats['total']:.1f}%)")
    
    report_lines.extend([
        "",
        "--- Failure Mode Distribution ---",
    ])
    
    for mode, count in current_stats['failure_modes'].most_common():
        report_lines.append(f"  {mode}: {count} ({100*count/current_stats['total']:.1f}%)")
    
    report_lines.extend([
        "",
        "--- Alternative Pattern Comparison ---",
    ])
    
    for pattern_name, result in alt_results.items():
        stats = result['stats']
        success_rate = 100 * (stats['correct_extraction'] + stats['correct_but_wrong_answer']) / stats['total']
        report_lines.append(f"\n{pattern_name}:")
        report_lines.append(f"  Pattern: {result['pattern']}")
        report_lines.append(f"  Success rate: {success_rate:.1f}%")
        report_lines.append(f"  Invalid fallback: {stats['invalid_fallback']} ({100*stats['invalid_fallback']/stats['total']:.1f}%)")
    
    report_lines.extend([
        "",
        "--- Sample Failure Cases (First 10) ---",
    ])
    
    for i, failure in enumerate(current_stats['extraction_failures'][:10], 1):
        report_lines.extend([
            f"\nFailure {i}:",
            f"  Doc ID: {failure['doc_id']}",
            f"  Target: {failure['target']}",
            f"  Extracted: {failure.get('extracted', 'None')}",
            f"  Format: {failure['format_type']}",
            f"  Reason: {failure['failure_reason']}",
            f"  Raw output (first 200 chars): {failure['raw_output']}",
        ])
    
    report = "\n".join(report_lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"Report saved to {output_path}")
    else:
        print(report)
    
    return report, current_stats, alt_results


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze MedQA extraction failures')
    parser.add_argument('--samples', type=str, 
                       default='results/medqa/samples_medqa_4options_generation_baseline.jsonl',
                       help='Path to samples JSONL file')
    parser.add_argument('--regex', type=str,
                       default=r'(?i)answer\W(?:is)*\W*([A-D])(?:\W|$)|(?:answer|boxed)?{\W*([A-D])\W+',
                       help='Regex pattern to test')
    parser.add_argument('--output', type=str,
                       help='Output path for report (optional)')
    
    args = parser.parse_args()
    
    # Load samples
    print(f"Loading samples from {args.samples}...")
    samples = load_samples(args.samples)
    print(f"Loaded {len(samples)} samples")
    
    # Generate report
    generate_report(samples, args.regex, args.output)


if __name__ == '__main__':
    main()

