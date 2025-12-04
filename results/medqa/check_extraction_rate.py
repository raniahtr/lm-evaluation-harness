#!/usr/bin/env python3
"""
Check invalid extraction rate for medqa generation samples
using the simple regex pattern \b([A-D])\b
"""
import json
import re
from pathlib import Path

# Regex pattern to extract A-D answers
PATTERN = re.compile(r'\b([A-D])\b')

def extract_answer(text):
    """Extract first A-D answer using word boundary regex"""
    match = PATTERN.search(text)
    if match:
        return match.group(1)
    return None

def analyze_samples(filepath):
    """Analyze samples and calculate invalid extraction rate"""
    total_samples = 0
    invalid_extractions = 0
    invalid_samples = []
    
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
                
            try:
                sample = json.loads(line)
                total_samples += 1
                
                # Get the raw response text
                if 'resps' in sample and sample['resps']:
                    raw_response = sample['resps'][0][0] if sample['resps'][0] else ""
                else:
                    raw_response = ""
                
                # Try to extract answer using the regex
                extracted = extract_answer(raw_response)
                
                if extracted is None:
                    invalid_extractions += 1
                    invalid_samples.append({
                        'doc_id': sample.get('doc_id', line_num - 1),
                        'response_preview': raw_response[:200] + '...' if len(raw_response) > 200 else raw_response,
                        'filtered_resp': sample.get('filtered_resps', ['N/A'])[0] if sample.get('filtered_resps') else 'N/A'
                    })
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
    
    invalid_rate = (invalid_extractions / total_samples * 100) if total_samples > 0 else 0
    
    print(f"\n{'='*80}")
    print(f"EXTRACTION ANALYSIS REPORT")
    print(f"{'='*80}")
    print(f"File: {filepath}")
    print(f"Total samples: {total_samples}")
    print(f"Invalid extractions: {invalid_extractions}")
    print(f"Invalid extraction rate: {invalid_rate:.2f}%")
    print(f"Valid extraction rate: {100 - invalid_rate:.2f}%")
    print(f"{'='*80}\n")
    
    # Show some examples of invalid extractions
    if invalid_samples:
        print(f"First 10 examples of invalid extractions:")
        print(f"{'-'*80}")
        for i, sample in enumerate(invalid_samples[:10], 1):
            print(f"\n{i}. Doc ID: {sample['doc_id']}")
            print(f"   Filtered resp (current): {sample['filtered_resp']}")
            print(f"   Response preview: {sample['response_preview']}")
    
    return {
        'total_samples': total_samples,
        'invalid_extractions': invalid_extractions,
        'invalid_rate': invalid_rate,
        'valid_rate': 100 - invalid_rate
    }

if __name__ == "__main__":
    # Analyze the latest baseline file
    filepath = Path("/mloscratch/users/hatrouho/lm-evaluation-harness/results/medqa/samples_medqa_4options_generation_2025-12-04T10-19-01.319316.jsonl")
    
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        exit(1)
    
    results = analyze_samples(filepath)
    
    # Save results to a file
    output_file = filepath.parent / "extraction_rate_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")




