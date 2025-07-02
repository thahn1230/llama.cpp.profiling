#!/usr/bin/env python3
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def parse_latency_file(filename):
    """Parse latency file and extract operation data"""
    operations = []
    
    with open(filename, 'r') as f:
        content = f.read()
    
    # Pattern to match latency entries
    pattern = r'\[\[\[\[Latency for Tensor\]\]\]\] \'([^\']+)\' \(([^)]+)\): (\d+) us'
    matches = re.findall(pattern, content)
    
    for name, op_type, latency_str in matches:
        latency = int(latency_str)
        operations.append({
            'name': name,
            'type': op_type,
            'latency': latency
        })
    
    return operations

def categorize_operation(name, op_type):
    """Categorize operations into main quantized operations or Others"""
    name_lower = name.lower()
    
    # Main quantized operations (MUL_MAT operations are the quantized ones)
    if op_type == 'MUL_MAT':
        if 'qcur' in name_lower:
            return 'Q'
        elif 'kcur' in name_lower:
            return 'K'
        elif 'vcur' in name_lower:
            return 'V'
        elif 'attn_out' in name_lower:
            return 'O'
        elif 'ffn_up' in name_lower:
            return 'Up'
        elif 'ffn_gate' in name_lower:
            return 'Gate'
        elif 'ffn_out' in name_lower:
            return 'Down'
        else:
            return 'Others'  # Other MUL_MAT operations
    else:
        return 'Others'  # All non-MUL_MAT operations

def analyze_others_detailed(filename, label):
    """Analyze Others category in detail"""
    operations = parse_latency_file(filename)
    
    others_ops = []
    others_by_type = defaultdict(list)
    
    for op in operations:
        category = categorize_operation(op['name'], op['type'])
        if category == 'Others':
            others_ops.append(op)
            others_by_type[op['type']].append(op)
    
    print(f"\n{'='*80}")
    print(f"DETAILED ANALYSIS OF 'OTHERS' CATEGORY - {label}")
    print(f"{'='*80}")
    
    # Sort by operation type and then by latency
    print(f"\nOperations by Type:")
    print(f"{'Type':<15} {'Count':<8} {'Total (ms)':<12} {'Avg (ms)':<10} {'Examples'}")
    print("-" * 80)
    
    total_others = 0
    for op_type in sorted(others_by_type.keys()):
        ops = others_by_type[op_type]
        total_latency = sum(op['latency'] for op in ops)
        avg_latency = total_latency / len(ops)
        total_others += total_latency
        
        # Get a few example names
        examples = [op['name'] for op in sorted(ops, key=lambda x: x['latency'], reverse=True)[:3]]
        examples_str = ', '.join(examples[:2]) + ('...' if len(examples) > 2 else '')
        
        print(f"{op_type:<15} {len(ops):<8} {total_latency/1000:<12.1f} {avg_latency/1000:<10.3f} {examples_str}")
    
    print("-" * 80)
    print(f"{'TOTAL':<15} {len(others_ops):<8} {total_others/1000:<12.1f}")
    
    # Show top 10 individual operations by latency
    print(f"\nTop 10 Individual Operations in Others:")
    print(f"{'Rank':<5} {'Name':<40} {'Type':<15} {'Latency (ms)':<12}")
    print("-" * 80)
    
    sorted_others = sorted(others_ops, key=lambda x: x['latency'], reverse=True)
    for i, op in enumerate(sorted_others[:10]):
        print(f"{i+1:<5} {op['name'][:39]:<40} {op['type']:<15} {op['latency']/1000:<12.3f}")
    
    return others_by_type, total_others

def main():
    print("Analyzing Others category for both W4A8 and W8A8...")
    
    # Analyze both files
    w4a8_others, w4a8_total = analyze_others_detailed('q4_0-1-t1.txt', 'W4A8')
    w8a8_others, w8a8_total = analyze_others_detailed('q8_0-1-t1.txt', 'W8A8')
    
    # Compare operation types between W4A8 and W8A8
    print(f"\n{'='*80}")
    print("COMPARISON OF OTHERS OPERATIONS")
    print(f"{'='*80}")
    
    all_types = set(w4a8_others.keys()) | set(w8a8_others.keys())
    
    print(f"\n{'Type':<15} {'W4A8 (ms)':<12} {'W8A8 (ms)':<12} {'Ratio':<8} {'W4A8 Count':<12} {'W8A8 Count'}")
    print("-" * 80)
    
    for op_type in sorted(all_types):
        w4a8_latency = sum(op['latency'] for op in w4a8_others.get(op_type, []))
        w8a8_latency = sum(op['latency'] for op in w8a8_others.get(op_type, []))
        w4a8_count = len(w4a8_others.get(op_type, []))
        w8a8_count = len(w8a8_others.get(op_type, []))
        
        ratio = w8a8_latency / w4a8_latency if w4a8_latency > 0 else float('inf')
        
        print(f"{op_type:<15} {w4a8_latency/1000:<12.1f} {w8a8_latency/1000:<12.1f} {ratio:<8.1f} {w4a8_count:<12} {w8a8_count}")
    
    print("-" * 80)
    print(f"{'TOTAL':<15} {w4a8_total/1000:<12.1f} {w8a8_total/1000:<12.1f} {(w8a8_total/w4a8_total):<8.1f}")
    
    # Possible explanations
    print(f"\n{'='*80}")
    print("POSSIBLE EXPLANATIONS FOR DIFFERENCES")
    print(f"{'='*80}")
    
    print("""
1. MEMORY BANDWIDTH EFFECTS:
   - W4A8 uses less memory bandwidth for weights
   - Better cache utilization affects ALL operations
   - Non-quantized ops benefit from improved cache hit rates

2. PIPELINE AND SCHEDULING:
   - Different execution patterns between W4A8 and W8A8
   - CPU scheduling and resource contention
   - Memory controller efficiency

3. SYSTEM-LEVEL EFFECTS:
   - Different memory access patterns
   - Cache pollution from quantized operations
   - NUMA effects and memory locality

4. MEASUREMENT ARTIFACTS:
   - Timer precision and overhead
   - Background system activity
   - Thermal throttling differences
    """)

if __name__ == "__main__":
    main() 