import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np

def extract_operations_by_layer(file_path):
    """Extract operations grouped by layer with detailed information"""
    operations = {}
    current_layer = -1
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    for i in range(0, len(lines), 3):
        if i + 2 >= len(lines):
            break
            
        latency_line = lines[i].strip()
        if not latency_line.startswith('[[[[Latency for Tensor]]]]'):
            continue
            
        # Extract operation name and latency
        match = re.search(r"'([^']+)' \(([^)]+)\): (\d+) us", latency_line)
        if not match:
            continue
            
        name, op_type, latency = match.groups()
        
        # Extract layer number if present
        layer_match = re.search(r'-(\d+)', name)
        if layer_match:
            current_layer = int(layer_match.group(1))
        elif name == 'inp_embd':
            current_layer = -1  # Embedding layer
            
        # Get tensor dimensions
        dim_line = lines[i+1].strip() if i+1 < len(lines) else ""
        dims = re.findall(r'\[([^\]]+)\]', dim_line)
        tensor_dims = dims[0] if dims else ""
        
        if current_layer not in operations:
            operations[current_layer] = []
            
        operations[current_layer].append({
            'name': name,
            'operation': op_type,
            'latency': int(latency),
            'dimensions': tensor_dims,
            'layer': current_layer
        })
        
    return operations

def compare_files_detailed():
    print("="*80)
    print("DETAILED OPERATION-BY-OPERATION ANALYSIS")
    print("="*80)
    
    # Extract operations from both files
    ops_q4_orig = extract_operations_by_layer('q4_0.txt')
    ops_q8_orig = extract_operations_by_layer('q8_0.txt')
    ops_q4_new = extract_operations_by_layer('q4_0-1.txt')
    ops_q8_new = extract_operations_by_layer('q8_0-1.txt')
    
    print(f"\n📋 LAYER STRUCTURE COMPARISON:")
    print("-" * 50)
    print(f"Original files - Layers: {sorted(ops_q4_orig.keys())}")
    print(f"New files - Layers: {sorted(ops_q4_new.keys())}")
    
    # Compare layer 0 in detail
    print(f"\n🔍 LAYER 0 DETAILED COMPARISON:")
    print("-" * 50)
    
    layer_0_orig_q4 = ops_q4_orig.get(0, [])
    layer_0_orig_q8 = ops_q8_orig.get(0, [])
    layer_0_new_q4 = ops_q4_new.get(0, [])
    layer_0_new_q8 = ops_q8_new.get(0, [])
    
    print(f"\nOriginal Q4 Layer 0 - {len(layer_0_orig_q4)} operations")
    print(f"Original Q8 Layer 0 - {len(layer_0_orig_q8)} operations")
    print(f"New Q4 Layer 0 - {len(layer_0_new_q4)} operations")
    print(f"New Q8 Layer 0 - {len(layer_0_new_q8)} operations")
    
    # Compare same operations across files
    print(f"\n🔍 OPERATION-BY-OPERATION COMPARISON (Layer 0):")
    print("-" * 70)
    print(f"{'Operation':<20} {'Orig Q4':<10} {'Orig Q8':<10} {'New Q4':<10} {'New Q8':<10} {'Ratio O':<8} {'Ratio N':<8}")
    print("-" * 70)
    
    # Create operation lookup dictionaries
    orig_q4_ops = {op['name']: op for op in layer_0_orig_q4}
    orig_q8_ops = {op['name']: op for op in layer_0_orig_q8}
    new_q4_ops = {op['name']: op for op in layer_0_new_q4}
    new_q8_ops = {op['name']: op for op in layer_0_new_q8}
    
    # Get all operation names
    all_ops = set(orig_q4_ops.keys()) | set(orig_q8_ops.keys()) | set(new_q4_ops.keys()) | set(new_q8_ops.keys())
    
    suspicious_ops = []
    
    for op_name in sorted(all_ops):
        orig_q4_lat = orig_q4_ops.get(op_name, {}).get('latency', 0)
        orig_q8_lat = orig_q8_ops.get(op_name, {}).get('latency', 0)
        new_q4_lat = new_q4_ops.get(op_name, {}).get('latency', 0)
        new_q8_lat = new_q8_ops.get(op_name, {}).get('latency', 0)
        
        ratio_orig = orig_q8_lat / orig_q4_lat if orig_q4_lat > 0 else 0
        ratio_new = new_q8_lat / new_q4_lat if new_q4_lat > 0 else 0
        
        print(f"{op_name[:20]:<20} {orig_q4_lat:<10} {orig_q8_lat:<10} {new_q4_lat:<10} {new_q8_lat:<10} {ratio_orig:<8.2f} {ratio_new:<8.2f}")
        
        # Flag suspicious operations
        if abs(ratio_orig - ratio_new) > 2.0 and min(orig_q4_lat, orig_q8_lat, new_q4_lat, new_q8_lat) > 100:
            suspicious_ops.append((op_name, ratio_orig, ratio_new))
    
    # Analyze specific operation types
    print(f"\n🚨 SUSPICIOUS OPERATIONS (Large ratio differences):")
    print("-" * 50)
    for op_name, ratio_orig, ratio_new in suspicious_ops:
        op_type = orig_q4_ops.get(op_name, {}).get('operation', 'Unknown')
        print(f"{op_name} ({op_type}): Original ratio={ratio_orig:.2f}, New ratio={ratio_new:.2f}")
        
        # Check if it's supposed to be FP32
        if op_type in ['ROPE', 'SOFT_MAX', 'GLU', 'RMS_NORM', 'ADD']:
            print(f"  ⚠️  This should be FP32 operation - ratios should be ~1.0!")
        elif op_type in ['MUL_MAT']:
            print(f"  ℹ️  Weight matrix operation - ratio difference expected")
        else:
            print(f"  ❓ Unknown operation type")
    
    # Summary statistics
    print(f"\n📊 SUMMARY:")
    print("-" * 30)
    
    total_orig_q4 = sum(op['latency'] for ops in ops_q4_orig.values() for op in ops)
    total_orig_q8 = sum(op['latency'] for ops in ops_q8_orig.values() for op in ops)
    total_new_q4 = sum(op['latency'] for ops in ops_q4_new.values() for op in ops)
    total_new_q8 = sum(op['latency'] for ops in ops_q8_new.values() for op in ops)
    
    print(f"Total latencies:")
    print(f"  Original: Q4={total_orig_q4/1000:.1f}ms, Q8={total_orig_q8/1000:.1f}ms (ratio: {total_orig_q8/total_orig_q4:.2f})")
    print(f"  New:      Q4={total_new_q4/1000:.1f}ms, Q8={total_new_q8/1000:.1f}ms (ratio: {total_new_q8/total_new_q4:.2f})")
    
    # Check for obvious issues
    print(f"\n🔬 POTENTIAL ISSUES:")
    print("-" * 30)
    
    if total_new_q4 > total_orig_q4 * 5:
        print(f"❌ New Q4 is {total_new_q4/total_orig_q4:.1f}x slower than original Q4")
        print("   This suggests system performance degradation or different measurement conditions")
    
    if abs((total_new_q8/total_new_q4) - (total_orig_q8/total_orig_q4)) > 5:
        print(f"❌ Speedup ratios are very different:")
        print(f"   Original: {total_orig_q8/total_orig_q4:.1f}x")
        print(f"   New: {total_new_q8/total_new_q4:.1f}x")
        print("   This suggests inconsistent measurement conditions")
    
    # Check first few operations for pattern
    print(f"\n🔍 FIRST 10 OPERATIONS COMPARISON:")
    print("-" * 50)
    
    # Get embedding and first layer operations
    embedding_ops_orig_q4 = ops_q4_orig.get(-1, [])
    embedding_ops_new_q4 = ops_q4_new.get(-1, [])
    
    if embedding_ops_orig_q4 and embedding_ops_new_q4:
        emb_orig = embedding_ops_orig_q4[0]['latency']
        emb_new = embedding_ops_new_q4[0]['latency']
        print(f"Embedding: Original={emb_orig}us, New={emb_new}us (ratio: {emb_new/emb_orig:.2f})")
        
        if emb_new > emb_orig * 10:
            print("❌ Embedding operation much slower in new files - likely system issue")
    
    return {
        'orig_q4': ops_q4_orig,
        'orig_q8': ops_q8_orig,
        'new_q4': ops_q4_new,
        'new_q8': ops_q8_new,
        'suspicious_ops': suspicious_ops
    }

if __name__ == "__main__":
    results = compare_files_detailed() 