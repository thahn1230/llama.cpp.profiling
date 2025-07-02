import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np

def parse_latency_file(file_path):
    data = []
    current_layer = -1
    execution_order = 0
    
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
            
        # More precise grouping to separate quantized vs non-quantized operations
        if 'inp_embd' in name or 'GET_ROWS' in op_type:
            op_group = '1. Embedding'
            is_quantized = True  # Embedding lookup is affected by quantization
        elif 'Qcur' in name and 'MUL_MAT' in op_type:
            op_group = '2. Q Projection'
            is_quantized = True  # Weight matrix operation
        elif 'Kcur' in name and 'MUL_MAT' in op_type:
            op_group = '3. K Projection'
            is_quantized = True  # Weight matrix operation
        elif 'Vcur' in name and 'MUL_MAT' in op_type:
            op_group = '4. V Projection'
            is_quantized = True  # Weight matrix operation
        elif 'ROPE' in op_type:
            op_group = '5. Positional Encoding'
            is_quantized = False  # FP32 operation
        elif any(x in name for x in ['node_']) and 'SOFT_MAX' in op_type:
            op_group = '6. Attention Softmax'
            is_quantized = False  # FP32 operation
        elif any(x in name for x in ['node_']) and 'MUL_MAT' in op_type:
            op_group = '7. Attention Score/Context'
            is_quantized = False  # FP32 attention computation
        elif 'attn_out' in name and 'MUL_MAT' in op_type:
            op_group = '8. O Projection'
            is_quantized = True  # Weight matrix operation
        elif 'ffn_gate' in name or 'ffn_up' in name:
            op_group = '9. FFN Up/Gate'
            is_quantized = True  # Weight matrix operation
        elif 'ffn_out' in name:
            op_group = '10. FFN Down'
            is_quantized = True  # Weight matrix operation
        elif 'GLU' in op_type or 'ffn_swiglu' in name:
            op_group = '11. FFN Activation'
            is_quantized = False  # FP32 activation
        elif 'norm' in name and 'RMS_NORM' in op_type:
            op_group = '12. Layer Norm'
            is_quantized = False  # FP32 operation
        elif 'ADD' in op_type:
            op_group = '13. Residual Add'
            is_quantized = False  # FP32 operation
        elif any(x in name for x in ['cache_', 'CPY', 'VIEW', 'PERMUTE', 'TRANSPOSE', 'RESHAPE', 'CONT']):
            op_group = '14. Memory/Reshape Ops'
            is_quantized = False  # Memory operations
        elif 'attn_norm' in name or 'ffn_norm' in name:
            op_group = '15. Pre-norm Scaling'
            is_quantized = False  # FP32 scaling
        else:
            op_group = '16. Other'
            is_quantized = False
            
        data.append({
            'name': name,
            'operation': op_type,
            'latency': int(latency),
            'layer': current_layer,
            'op_group': op_group,
            'execution_order': execution_order,
            'is_quantized_affected': is_quantized
        })
        
        execution_order += 1
        
    return pd.DataFrame(data)

def compare_fp32_consistency(df1, df2, label1, label2):
    """Compare FP32 operations between two datasets"""
    
    fp32_ops = ['5. Positional Encoding', '6. Attention Softmax', '7. Attention Score/Context', 
                '11. FFN Activation', '12. Layer Norm', '13. Residual Add']
    
    print(f"\n🔍 FP32 OPERATION CONSISTENCY ANALYSIS:")
    print(f"Comparing {label1} vs {label2}")
    print("-" * 60)
    
    inconsistencies = []
    
    for op_group in fp32_ops:
        data1 = df1[df1['op_group'] == op_group]
        data2 = df2[df2['op_group'] == op_group]
        
        if len(data1) > 0 and len(data2) > 0:
            # For W4A8 vs W8A8 within same dataset
            if 'quantization' in df1.columns:
                lat1_w4 = data1[data1['quantization'] == 'W4A8']['latency'].sum()
                lat1_w8 = data1[data1['quantization'] == 'W8A8']['latency'].sum()
                lat2_w4 = data2[data2['quantization'] == 'W4A8']['latency'].sum()
                lat2_w8 = data2[data2['quantization'] == 'W8A8']['latency'].sum()
                
                if lat1_w4 > 0 and lat1_w8 > 0 and lat2_w4 > 0 and lat2_w8 > 0:
                    ratio1 = lat1_w8 / lat1_w4 if lat1_w4 > 0 else 0
                    ratio2 = lat2_w8 / lat2_w4 if lat2_w4 > 0 else 0
                    
                    print(f"{op_group}:")
                    print(f"  {label1}: W4A8={lat1_w4/1000:.2f}ms, W8A8={lat1_w8/1000:.2f}ms (ratio: {ratio1:.2f})")
                    print(f"  {label2}: W4A8={lat2_w4/1000:.2f}ms, W8A8={lat2_w8/1000:.2f}ms (ratio: {ratio2:.2f})")
                    
                    if abs(ratio1 - 1.0) > 0.5 or abs(ratio2 - 1.0) > 0.5:
                        inconsistencies.append((op_group, ratio1, ratio2))
                        print(f"  ⚠️  INCONSISTENCY DETECTED!")
                    else:
                        print(f"  ✓  Consistent")
                    print()
    
    return inconsistencies

def analyze_both_datasets():
    # Analyze original files
    print("="*80)
    print("ANALYZING ORIGINAL FILES: q4_0.txt vs q8_0.txt")
    print("="*80)
    
    df_orig_q4 = parse_latency_file('q4_0.txt')
    df_orig_q8 = parse_latency_file('q8_0.txt')
    df_orig_q4['quantization'] = 'W4A8'
    df_orig_q8['quantization'] = 'W8A8'
    df_orig = pd.concat([df_orig_q4, df_orig_q8])
    
    # Analyze new files
    print("\n" + "="*80)
    print("ANALYZING NEW FILES: q4_0-1.txt vs q8_0-1.txt")
    print("="*80)
    
    df_new_q4 = parse_latency_file('q4_0-1.txt')
    df_new_q8 = parse_latency_file('q8_0-1.txt')
    df_new_q4['quantization'] = 'W4A8'
    df_new_q8['quantization'] = 'W8A8'
    df_new = pd.concat([df_new_q4, df_new_q8])
    
    # Compare totals
    print(f"\n📊 TOTAL LATENCY COMPARISON:")
    print("-" * 40)
    
    orig_totals = df_orig.groupby('quantization')['latency'].sum()
    new_totals = df_new.groupby('quantization')['latency'].sum()
    
    print("Original files:")
    for quant, lat in orig_totals.items():
        print(f"  {quant}: {lat/1000:.2f} ms")
    
    print("New files:")
    for quant, lat in new_totals.items():
        print(f"  {quant}: {lat/1000:.2f} ms")
    
    # Compare FP32 operations consistency
    inconsistencies = compare_fp32_consistency(df_orig, df_new, "Original", "New")
    
    # Analysis of the inconsistency problem
    print("\n" + "="*80)
    print("🚨 CRITICAL ANALYSIS: WHY FP32 OPERATIONS DIFFER")
    print("="*80)
    
    print("""
🔍 POSSIBLE EXPLANATIONS FOR FP32 OPERATION DIFFERENCES:

1. 📏 DIFFERENT SEQUENCE LENGTHS:
   - Original files might use different input lengths than new files
   - Attention operations scale quadratically with sequence length
   - This would explain why attention ops show largest differences

2. 🔧 DIFFERENT COMPILATION/OPTIMIZATION FLAGS:
   - Different BLAS libraries (OpenBLAS vs Intel MKL)
   - Different compiler optimizations
   - Different SIMD instruction usage

3. 🖥️  DIFFERENT HARDWARE/ENVIRONMENT:
   - CPU frequency scaling
   - Memory bandwidth differences
   - Thermal throttling
   - Background processes

4. ⚙️  DIFFERENT IMPLEMENTATION VERSIONS:
   - llama.cpp version differences
   - GGML backend changes
   - Operator implementation updates

5. 📊 MEASUREMENT CONDITIONS:
   - Cold vs warm cache
   - System load during measurement
   - Multiple runs vs single run
    """)
    
    # Specific analysis for the inconsistencies
    if inconsistencies:
        print(f"\n⚠️  DETECTED {len(inconsistencies)} INCONSISTENT OPERATIONS:")
        print("-" * 50)
        
        for op_group, ratio1, ratio2 in inconsistencies:
            print(f"{op_group}:")
            print(f"  Original ratio (W8A8/W4A8): {ratio1:.2f}")
            print(f"  New ratio (W8A8/W4A8): {ratio2:.2f}")
            print(f"  Difference: {abs(ratio2-ratio1):.2f}")
            
            if ratio1 > 2.0 or ratio2 > 2.0:
                print(f"  🚨 HIGH INCONSISTENCY - Likely sequence length or env difference")
            elif abs(ratio1 - ratio2) > 1.0:
                print(f"  ⚠️  MODERATE INCONSISTENCY - Possible implementation difference")
            print()
    
    print(f"\n💡 RECOMMENDATIONS:")
    print("-" * 30)
    print("""
1. CHECK EXPERIMENTAL CONDITIONS:
   - Verify same input sequence length
   - Use identical hardware/environment
   - Same llama.cpp version and compilation flags

2. CONTROLLED COMPARISON:
   - Run multiple measurements and average
   - Ensure consistent system state
   - Use same model and parameters

3. FOCUS ON RELATIVE TRENDS:
   - Despite absolute differences, relative patterns may still be valid
   - Look for consistent speedup ratios in quantized operations
   - Pay attention to operation type rankings
    """)
    
    return df_orig, df_new

if __name__ == "__main__":
    df_orig, df_new = analyze_both_datasets() 