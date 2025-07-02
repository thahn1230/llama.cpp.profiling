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

def analyze_latency_comparison(q4_path, q8_path, output_prefix="comparison"):
    # Read and process data
    print(f"Reading {q4_path} and {q8_path}...")
    df_q4 = parse_latency_file(q4_path)
    df_q8 = parse_latency_file(q8_path)
    
    df_q4['quantization'] = 'W4A8'
    df_q8['quantization'] = 'W8A8'
    
    # Combine dataframes
    df_combined = pd.concat([df_q4, df_q8])
    
    # Set style for better plots
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create output directory
    import os
    os.makedirs('profiling_results', exist_ok=True)
    
    # 1. Quantized vs Non-quantized Operations Analysis
    plt.figure(figsize=(16, 8))
    quant_analysis = df_combined.groupby(['quantization', 'is_quantized_affected'])['latency'].sum().unstack()
    quant_analysis.columns = ['FP32 Operations', 'Quantized Operations']
    
    ax = quant_analysis.plot(kind='bar', width=0.8)
    plt.title('Quantized vs FP32 Operations Latency', fontsize=14, fontweight='bold')
    plt.xlabel('Quantization Method', fontsize=12)
    plt.ylabel('Total Latency (us)', fontsize=12)
    plt.legend(title='Operation Type')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'profiling_results/{output_prefix}_quantized_vs_fp32.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Operation Group Analysis (ordered by execution)
    plt.figure(figsize=(18, 10))
    group_latency = df_combined.groupby(['quantization', 'op_group'])['latency'].sum().unstack()
    
    # Sort columns by group number
    group_order = sorted(group_latency.columns, key=lambda x: int(x.split('.')[0]))
    group_latency = group_latency[group_order]
    
    ax = group_latency.plot(kind='bar', width=0.8, figsize=(18, 10))
    plt.title(f'Total Latency by Operation Group - {output_prefix}', fontsize=16, fontweight='bold')
    plt.xlabel('Quantization Method', fontsize=14)
    plt.ylabel('Total Latency (us)', fontsize=14)
    plt.legend(title='Operation Group', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'profiling_results/{output_prefix}_op_group_latency.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Layer-wise Analysis
    plt.figure(figsize=(16, 8))
    layer_data = df_combined[df_combined['layer'] >= 0]
    layer_latency = layer_data.groupby(['quantization', 'layer'])['latency'].sum()
    layer_latency_pivot = layer_latency.unstack(level=0)
    
    if not layer_latency_pivot.empty:
        layer_latency_pivot.plot(marker='o', linewidth=2, markersize=6)
        plt.title(f'Layer-wise Latency Comparison - {output_prefix}', fontsize=14, fontweight='bold')
        plt.xlabel('Layer Number', fontsize=12)
        plt.ylabel('Total Latency (us)', fontsize=12)
        plt.legend(title='Quantization Method')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'profiling_results/{output_prefix}_layer_latency.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Attention vs FFN Analysis
    plt.figure(figsize=(14, 8))
    attention_ops = ['2. Q Projection', '3. K Projection', '4. V Projection', 
                    '5. Positional Encoding', '6. Attention Softmax', 
                    '7. Attention Score/Context', '8. O Projection']
    ffn_ops = ['9. FFN Up/Gate', '10. FFN Down', '11. FFN Activation']
    
    attention_data = df_combined[df_combined['op_group'].isin(attention_ops)]
    ffn_data = df_combined[df_combined['op_group'].isin(ffn_ops)]
    
    attn_total = attention_data.groupby('quantization')['latency'].sum()
    ffn_total = ffn_data.groupby('quantization')['latency'].sum()
    
    comparison_data = pd.DataFrame({
        'Attention': attn_total,
        'FFN': ffn_total
    })
    
    ax = comparison_data.plot(kind='bar', width=0.8)
    plt.title(f'Attention vs FFN Latency - {output_prefix}', fontsize=14, fontweight='bold')
    plt.xlabel('Quantization Method', fontsize=12)
    plt.ylabel('Total Latency (us)', fontsize=12)
    plt.legend(title='Component')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'profiling_results/{output_prefix}_attention_vs_ffn.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print detailed analysis
    print("\n" + "="*80)
    print(f"DETAILED LATENCY ANALYSIS REPORT - {output_prefix}")
    print("="*80)
    
    # Total latency comparison
    total_latency = df_combined.groupby('quantization')['latency'].sum()
    print(f"\n📊 TOTAL LATENCY COMPARISON:")
    print("-" * 40)
    for quant, lat in total_latency.items():
        print(f"{quant}: {lat/1000:.2f} ms")
    
    if len(total_latency) == 2:
        speedup = total_latency['W8A8'] / total_latency['W4A8']
        print(f"\nW4A8 is {speedup:.1f}x faster than W8A8")
    
    # Quantized vs FP32 operations analysis
    print(f"\n🔍 QUANTIZED vs FP32 OPERATIONS:")
    print("-" * 40)
    for quant in ['W4A8', 'W8A8']:
        if quant in df_combined['quantization'].values:
            quant_data = df_combined[df_combined['quantization'] == quant]
            fp32_total = quant_data[~quant_data['is_quantized_affected']]['latency'].sum()
            quantized_total = quant_data[quant_data['is_quantized_affected']]['latency'].sum()
            total = fp32_total + quantized_total
            
            print(f"\n{quant}:")
            print(f"  FP32 Operations: {fp32_total/1000:.2f} ms ({fp32_total/total*100:.1f}%)")
            print(f"  Quantized Operations: {quantized_total/1000:.2f} ms ({quantized_total/total*100:.1f}%)")
    
    # Operation group breakdown
    print(f"\n🔍 OPERATION GROUP BREAKDOWN:")
    print("-" * 40)
    for quant in ['W4A8', 'W8A8']:
        if quant in df_combined['quantization'].values:
            print(f"\n{quant}:")
            group_data = df_combined[df_combined['quantization'] == quant]
            group_pct = group_data.groupby('op_group')['latency'].sum()
            total = group_pct.sum()
            
            # Sort by execution order
            group_pct_sorted = group_pct.reindex(sorted(group_pct.index, key=lambda x: int(x.split('.')[0])))
            
            for group, lat in group_pct_sorted.items():
                print(f"  {group}: {lat/1000:.2f} ms ({lat/total*100:.1f}%)")
    
    # Check for inconsistencies in FP32 operations
    print(f"\n⚠️  FP32 OPERATION CONSISTENCY CHECK:")
    print("-" * 40)
    
    fp32_ops = ['5. Positional Encoding', '6. Attention Softmax', '7. Attention Score/Context', 
                '11. FFN Activation', '12. Layer Norm', '13. Residual Add']
    
    for op_group in fp32_ops:
        op_data = df_combined[df_combined['op_group'] == op_group]
        if len(op_data) > 0:
            latencies = op_data.groupby('quantization')['latency'].sum()
            if len(latencies) == 2:
                ratio = latencies.max() / latencies.min()
                if ratio > 1.5:  # Flag if difference is more than 50%
                    print(f"⚠️  {op_group}: W4A8={latencies.get('W4A8', 0)/1000:.2f}ms, W8A8={latencies.get('W8A8', 0)/1000:.2f}ms (ratio: {ratio:.2f})")
                else:
                    print(f"✓  {op_group}: W4A8={latencies.get('W4A8', 0)/1000:.2f}ms, W8A8={latencies.get('W8A8', 0)/1000:.2f}ms (ratio: {ratio:.2f})")
    
    return df_combined

if __name__ == "__main__":
    # Analyze the new files
    print("Analyzing q4_0-1.txt vs q8_0-1.txt...")
    df_new = analyze_latency_comparison('q4_0-1.txt', 'q8_0-1.txt', 'new_comparison')
    
    print("\n" + "="*80)
    print("ADDITIONAL ANALYSIS COMPLETE")
    print("="*80) 