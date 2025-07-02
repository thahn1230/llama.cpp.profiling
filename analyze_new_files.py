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

def analyze_16_token_generation():
    print("="*80)
    print("16 TOKEN GENERATION ANALYSIS: W4A8 vs W8A8")
    print("="*80)
    
    # Read and process data
    df_q4 = parse_latency_file('q4_0-1.txt')
    df_q8 = parse_latency_file('q8_0-1.txt')
    
    df_q4['quantization'] = 'W4A8'
    df_q8['quantization'] = 'W8A8'
    
    # Combine dataframes
    df_combined = pd.concat([df_q4, df_q8])
    
    # Set style for better plots
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Total latency comparison
    total_latency = df_combined.groupby('quantization')['latency'].sum()
    print(f"\n📊 TOTAL LATENCY COMPARISON (16 tokens):")
    print("-" * 40)
    for quant, lat in total_latency.items():
        print(f"{quant}: {lat/1000:.2f} ms")
    
    speedup = total_latency['W8A8'] / total_latency['W4A8']
    print(f"\nW4A8 is {speedup:.1f}x faster than W8A8")
    
    # Per-token analysis
    print(f"\nPer-token latency:")
    for quant, lat in total_latency.items():
        print(f"{quant}: {lat/16000:.2f} ms/token")
    
    # 2. Quantized vs FP32 operations analysis
    print(f"\n🔍 QUANTIZED vs FP32 OPERATIONS:")
    print("-" * 40)
    for quant in ['W4A8', 'W8A8']:
        quant_data = df_combined[df_combined['quantization'] == quant]
        fp32_total = quant_data[~quant_data['is_quantized_affected']]['latency'].sum()
        quantized_total = quant_data[quant_data['is_quantized_affected']]['latency'].sum()
        total = fp32_total + quantized_total
        
        print(f"\n{quant}:")
        print(f"  FP32 Operations: {fp32_total/1000:.2f} ms ({fp32_total/total*100:.1f}%)")
        print(f"  Quantized Operations: {quantized_total/1000:.2f} ms ({quantized_total/total*100:.1f}%)")
    
    # 3. Operation group breakdown
    print(f"\n🔍 OPERATION GROUP BREAKDOWN:")
    print("-" * 40)
    for quant in ['W4A8', 'W8A8']:
        print(f"\n{quant}:")
        group_data = df_combined[df_combined['quantization'] == quant]
        group_pct = group_data.groupby('op_group')['latency'].sum()
        total = group_pct.sum()
        
        # Sort by execution order
        group_pct_sorted = group_pct.reindex(sorted(group_pct.index, key=lambda x: int(x.split('.')[0])))
        
        for group, lat in group_pct_sorted.items():
            print(f"  {group}: {lat/1000:.2f} ms ({lat/total*100:.1f}%)")
    
    # 4. Top operations analysis
    print(f"\n⚠️  TOP BOTTLENECKS BY OPERATION TYPE:")
    print("-" * 40)
    for quant in ['W4A8', 'W8A8']:
        print(f"\n{quant}:")
        top_5 = df_combined[df_combined['quantization'] == quant].groupby('operation')['latency'].sum().sort_values(ascending=False).head(5)
        for i, (op, lat) in enumerate(top_5.items(), 1):
            print(f"  {i}. {op}: {lat/1000:.2f} ms")
    
    # 5. Attention vs FFN analysis
    attention_ops = ['2. Q Projection', '3. K Projection', '4. V Projection', 
                    '5. Positional Encoding', '6. Attention Softmax', 
                    '7. Attention Score/Context', '8. O Projection']
    ffn_ops = ['9. FFN Up/Gate', '10. FFN Down', '11. FFN Activation']
    
    attention_data = df_combined[df_combined['op_group'].isin(attention_ops)]
    ffn_data = df_combined[df_combined['op_group'].isin(ffn_ops)]
    
    attn_total = attention_data.groupby('quantization')['latency'].sum()
    ffn_total = ffn_data.groupby('quantization')['latency'].sum()
    
    print(f"\n🎯 ATTENTION vs FFN ANALYSIS:")
    print("-" * 40)
    for quant in ['W4A8', 'W8A8']:
        total_lat = total_latency[quant]
        attn_lat = attn_total.get(quant, 0)
        ffn_lat = ffn_total.get(quant, 0)
        
        print(f"\n{quant}:")
        print(f"  Attention Total: {attn_lat/1000:.2f} ms ({attn_lat/total_lat*100:.1f}%)")
        print(f"  FFN Total: {ffn_lat/1000:.2f} ms ({ffn_lat/total_lat*100:.1f}%)")
        print(f"  Attention/FFN Ratio: {attn_lat/ffn_lat:.2f}" if ffn_lat > 0 else "  Attention/FFN Ratio: N/A")
    
    # 6. Layer-wise analysis
    layer_data = df_combined[df_combined['layer'] >= 0]
    if not layer_data.empty:
        layer_latency = layer_data.groupby(['quantization', 'layer'])['latency'].sum()
        layer_latency_pivot = layer_latency.unstack(level=0)
        
        print(f"\n📈 LAYER-WISE ANALYSIS:")
        print("-" * 40)
        layer_stats = layer_latency_pivot.describe()
        print(f"Layer latency statistics (ms):")
        print((layer_stats/1000).round(2))
        
        # Find most expensive layers
        print(f"\nMost expensive layers:")
        for quant in ['W4A8', 'W8A8']:
            if quant in layer_latency_pivot.columns:
                top_layers = layer_latency_pivot[quant].nlargest(3)
                print(f"\n{quant}:")
                for layer, lat in top_layers.items():
                    print(f"  Layer {layer}: {lat/1000:.2f} ms")
    
    # 7. Create visualizations
    import os
    os.makedirs('profiling_results', exist_ok=True)
    
    # Quantized vs FP32 Operations
    plt.figure(figsize=(12, 6))
    quant_analysis = df_combined.groupby(['quantization', 'is_quantized_affected'])['latency'].sum().unstack()
    quant_analysis.columns = ['FP32 Operations', 'Quantized Operations']
    quant_analysis = quant_analysis / 1000  # Convert to ms
    
    ax = quant_analysis.plot(kind='bar', width=0.8)
    plt.title('16-Token Generation: Quantized vs FP32 Operations', fontsize=14, fontweight='bold')
    plt.xlabel('Quantization Method', fontsize=12)
    plt.ylabel('Total Latency (ms)', fontsize=12)
    plt.legend(title='Operation Type')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('profiling_results/16token_quantized_vs_fp32.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Operation Group Analysis
    plt.figure(figsize=(18, 10))
    group_latency = df_combined.groupby(['quantization', 'op_group'])['latency'].sum().unstack()
    group_latency = group_latency / 1000  # Convert to ms
    
    # Sort columns by group number
    group_order = sorted(group_latency.columns, key=lambda x: int(x.split('.')[0]))
    group_latency = group_latency[group_order]
    
    ax = group_latency.plot(kind='bar', width=0.8, figsize=(18, 10))
    plt.title('16-Token Generation: Total Latency by Operation Group', fontsize=16, fontweight='bold')
    plt.xlabel('Quantization Method', fontsize=14)
    plt.ylabel('Total Latency (ms)', fontsize=14)
    plt.legend(title='Operation Group', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('profiling_results/16token_op_group_latency.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Attention vs FFN Analysis
    plt.figure(figsize=(12, 6))
    comparison_data = pd.DataFrame({
        'Attention': attn_total / 1000,
        'FFN': ffn_total / 1000
    })
    
    ax = comparison_data.plot(kind='bar', width=0.8)
    plt.title('16-Token Generation: Attention vs FFN Latency', fontsize=14, fontweight='bold')
    plt.xlabel('Quantization Method', fontsize=12)
    plt.ylabel('Total Latency (ms)', fontsize=12)
    plt.legend(title='Component')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('profiling_results/16token_attention_vs_ffn.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n" + "="*80)
    print("16-TOKEN GENERATION INSIGHTS")
    print("="*80)
    
    print(f"""
🔬 KEY FINDINGS (16 tokens generation):

1. QUANTIZATION IMPACT:
   - W4A8 achieves {speedup:.1f}x speedup over W8A8
   - W4A8: {total_latency['W4A8']/16000:.2f} ms/token
   - W8A8: {total_latency['W8A8']/16000:.2f} ms/token

2. BOTTLENECK ANALYSIS:
   - Total inference time for 16 tokens: W4A8={total_latency['W4A8']/1000:.1f}ms, W8A8={total_latency['W8A8']/1000:.1f}ms
   - Quantized operations dominate the latency in both configurations
   - Matrix multiplications (MUL_MAT) are the primary bottleneck

3. ATTENTION vs FFN COMPARISON:
   - W4A8: Attention={attn_total.get('W4A8', 0)/total_latency['W4A8']*100:.1f}%, FFN={ffn_total.get('W4A8', 0)/total_latency['W4A8']*100:.1f}%
   - W8A8: Attention={attn_total.get('W8A8', 0)/total_latency['W8A8']*100:.1f}%, FFN={ffn_total.get('W8A8', 0)/total_latency['W8A8']*100:.1f}%

4. OPTIMIZATION OPPORTUNITIES:
   - Focus on optimizing quantized matrix operations (MUL_MAT)
   - Q Projection shows significant overhead in W8A8
   - FFN operations benefit significantly from 4-bit quantization
""")

if __name__ == "__main__":
    analyze_16_token_generation() 