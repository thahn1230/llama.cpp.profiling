import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
from collections import defaultdict

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
        elif name == 'inp_embd':
            current_layer = -1  # Embedding layer
            
        # Get tensor dimensions
        dim_line = lines[i+1].strip() if i+1 < len(lines) else ""
        dims = re.findall(r'\[([^\]]+)\]', dim_line)
        tensor_dims = dims[0] if dims else ""
            
        # More precise grouping
        if 'inp_embd' in name or 'GET_ROWS' in op_type:
            op_group = 'Embedding'
            is_quantized = True
        elif 'Qcur' in name and 'MUL_MAT' in op_type:
            op_group = 'Q Projection'
            is_quantized = True
        elif 'Kcur' in name and 'MUL_MAT' in op_type:
            op_group = 'K Projection'
            is_quantized = True
        elif 'Vcur' in name and 'MUL_MAT' in op_type:
            op_group = 'V Projection'
            is_quantized = True
        elif 'ROPE' in op_type:
            op_group = 'Positional Encoding'
            is_quantized = False
        elif any(x in name for x in ['node_']) and 'SOFT_MAX' in op_type:
            op_group = 'Attention Softmax'
            is_quantized = False
        elif any(x in name for x in ['node_']) and 'MUL_MAT' in op_type:
            op_group = 'Attention Score/Context'
            is_quantized = False
        elif 'attn_out' in name and 'MUL_MAT' in op_type:
            op_group = 'O Projection'
            is_quantized = True
        elif 'ffn_gate' in name or 'ffn_up' in name:
            op_group = 'FFN Up/Gate'
            is_quantized = True
        elif 'ffn_out' in name:
            op_group = 'FFN Down'
            is_quantized = True
        elif 'GLU' in op_type or 'ffn_swiglu' in name:
            op_group = 'FFN Activation'
            is_quantized = False
        elif 'norm' in name and 'RMS_NORM' in op_type:
            op_group = 'Layer Norm'
            is_quantized = False
        elif 'ADD' in op_type:
            op_group = 'Residual Add'
            is_quantized = False
        elif any(x in name for x in ['cache_', 'CPY', 'VIEW', 'PERMUTE', 'TRANSPOSE', 'RESHAPE', 'CONT']):
            op_group = 'Memory/Reshape'
            is_quantized = False
        elif 'attn_norm' in name or 'ffn_norm' in name:
            op_group = 'Pre-norm Scaling'
            is_quantized = False
        else:
            op_group = 'Other'
            is_quantized = False
            
        data.append({
            'name': name,
            'operation': op_type,
            'latency': int(latency),
            'layer': current_layer,
            'op_group': op_group,
            'execution_order': execution_order,
            'is_quantized_affected': is_quantized,
            'dimensions': tensor_dims
        })
        
        execution_order += 1
        
    return pd.DataFrame(data)

def create_comprehensive_analysis():
    print("="*80)
    print("COMPREHENSIVE LATENCY ANALYSIS: W4A8 vs W8A8")
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
    
    # Create output directory
    import os
    os.makedirs('profiling_results', exist_ok=True)
    
    # 1. Overall Statistics
    total_latency = df_combined.groupby('quantization')['latency'].sum()
    print(f"\n📊 OVERALL PERFORMANCE:")
    print("-" * 40)
    for quant, lat in total_latency.items():
        print(f"{quant}: {lat/1000:.2f} ms ({lat/16000:.2f} ms/token)")
    
    speedup = total_latency['W8A8'] / total_latency['W4A8']
    print(f"\nSpeedup: W4A8 is {speedup:.2f}x faster than W8A8")
    
    # 2. Operation Group Analysis with Percentages
    print(f"\n🔍 OPERATION GROUP BREAKDOWN:")
    print("-" * 60)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    for i, quant in enumerate(['W4A8', 'W8A8']):
        quant_data = df_combined[df_combined['quantization'] == quant]
        group_latency = quant_data.groupby('op_group')['latency'].sum().sort_values(ascending=False)
        total = group_latency.sum()
        group_pct = (group_latency / total * 100)
        
        print(f"\n{quant}:")
        print(f"{'Operation Group':<20} {'Latency (ms)':<12} {'Percentage':<10}")
        print("-" * 45)
        for group, lat in group_latency.items():
            pct = lat / total * 100
            print(f"{group:<20} {lat/1000:<12.2f} {pct:<10.1f}%")
        
        # Create pie chart
        ax = ax1 if i == 0 else ax2
        colors = plt.cm.Set3(np.linspace(0, 1, len(group_pct)))
        wedges, texts, autotexts = ax.pie(group_pct.values, labels=group_pct.index, autopct='%1.1f%%',
                                         colors=colors, startangle=90)
        ax.set_title(f'{quant} - Operation Group Distribution', fontsize=14, fontweight='bold')
        
        # Improve text readability
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    
    plt.tight_layout()
    plt.savefig('profiling_results/operation_group_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Layer-wise Analysis
    print(f"\n📈 LAYER-WISE ANALYSIS:")
    print("-" * 40)
    
    layer_data = df_combined[df_combined['layer'] >= 0]
    layer_latency = layer_data.groupby(['quantization', 'layer'])['latency'].sum()
    layer_latency_pivot = layer_latency.unstack(level=0) / 1000  # Convert to ms
    
    # Layer statistics
    layer_stats = layer_latency_pivot.describe()
    print("Layer latency statistics (ms):")
    print(layer_stats.round(2))
    
    # Layer-wise comparison plot
    plt.figure(figsize=(16, 10))
    
    # Main comparison plot
    plt.subplot(2, 2, 1)
    layer_latency_pivot.plot(marker='o', linewidth=2, markersize=4)
    plt.title('Layer-wise Latency Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Layer Number')
    plt.ylabel('Latency (ms)')
    plt.legend(title='Quantization')
    plt.grid(True, alpha=0.3)
    
    # Speedup per layer
    plt.subplot(2, 2, 2)
    speedup_per_layer = layer_latency_pivot['W8A8'] / layer_latency_pivot['W4A8']
    speedup_per_layer.plot(kind='bar', color='skyblue', alpha=0.7)
    plt.title('Speedup per Layer (W8A8/W4A8)', fontsize=14, fontweight='bold')
    plt.xlabel('Layer Number')
    plt.ylabel('Speedup Ratio')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Layer latency distribution
    plt.subplot(2, 2, 3)
    layer_latency_pivot.boxplot(ax=plt.gca())
    plt.title('Layer Latency Distribution', fontsize=14, fontweight='bold')
    plt.ylabel('Latency (ms)')
    
    # Cumulative latency
    plt.subplot(2, 2, 4)
    cumulative_latency = layer_latency_pivot.cumsum()
    cumulative_latency.plot(marker='o', linewidth=2)
    plt.title('Cumulative Latency by Layer', fontsize=14, fontweight='bold')
    plt.xlabel('Layer Number')
    plt.ylabel('Cumulative Latency (ms)')
    plt.legend(title='Quantization')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('profiling_results/layer_wise_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Operation Type Analysis
    plt.figure(figsize=(16, 10))
    
    # Top operations by total time
    plt.subplot(2, 2, 1)
    op_totals = df_combined.groupby(['quantization', 'operation'])['latency'].sum().unstack(level=0) / 1000
    top_ops = op_totals.sum(axis=1).nlargest(10)
    op_totals.loc[top_ops.index].plot(kind='barh', ax=plt.gca())
    plt.title('Top 10 Operations by Total Time', fontsize=14, fontweight='bold')
    plt.xlabel('Total Latency (ms)')
    
    # Operation count
    plt.subplot(2, 2, 2)
    op_counts = df_combined.groupby(['quantization', 'operation']).size().unstack(level=0)
    top_op_counts = op_counts.sum(axis=1).nlargest(10)
    op_counts.loc[top_op_counts.index].plot(kind='barh', ax=plt.gca())
    plt.title('Top 10 Operations by Count', fontsize=14, fontweight='bold')
    plt.xlabel('Operation Count')
    
    # Average latency per operation
    plt.subplot(2, 2, 3)
    avg_latency = df_combined.groupby(['quantization', 'operation'])['latency'].mean().unstack(level=0)
    avg_latency = avg_latency.dropna()
    top_avg = avg_latency.mean(axis=1).nlargest(10)
    avg_latency.loc[top_avg.index].plot(kind='barh', ax=plt.gca())
    plt.title('Top 10 Operations by Average Latency', fontsize=14, fontweight='bold')
    plt.xlabel('Average Latency (μs)')
    
    # Quantization impact
    plt.subplot(2, 2, 4)
    quant_impact = df_combined.groupby(['is_quantized_affected', 'quantization'])['latency'].sum().unstack(level=1) / 1000
    quant_impact.index = ['FP32 Operations', 'Quantized Operations']
    quant_impact.plot(kind='bar', ax=plt.gca())
    plt.title('Quantization Impact', fontsize=14, fontweight='bold')
    plt.ylabel('Total Latency (ms)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('profiling_results/operation_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Attention vs FFN Detailed Analysis
    attention_ops = ['Q Projection', 'K Projection', 'V Projection', 
                    'Positional Encoding', 'Attention Softmax', 
                    'Attention Score/Context', 'O Projection']
    ffn_ops = ['FFN Up/Gate', 'FFN Down', 'FFN Activation']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Attention vs FFN overall
    ax1 = axes[0, 0]
    attention_data = df_combined[df_combined['op_group'].isin(attention_ops)]
    ffn_data = df_combined[df_combined['op_group'].isin(ffn_ops)]
    
    attn_total = attention_data.groupby('quantization')['latency'].sum() / 1000
    ffn_total = ffn_data.groupby('quantization')['latency'].sum() / 1000
    
    comparison_data = pd.DataFrame({
        'Attention': attn_total,
        'FFN': ffn_total
    })
    comparison_data.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_title('Attention vs FFN Total Latency', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Latency (ms)')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
    
    # Attention breakdown
    ax2 = axes[0, 1]
    attn_breakdown = attention_data.groupby(['quantization', 'op_group'])['latency'].sum().unstack(level=0) / 1000
    attn_breakdown.plot(kind='bar', ax=ax2, width=0.8)
    ax2.set_title('Attention Operations Breakdown', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Latency (ms)')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    
    # FFN breakdown
    ax3 = axes[1, 0]
    ffn_breakdown = ffn_data.groupby(['quantization', 'op_group'])['latency'].sum().unstack(level=0) / 1000
    ffn_breakdown.plot(kind='bar', ax=ax3, width=0.8)
    ax3.set_title('FFN Operations Breakdown', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Latency (ms)')
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
    
    # Layer-wise Attention vs FFN
    ax4 = axes[1, 1]
    layer_attn = attention_data[attention_data['layer'] >= 0].groupby(['quantization', 'layer'])['latency'].sum().unstack(level=0) / 1000
    layer_ffn = ffn_data[ffn_data['layer'] >= 0].groupby(['quantization', 'layer'])['latency'].sum().unstack(level=0) / 1000
    
    layer_attn['W4A8'].plot(marker='o', label='W4A8 Attention', ax=ax4)
    layer_ffn['W4A8'].plot(marker='s', label='W4A8 FFN', ax=ax4)
    layer_attn['W8A8'].plot(marker='o', linestyle='--', label='W8A8 Attention', ax=ax4)
    layer_ffn['W8A8'].plot(marker='s', linestyle='--', label='W8A8 FFN', ax=ax4)
    
    ax4.set_title('Layer-wise Attention vs FFN', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Layer Number')
    ax4.set_ylabel('Latency (ms)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('profiling_results/attention_ffn_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Print detailed insights
    print(f"\n🎯 DETAILED INSIGHTS:")
    print("-" * 40)
    
    # Find most expensive layers
    print("Most expensive layers:")
    for quant in ['W4A8', 'W8A8']:
        if quant in layer_latency_pivot.columns:
            top_layers = layer_latency_pivot[quant].nlargest(3)
            print(f"\n{quant}:")
            for layer, lat in top_layers.items():
                print(f"  Layer {layer}: {lat:.2f} ms")
    
    # Attention vs FFN ratios
    print(f"\nAttention vs FFN Analysis:")
    for quant in ['W4A8', 'W8A8']:
        total_lat = total_latency[quant]
        attn_lat = attn_total.get(quant, 0) * 1000  # Convert back to μs
        ffn_lat = ffn_total.get(quant, 0) * 1000
        
        print(f"\n{quant}:")
        print(f"  Attention: {attn_lat/1000:.2f} ms ({attn_lat/total_lat*100:.1f}%)")
        print(f"  FFN: {ffn_lat/1000:.2f} ms ({ffn_lat/total_lat*100:.1f}%)")
        print(f"  Attention/FFN Ratio: {attn_lat/ffn_lat:.2f}")
    
    # Quantization efficiency
    quantized_ops = df_combined[df_combined['is_quantized_affected']]
    fp32_ops = df_combined[~df_combined['is_quantized_affected']]
    
    print(f"\nQuantization Efficiency:")
    for quant in ['W4A8', 'W8A8']:
        quant_lat = quantized_ops[quantized_ops['quantization'] == quant]['latency'].sum()
        fp32_lat = fp32_ops[fp32_ops['quantization'] == quant]['latency'].sum()
        
        print(f"\n{quant}:")
        print(f"  Quantized ops: {quant_lat/1000:.2f} ms ({quant_lat/(quant_lat+fp32_lat)*100:.1f}%)")
        print(f"  FP32 ops: {fp32_lat/1000:.2f} ms ({fp32_lat/(quant_lat+fp32_lat)*100:.1f}%)")
    
    return df_combined

if __name__ == "__main__":
    df = create_comprehensive_analysis()
    print(f"\n✅ Analysis complete! Check 'profiling_results/' for generated graphs.")
    print(f"   - operation_group_distribution.png")
    print(f"   - layer_wise_analysis.png") 
    print(f"   - operation_analysis.png")
    print(f"   - attention_ffn_analysis.png") 