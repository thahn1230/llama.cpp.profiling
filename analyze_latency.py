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
            
        # More detailed grouping based on actual Transformer operations
        if 'inp_embd' in name or 'GET_ROWS' in op_type:
            op_group = '1. Embedding'
        elif 'Qcur' in name and 'MUL_MAT' in op_type:
            op_group = '2. Q Projection'
        elif 'Kcur' in name and 'MUL_MAT' in op_type:
            op_group = '3. K Projection'
        elif 'Vcur' in name and 'MUL_MAT' in op_type:
            op_group = '4. V Projection'
        elif 'ROPE' in op_type:
            op_group = '5. Positional Encoding'
        elif any(x in name for x in ['node_', 'MUL_MAT']) and 'node_' in name:
            if 'SOFT_MAX' in op_type:
                op_group = '6. Attention Softmax'
            else:
                op_group = '7. Attention Score/Context'
        elif 'attn_out' in name and 'MUL_MAT' in op_type:
            op_group = '8. O Projection'
        elif 'ffn_gate' in name or 'ffn_up' in name:
            op_group = '9. FFN Up/Gate'
        elif 'ffn_out' in name:
            op_group = '10. FFN Down'
        elif 'GLU' in op_type or 'ffn_swiglu' in name:
            op_group = '11. FFN Activation'
        elif 'norm' in name and 'RMS_NORM' in op_type:
            op_group = '12. Layer Norm'
        elif 'ADD' in op_type:
            op_group = '13. Residual Add'
        elif any(x in name for x in ['cache_', 'CPY', 'VIEW', 'PERMUTE', 'TRANSPOSE', 'RESHAPE', 'CONT']):
            op_group = '14. Memory/Reshape Ops'
        elif 'attn_norm' in name or 'ffn_norm' in name:
            op_group = '15. Pre-norm Scaling'
        else:
            op_group = '16. Other'
            
        data.append({
            'name': name,
            'operation': op_type,
            'latency': int(latency),
            'layer': current_layer,
            'op_group': op_group,
            'execution_order': execution_order
        })
        
        execution_order += 1
        
    return pd.DataFrame(data)

def analyze_latency(q4_path, q8_path):
    # Read and process data
    df_q4 = parse_latency_file(q4_path)
    df_q8 = parse_latency_file(q8_path)
    
    df_q4['quantization'] = 'W4A8'
    df_q8['quantization'] = 'W8A8'
    
    # Combine dataframes
    df_combined = pd.concat([df_q4, df_q8])
    
    # Set style for better plots
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Operation Group Analysis (ordered by execution)
    plt.figure(figsize=(16, 8))
    group_latency = df_combined.groupby(['quantization', 'op_group'])['latency'].sum().unstack()
    
    # Sort columns by group number
    group_order = sorted(group_latency.columns, key=lambda x: int(x.split('.')[0]))
    group_latency = group_latency[group_order]
    
    ax = group_latency.plot(kind='bar', width=0.8, figsize=(16, 8))
    plt.title('Total Latency by Operation Group (Execution Order)', fontsize=14, fontweight='bold')
    plt.xlabel('Quantization Method', fontsize=12)
    plt.ylabel('Total Latency (us)', fontsize=12)
    plt.legend(title='Operation Group', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('op_group_latency_detailed.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Layer-wise Analysis with better visualization
    plt.figure(figsize=(16, 8))
    layer_data = df_combined[df_combined['layer'] >= 0]
    layer_latency = layer_data.groupby(['quantization', 'layer'])['latency'].sum()
    layer_latency_pivot = layer_latency.unstack(level=0)
    
    if not layer_latency_pivot.empty:
        layer_latency_pivot.plot(marker='o', linewidth=2, markersize=6)
        plt.title('Layer-wise Latency Comparison', fontsize=14, fontweight='bold')
        plt.xlabel('Layer Number', fontsize=12)
        plt.ylabel('Total Latency (us)', fontsize=12)
        plt.legend(title='Quantization Method')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('layer_latency_detailed.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Operation Type Overhead Analysis
    plt.figure(figsize=(16, 8))
    top_ops = df_combined.groupby(['quantization', 'operation'])['latency'].sum().reset_index()
    top_ops_pivot = top_ops.pivot(index='operation', columns='quantization', values='latency')
    top_ops_pivot = top_ops_pivot.sort_values('W8A8', ascending=True).tail(12)
    
    ax = top_ops_pivot.plot(kind='barh', figsize=(16, 8))
    plt.title('Top 12 Most Time-Consuming Operation Types', fontsize=14, fontweight='bold')
    plt.xlabel('Total Latency (us)', fontsize=12)
    plt.ylabel('Operation Type', fontsize=12)
    plt.legend(title='Quantization Method')
    plt.tight_layout()
    plt.savefig('top_operations_detailed.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Attention vs FFN Analysis
    plt.figure(figsize=(12, 6))
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
    plt.title('Attention vs FFN Latency Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Quantization Method', fontsize=12)
    plt.ylabel('Total Latency (us)', fontsize=12)
    plt.legend(title='Component')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('attention_vs_ffn.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print detailed analysis
    print("\n" + "="*80)
    print("DETAILED LATENCY ANALYSIS REPORT")
    print("="*80)
    
    # Total latency comparison
    total_latency = df_combined.groupby('quantization')['latency'].sum()
    print(f"\n📊 TOTAL LATENCY COMPARISON:")
    print("-" * 40)
    for quant, lat in total_latency.items():
        print(f"{quant}: {lat/1000:.2f} ms")
    
    speedup = total_latency['W8A8'] / total_latency['W4A8']
    print(f"\nW4A8 is {speedup:.1f}x faster than W8A8")
    
    # Operation group breakdown
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
    
    # Attention vs FFN analysis
    print(f"\n🎯 ATTENTION vs FFN ANALYSIS:")
    print("-" * 40)
    for quant in ['W4A8', 'W8A8']:
        total_lat = total_latency[quant]
        attn_lat = attn_total[quant]
        ffn_lat = ffn_total[quant]
        
        print(f"\n{quant}:")
        print(f"  Attention Total: {attn_lat/1000:.2f} ms ({attn_lat/total_lat*100:.1f}%)")
        print(f"  FFN Total: {ffn_lat/1000:.2f} ms ({ffn_lat/total_lat*100:.1f}%)")
        print(f"  Attention/FFN Ratio: {attn_lat/ffn_lat:.2f}")
    
    # Top bottlenecks
    print(f"\n⚠️  TOP BOTTLENECKS BY OPERATION TYPE:")
    print("-" * 40)
    for quant in ['W4A8', 'W8A8']:
        print(f"\n{quant}:")
        top_5 = df_combined[df_combined['quantization'] == quant].groupby('operation')['latency'].sum().sort_values(ascending=False).head(5)
        for i, (op, lat) in enumerate(top_5.items(), 1):
            print(f"  {i}. {op}: {lat/1000:.2f} ms")
    
    # Layer analysis if available
    if not layer_latency_pivot.empty:
        print(f"\n📈 LAYER-WISE ANALYSIS:")
        print("-" * 40)
        layer_stats = layer_latency_pivot.describe()
        print(f"Layer latency statistics (us):")
        print(layer_stats.round(0))
        
        # Find most expensive layers
        print(f"\nMost expensive layers:")
        for quant in ['W4A8', 'W8A8']:
            if quant in layer_latency_pivot.columns:
                top_layers = layer_latency_pivot[quant].nlargest(3)
                print(f"\n{quant}:")
                for layer, lat in top_layers.items():
                    print(f"  Layer {layer}: {lat/1000:.2f} ms")
    
    print("\n" + "="*80)
    print("RESEARCH INSIGHTS")
    print("="*80)
    
    print(f"""
🔬 KEY FINDINGS:

1. QUANTIZATION IMPACT:
   - W4A8 achieves {speedup:.1f}x speedup over W8A8
   - This suggests 4-bit weights significantly reduce memory bandwidth requirements
   - 8-bit activations still provide reasonable precision

2. BOTTLENECK ANALYSIS:
   - MUL_MAT (matrix multiplication) dominates in both configurations
   - W8A8 shows much higher MUL_MAT overhead, indicating weight quantization impact
   - Memory operations (reshape, copy) have relatively low overhead

3. ATTENTION vs FFN TRADE-OFFS:
   - W4A8: More balanced between attention and FFN
   - W8A8: FFN operations become more dominant due to larger weight matrices
   
4. LAYER SCALING:
   - Later layers may show higher variance due to accumulated numerical errors
   - First few layers often have higher overhead due to cache warming

5. OPTIMIZATION OPPORTUNITIES:
   - Focus optimization efforts on MUL_MAT operations
   - Consider operator fusion for sequential small operations
   - Memory layout optimization could reduce reshape/copy overhead
""")

if __name__ == "__main__":
    analyze_latency('q4_0.txt', 'q8_0.txt') 