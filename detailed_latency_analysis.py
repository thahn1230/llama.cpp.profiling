#!/usr/bin/env python3
import re
import matplotlib.pyplot as plt
import numpy as np

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
        return 'Others'  # Non-MUL_MAT operations

def analyze_others_operations(operations):
    """Analyze what operations are in the Others category"""
    others_ops = {}
    others_total = 0
    
    for op in operations:
        category = categorize_operation(op['name'], op['type'])
        if category == 'Others':
            op_type = op['type']
            if op_type not in others_ops:
                others_ops[op_type] = {'count': 0, 'total_latency': 0, 'examples': []}
            others_ops[op_type]['count'] += 1
            others_ops[op_type]['total_latency'] += op['latency']
            others_total += op['latency']
            
            # Keep some examples
            if len(others_ops[op_type]['examples']) < 3:
                others_ops[op_type]['examples'].append(op['name'])
    
    return others_ops, others_total

def analyze_files():
    """Analyze both quantization files"""
    files = {
        'W4A8': 'q4_0-1.txt',
        'W8A8': 'q8_0-1.txt'
    }
    
    results = {}
    
    for quant_type, filename in files.items():
        try:
            operations = parse_latency_file(filename)
            
            # Group by operation category
            categories = {}
            for op in operations:
                category = categorize_operation(op['name'], op['type'])
                if category not in categories:
                    categories[category] = 0
                categories[category] += op['latency']
            
            # Convert to percentages
            total_time = sum(categories.values())
            percentages = {cat: (time / total_time) * 100 for cat, time in categories.items()}
            
            # Analyze Others operations
            others_ops, others_total = analyze_others_operations(operations)
            
            results[quant_type] = {
                'categories': categories,
                'percentages': percentages,
                'total_time': total_time,
                'others_breakdown': others_ops,
                'others_total': others_total
            }
            
        except FileNotFoundError:
            print(f"Error: Could not find {filename}")
            return None
    
    return results

def create_detailed_comparison(results):
    """Create detailed comparison with both latency and percentage"""
    categories = ['Q', 'K', 'V', 'O', 'Up', 'Gate', 'Down', 'Others']
    
    print("="*80)
    print("DETAILED LATENCY AND PERCENTAGE ANALYSIS")
    print("="*80)
    
    print(f"\n{'Category':<10} {'W4A8 (ms)':<12} {'W4A8 (%)':<10} {'W8A8 (ms)':<12} {'W8A8 (%)':<10} {'Latency Ratio':<15}")
    print("-" * 80)
    
    for cat in categories:
        w4a8_ms = results['W4A8']['categories'].get(cat, 0) / 1000
        w4a8_pct = results['W4A8']['percentages'].get(cat, 0)
        w8a8_ms = results['W8A8']['categories'].get(cat, 0) / 1000
        w8a8_pct = results['W8A8']['percentages'].get(cat, 0)
        
        if w4a8_ms > 0:
            ratio = w8a8_ms / w4a8_ms
        else:
            ratio = 0
            
        print(f"{cat:<10} {w4a8_ms:>8.1f}    {w4a8_pct:>6.1f}%   {w8a8_ms:>8.1f}    {w8a8_pct:>6.1f}%   {ratio:>8.1f}x")
    
    # Total comparison
    w4a8_total = results['W4A8']['total_time'] / 1000
    w8a8_total = results['W8A8']['total_time'] / 1000
    total_ratio = w8a8_total / w4a8_total
    
    print("-" * 80)
    print(f"{'TOTAL':<10} {w4a8_total:>8.1f}    {'100.0%':>6}   {w8a8_total:>8.1f}    {'100.0%':>6}   {total_ratio:>8.1f}x")
    
    return categories

def analyze_others_breakdown(results):
    """Detailed breakdown of Others category"""
    print("\n" + "="*60)
    print("OTHERS CATEGORY BREAKDOWN")
    print("="*60)
    
    for quant_type in ['W4A8', 'W8A8']:
        print(f"\n{quant_type} - Others Operations:")
        others_ops = results[quant_type]['others_breakdown']
        others_total = results[quant_type]['others_total']
        
        # Sort by latency
        sorted_ops = sorted(others_ops.items(), key=lambda x: x[1]['total_latency'], reverse=True)
        
        for op_type, data in sorted_ops:
            latency_ms = data['total_latency'] / 1000
            pct_of_others = (data['total_latency'] / others_total) * 100 if others_total > 0 else 0
            pct_of_total = (data['total_latency'] / results[quant_type]['total_time']) * 100
            
            print(f"  {op_type:<15}: {latency_ms:>6.1f}ms ({pct_of_others:>4.1f}% of Others, {pct_of_total:>4.1f}% of Total)")
            print(f"                   Count: {data['count']}, Examples: {', '.join(data['examples'][:3])}")

def create_latency_bar_chart(results, categories):
    """Create bar chart with latency values"""
    w4a8_values = []
    w8a8_values = []
    
    for cat in categories:
        w4a8_values.append(results['W4A8']['categories'].get(cat, 0) / 1000)  # Convert to ms
        w8a8_values.append(results['W8A8']['categories'].get(cat, 0) / 1000)
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Latency chart
    bars1 = ax1.bar(x - width/2, w4a8_values, width, label='W4A8', alpha=0.8, color='skyblue')
    bars2 = ax1.bar(x + width/2, w8a8_values, width, label='W8A8', alpha=0.8, color='lightcoral')
    
    ax1.set_xlabel('Operation Categories', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Operation Latency Comparison: W4A8 vs W8A8', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    def add_latency_labels(bars, ax):
        for bar in bars:
            height = bar.get_height()
            if height > 10:  # Only show labels for significant bars
                ax.annotate(f'{height:.0f}ms',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
    
    add_latency_labels(bars1, ax1)
    add_latency_labels(bars2, ax1)
    
    # Percentage chart
    w4a8_pct = [results['W4A8']['percentages'].get(cat, 0) for cat in categories]
    w8a8_pct = [results['W8A8']['percentages'].get(cat, 0) for cat in categories]
    
    bars3 = ax2.bar(x - width/2, w4a8_pct, width, label='W4A8', alpha=0.8, color='skyblue')
    bars4 = ax2.bar(x + width/2, w8a8_pct, width, label='W8A8', alpha=0.8, color='lightcoral')
    
    ax2.set_xlabel('Operation Categories', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Percentage of Total Time (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Operation Percentage Distribution', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add percentage labels
    def add_pct_labels(bars, ax):
        for bar in bars:
            height = bar.get_height()
            if height > 1:
                ax.annotate(f'{height:.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
    
    add_pct_labels(bars3, ax2)
    add_pct_labels(bars4, ax2)
    
    plt.tight_layout()
    plt.savefig('detailed_latency_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    results = analyze_files()
    if results:
        categories = create_detailed_comparison(results)
        analyze_others_breakdown(results)
        create_latency_bar_chart(results, categories) 