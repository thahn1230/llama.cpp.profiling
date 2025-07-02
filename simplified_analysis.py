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
            
            results[quant_type] = {
                'categories': categories,
                'percentages': percentages,
                'total_time': total_time
            }
            
            print(f"\n{quant_type} Results:")
            print(f"Total time: {total_time/1000:.1f}ms")
            for cat in ['Q', 'K', 'V', 'O', 'Up', 'Gate', 'Down', 'Others']:
                if cat in percentages:
                    print(f"  {cat}: {percentages[cat]:.1f}% ({categories[cat]/1000:.1f}ms)")
                    
        except FileNotFoundError:
            print(f"Error: Could not find {filename}")
            return None
    
    return results

def create_bar_chart(results):
    """Create bar chart comparing W4A8 vs W8A8"""
    categories = ['Q', 'K', 'V', 'O', 'Up', 'Gate', 'Down', 'Others']
    
    w4a8_values = []
    w8a8_values = []
    
    for cat in categories:
        w4a8_values.append(results['W4A8']['percentages'].get(cat, 0))
        w8a8_values.append(results['W8A8']['percentages'].get(cat, 0))
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars1 = ax.bar(x - width/2, w4a8_values, width, label='W4A8', alpha=0.8, color='skyblue')
    bars2 = ax.bar(x + width/2, w8a8_values, width, label='W8A8', alpha=0.8, color='lightcoral')
    
    ax.set_xlabel('Operation Categories', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage of Total Time (%)', fontsize=12, fontweight='bold')
    ax.set_title('Operation Time Distribution: W4A8 vs W8A8\n(Quantized Operations + Others)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0.5:  # Only show labels for bars with significant height
                ax.annotate(f'{height:.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
    
    add_labels(bars1)
    add_labels(bars2)
    
    plt.tight_layout()
    plt.savefig('quantized_operations_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create summary table
    print("\n" + "="*60)
    print("QUANTIZED OPERATIONS ANALYSIS")
    print("="*60)
    
    print(f"\n{'Category':<10} {'W4A8':<12} {'W8A8':<12} {'Difference':<12}")
    print("-" * 50)
    
    for cat in categories:
        w4a8_val = results['W4A8']['percentages'].get(cat, 0)
        w8a8_val = results['W8A8']['percentages'].get(cat, 0)
        diff = w4a8_val - w8a8_val
        print(f"{cat:<10} {w4a8_val:>8.1f}%   {w8a8_val:>8.1f}%   {diff:>+8.1f}%")
    
    # Summary statistics
    w4a8_quant = sum(results['W4A8']['percentages'].get(cat, 0) for cat in categories[:-1])  # Exclude Others
    w8a8_quant = sum(results['W8A8']['percentages'].get(cat, 0) for cat in categories[:-1])
    
    print(f"\nQuantized Operations (Q+K+V+O+Up+Gate+Down):")
    print(f"  W4A8: {w4a8_quant:.1f}%")
    print(f"  W8A8: {w8a8_quant:.1f}%")
    print(f"\nOthers (Non-quantized operations):")
    print(f"  W4A8: {results['W4A8']['percentages'].get('Others', 0):.1f}%")
    print(f"  W8A8: {results['W8A8']['percentages'].get('Others', 0):.1f}%")

if __name__ == "__main__":
    results = analyze_files()
    if results:
        create_bar_chart(results) 