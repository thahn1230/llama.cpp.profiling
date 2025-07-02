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
        return 'Others'  # All non-MUL_MAT operations

def analyze_file(filename):
    """Analyze a single latency file"""
    operations = parse_latency_file(filename)
    
    # Group by category
    categories = {}
    for op in operations:
        category = categorize_operation(op['name'], op['type'])
        if category not in categories:
            categories[category] = 0
        categories[category] += op['latency']
    
    return categories

def main():
    # Analyze both files
    w4a8_data = analyze_file('q4_0-1-t1.txt')
    w8a8_data = analyze_file('q8_0-1-t1.txt')
    
    # Get all categories and ensure consistent ordering
    all_categories = ['Q', 'K', 'V', 'O', 'Up', 'Gate', 'Down', 'Others']
    
    # Prepare data for plotting
    w4a8_values = [w4a8_data.get(cat, 0) / 1000.0 for cat in all_categories]  # Convert to ms
    w8a8_values = [w8a8_data.get(cat, 0) / 1000.0 for cat in all_categories]  # Convert to ms
    
    # Create the comparison bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(all_categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, w4a8_values, width, label='W4A8', color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x + width/2, w8a8_values, width, label='W8A8', color='#A23B72', alpha=0.8)
    
    # Add value labels on bars
    def add_value_labels(bars, values):
        for bar, value in zip(bars, values):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + max(w4a8_values + w8a8_values) * 0.01,
                       f'{value:.1f}ms', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    add_value_labels(bars1, w4a8_values)
    add_value_labels(bars2, w8a8_values)
    
    # Customize the chart
    ax.set_xlabel('Operation Categories', fontsize=14, fontweight='bold')
    ax.set_ylabel('Latency (ms)', fontsize=14, fontweight='bold')
    ax.set_title('W4A8 vs W8A8 Latency Comparison by Operation Category', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(all_categories, fontsize=12)
    ax.legend(fontsize=12, loc='upper right')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)
    
    # Make the plot look more professional
    plt.tight_layout()
    
    # Add summary statistics
    total_w4a8 = sum(w4a8_values)
    total_w8a8 = sum(w8a8_values)
    speedup = total_w8a8 / total_w4a8 if total_w4a8 > 0 else 0
    
    # Add text box with summary
    textstr = f'Total Latency:\nW4A8: {total_w4a8:.1f}ms\nW8A8: {total_w8a8:.1f}ms\nSpeedup: {speedup:.1f}x'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    # Save the plot
    plt.savefig('latency_comparison_t1.png', dpi=300, bbox_inches='tight')
    print("✅ Latency comparison chart saved as 'latency_comparison_t1.png'")
    
    # Print detailed comparison
    print("\n" + "="*80)
    print("DETAILED LATENCY COMPARISON (T1 FILES)")
    print("="*80)
    
    print(f"\n{'Category':<10} {'W4A8 (ms)':<12} {'W8A8 (ms)':<12} {'Ratio (W8A8/W4A8)':<20}")
    print("-" * 60)
    
    for i, cat in enumerate(all_categories):
        w4a8_val = w4a8_values[i]
        w8a8_val = w8a8_values[i]
        ratio = w8a8_val / w4a8_val if w4a8_val > 0 else float('inf')
        print(f"{cat:<10} {w4a8_val:<12.1f} {w8a8_val:<12.1f} {ratio:<20.1f}")
    
    print("-" * 60)
    print(f"{'TOTAL':<10} {total_w4a8:<12.1f} {total_w8a8:<12.1f} {speedup:<20.1f}")
    
    print(f"\n🚀 W4A8 is {speedup:.1f}x faster than W8A8")
    print(f"📊 W4A8 total: {total_w4a8:.1f}ms vs W8A8 total: {total_w8a8:.1f}ms")

if __name__ == "__main__":
    main() 