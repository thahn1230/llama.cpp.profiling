import re
import matplotlib.pyplot as plt
import pandas as pd

def parse_latency_file_with_quant(filename):
    """Parse latency file and extract both tensor operations and quantization times"""
    operations = []
    quant_operations = []
    
    with open(filename, 'r') as f:
        content = f.read()
    
    # Parse tensor operations
    tensor_pattern = r"\[\[\[\[Latency for Tensor\]\]\]\] '([^']+)' \(([^)]+)\): (\d+) us"
    tensor_matches = re.findall(tensor_pattern, content)
    
    for match in tensor_matches:
        op_name, op_type, latency = match
        operations.append({
            'name': op_name,
            'type': op_type,
            'latency_us': int(latency)
        })
    
    # Parse quantization operations
    quant_pattern = r"\[QUANT\] Thread \d+: ([0-9.]+) ms, (\d+) elements, ([0-9.]+) MB/s \(src1: (.+) -> (.+)\)"
    quant_matches = re.findall(quant_pattern, content)
    
    for match in quant_matches:
        time_ms, elements, speed, src_type, dst_type = match
        quant_operations.append({
            'time_ms': float(time_ms),
            'time_us': float(time_ms) * 1000,  # Convert to microseconds
            'elements': int(elements),
            'speed_mbps': float(speed),
            'src_type': src_type,
            'dst_type': dst_type
        })
    
    return operations, quant_operations

def categorize_operation(op_name, op_type):
    """Categorize operations by type"""
    if 'Qcur' in op_name and op_type == 'MUL_MAT':
        return 'Q Projection'
    elif 'Kcur' in op_name and op_type == 'MUL_MAT':
        return 'K Projection'
    elif 'Vcur' in op_name and op_type == 'MUL_MAT':
        return 'V Projection'
    elif 'attn_out' in op_name and op_type == 'MUL_MAT':
        return 'O Projection'
    elif 'ffn_gate' in op_name and op_type == 'MUL_MAT':
        return 'FFN Gate'
    elif 'ffn_up' in op_name and op_type == 'MUL_MAT':
        return 'FFN Up'
    elif 'ffn_out' in op_name and op_type == 'MUL_MAT':
        return 'FFN Down'
    else:
        return 'Others'

def analyze_with_quantization():
    # Parse both files
    ops_q4, quant_q4 = parse_latency_file_with_quant('q4_0-t1-q.txt')
    ops_q8, quant_q8 = parse_latency_file_with_quant('q8_0-t1-q.txt')
    
    print("=== QUANTIZATION TIME ANALYSIS ===\n")
    
    # Analyze quantization operations
    total_quant_q4 = sum(q['time_us'] for q in quant_q4)
    total_quant_q8 = sum(q['time_us'] for q in quant_q8)
    
    print(f"W4A8 Quantization Operations: {len(quant_q4)}")
    print(f"W8A8 Quantization Operations: {len(quant_q8)}")
    print(f"W4A8 Total Quantization Time: {total_quant_q4:.0f} us ({total_quant_q4/1000:.2f} ms)")
    print(f"W8A8 Total Quantization Time: {total_quant_q8:.0f} us ({total_quant_q8/1000:.2f} ms)")
    print(f"Quantization Time Ratio (W8A8/W4A8): {total_quant_q8/total_quant_q4:.2f}x")
    
    # Analyze tensor operations (without quantization)
    total_tensor_q4 = sum(op['latency_us'] for op in ops_q4)
    total_tensor_q8 = sum(op['latency_us'] for op in ops_q8)
    
    print(f"\nW4A8 Total Tensor Operations: {total_tensor_q4:.0f} us ({total_tensor_q4/1000:.2f} ms)")
    print(f"W8A8 Total Tensor Operations: {total_tensor_q8:.0f} us ({total_tensor_q8/1000:.2f} ms)")
    print(f"Tensor Operations Ratio (W8A8/W4A8): {total_tensor_q8/total_tensor_q4:.2f}x")
    
    # Combined totals
    total_combined_q4 = total_tensor_q4 + total_quant_q4
    total_combined_q8 = total_tensor_q8 + total_quant_q8
    
    print(f"\nW4A8 Combined Total: {total_combined_q4:.0f} us ({total_combined_q4/1000:.2f} ms)")
    print(f"W8A8 Combined Total: {total_combined_q8:.0f} us ({total_combined_q8/1000:.2f} ms)")
    print(f"Combined Total Ratio (W8A8/W4A8): {total_combined_q8/total_combined_q4:.2f}x")
    
    # Percentage analysis
    quant_percent_q4 = (total_quant_q4 / total_combined_q4) * 100
    quant_percent_q8 = (total_quant_q8 / total_combined_q8) * 100
    
    print(f"\nQuantization Time Percentage:")
    print(f"W4A8: {quant_percent_q4:.2f}%")
    print(f"W8A8: {quant_percent_q8:.2f}%")
    
    # Quantization type analysis
    print(f"\n=== QUANTIZATION TYPE BREAKDOWN ===")
    
    # Group by quantization type
    quant_types_q4 = {}
    quant_types_q8 = {}
    
    for q in quant_q4:
        key = f"{q['src_type']} -> {q['dst_type']}"
        if key not in quant_types_q4:
            quant_types_q4[key] = {'count': 0, 'total_time': 0}
        quant_types_q4[key]['count'] += 1
        quant_types_q4[key]['total_time'] += q['time_us']
    
    for q in quant_q8:
        key = f"{q['src_type']} -> {q['dst_type']}"
        if key not in quant_types_q8:
            quant_types_q8[key] = {'count': 0, 'total_time': 0}
        quant_types_q8[key]['count'] += 1
        quant_types_q8[key]['total_time'] += q['time_us']
    
    print("W4A8 Quantization Types:")
    for qtype, data in quant_types_q4.items():
        print(f"  {qtype}: {data['count']} ops, {data['total_time']:.0f} us ({data['total_time']/1000:.2f} ms)")
    
    print("\nW8A8 Quantization Types:")
    for qtype, data in quant_types_q8.items():
        print(f"  {qtype}: {data['count']} ops, {data['total_time']:.0f} us ({data['total_time']/1000:.2f} ms)")
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Total time comparison (stacked bar)
    categories = ['W4A8', 'W8A8']
    tensor_times = [total_tensor_q4/1000, total_tensor_q8/1000]
    quant_times = [total_quant_q4/1000, total_quant_q8/1000]
    
    ax1.bar(categories, tensor_times, label='Tensor Operations', color='skyblue')
    ax1.bar(categories, quant_times, bottom=tensor_times, label='Quantization', color='orange')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Total Time Breakdown: Tensor Operations vs Quantization')
    ax1.legend()
    
    # Add percentage labels
    for i, (tensor, quant) in enumerate(zip(tensor_times, quant_times)):
        total = tensor + quant
        ax1.text(i, total/2, f'{tensor:.1f}ms\n({tensor/total*100:.1f}%)', 
                ha='center', va='center', fontweight='bold')
        ax1.text(i, tensor + quant/2, f'{quant:.1f}ms\n({quant/total*100:.1f}%)', 
                ha='center', va='center', fontweight='bold')
    
    # 2. Quantization percentage pie chart
    labels = ['Tensor Ops', 'Quantization']
    
    # W4A8 pie
    sizes_q4 = [100-quant_percent_q4, quant_percent_q4]
    ax2.pie(sizes_q4, labels=labels, autopct='%1.1f%%', startangle=90)
    ax2.set_title('W4A8: Time Distribution')
    
    # 3. W8A8 pie
    sizes_q8 = [100-quant_percent_q8, quant_percent_q8]
    ax3.pie(sizes_q8, labels=labels, autopct='%1.1f%%', startangle=90)
    ax3.set_title('W8A8: Time Distribution')
    
    # 4. Quantization time by type
    # Combine all unique quantization types
    all_types = set(quant_types_q4.keys()) | set(quant_types_q8.keys())
    type_names = list(all_types)
    q4_times = [quant_types_q4.get(t, {'total_time': 0})['total_time']/1000 for t in type_names]
    q8_times = [quant_types_q8.get(t, {'total_time': 0})['total_time']/1000 for t in type_names]
    
    x = range(len(type_names))
    width = 0.35
    
    ax4.bar([i - width/2 for i in x], q4_times, width, label='W4A8', color='lightblue')
    ax4.bar([i + width/2 for i in x], q8_times, width, label='W8A8', color='lightcoral')
    ax4.set_xlabel('Quantization Type')
    ax4.set_ylabel('Time (ms)')
    ax4.set_title('Quantization Time by Type')
    ax4.set_xticks(x)
    ax4.set_xticklabels(type_names, rotation=45, ha='right')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('quantization_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Detailed layer analysis
    print(f"\n=== LAYER-BY-LAYER QUANTIZATION ANALYSIS ===")
    
    # Group quantization by layer (assuming they appear in sequence)
    layer_quant_q4 = {}
    layer_quant_q8 = {}
    
    # Simple heuristic: assign quantization operations to layers based on order
    # This assumes quantization operations appear near their corresponding tensor operations
    
    # For now, let's just show the impact on main operation categories
    print("\nImpact of adding quantization time to main operation categories:")
    
    # Categorize tensor operations
    categories_q4 = {}
    categories_q8 = {}
    
    for op in ops_q4:
        cat = categorize_operation(op['name'], op['type'])
        if cat not in categories_q4:
            categories_q4[cat] = 0
        categories_q4[cat] += op['latency_us']
    
    for op in ops_q8:
        cat = categorize_operation(op['name'], op['type'])
        if cat not in categories_q8:
            categories_q8[cat] = 0
        categories_q8[cat] += op['latency_us']
    
    # Compare with and without quantization
    print(f"\nOperation breakdown (without quantization):")
    print(f"{'Category':<15} {'W4A8 (ms)':<12} {'W8A8 (ms)':<12} {'Ratio':<8}")
    print("-" * 50)
    
    for cat in sorted(set(categories_q4.keys()) | set(categories_q8.keys())):
        q4_time = categories_q4.get(cat, 0) / 1000
        q8_time = categories_q8.get(cat, 0) / 1000
        ratio = q8_time / q4_time if q4_time > 0 else 0
        print(f"{cat:<15} {q4_time:<12.2f} {q8_time:<12.2f} {ratio:<8.2f}")
    
    print(f"\nTotal quantization overhead:")
    print(f"W4A8: +{total_quant_q4/1000:.2f} ms ({quant_percent_q4:.1f}% overhead)")
    print(f"W8A8: +{total_quant_q8/1000:.2f} ms ({quant_percent_q8:.1f}% overhead)")

if __name__ == "__main__":
    analyze_with_quantization() 