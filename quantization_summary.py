import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

def create_comprehensive_analysis():
    # Parse quantization files
    ops_q4_quant, quant_q4 = parse_latency_file_with_quant('q4_0-t1-q.txt')
    ops_q8_quant, quant_q8 = parse_latency_file_with_quant('q8_0-t1-q.txt')
    
    # Calculate totals
    total_tensor_q4 = sum(op['latency_us'] for op in ops_q4_quant)
    total_tensor_q8 = sum(op['latency_us'] for op in ops_q8_quant)
    total_quant_q4 = sum(q['time_us'] for q in quant_q4)
    total_quant_q8 = sum(q['time_us'] for q in quant_q8)
    
    # Combined totals
    total_combined_q4 = total_tensor_q4 + total_quant_q4
    total_combined_q8 = total_tensor_q8 + total_quant_q8
    
    print("=== COMPREHENSIVE QUANTIZATION IMPACT ANALYSIS ===\n")
    
    print("📊 TIMING COMPARISON:")
    print(f"{'Metric':<25} {'W4A8':<15} {'W8A8':<15} {'Ratio (W8A8/W4A8)':<20}")
    print("-" * 75)
    print(f"{'Tensor Operations':<25} {total_tensor_q4/1000:<15.2f} {total_tensor_q8/1000:<15.2f} {total_tensor_q8/total_tensor_q4:<20.2f}")
    print(f"{'Quantization Time':<25} {total_quant_q4/1000:<15.2f} {total_quant_q8/1000:<15.2f} {total_quant_q8/total_quant_q4:<20.2f}")
    print(f"{'Combined Total':<25} {total_combined_q4/1000:<15.2f} {total_combined_q8/1000:<15.2f} {total_combined_q8/total_combined_q4:<20.2f}")
    
    print(f"\n🔍 QUANTIZATION OVERHEAD:")
    quant_percent_q4 = (total_quant_q4 / total_combined_q4) * 100
    quant_percent_q8 = (total_quant_q8 / total_combined_q8) * 100
    print(f"W4A8: {total_quant_q4/1000:.2f} ms ({quant_percent_q4:.2f}% of total time)")
    print(f"W8A8: {total_quant_q8/1000:.2f} ms ({quant_percent_q8:.2f}% of total time)")
    
    print(f"\n⚡ SPEEDUP ANALYSIS:")
    print(f"Without quantization overhead: {total_tensor_q8/total_tensor_q4:.2f}x slower")
    print(f"With quantization overhead: {total_combined_q8/total_combined_q4:.2f}x slower")
    print(f"Quantization impact on speedup: {((total_combined_q8/total_combined_q4) - (total_tensor_q8/total_tensor_q4)):.3f}x")
    
    # Categorize operations
    categories_q4 = {}
    categories_q8 = {}
    
    for op in ops_q4_quant:
        cat = categorize_operation(op['name'], op['type'])
        if cat not in categories_q4:
            categories_q4[cat] = 0
        categories_q4[cat] += op['latency_us']
    
    for op in ops_q8_quant:
        cat = categorize_operation(op['name'], op['type'])
        if cat not in categories_q8:
            categories_q8[cat] = 0
        categories_q8[cat] += op['latency_us']
    
    print(f"\n📈 OPERATION BREAKDOWN (ms):")
    print(f"{'Category':<15} {'W4A8 Tensor':<12} {'W8A8 Tensor':<12} {'Ratio':<8} {'W4A8+Quant':<12} {'W8A8+Quant':<12} {'New Ratio':<10}")
    print("-" * 95)
    
    # Add quantization proportionally to categories (simplified approach)
    quant_per_op_q4 = total_quant_q4 / len(categories_q4) if categories_q4 else 0
    quant_per_op_q8 = total_quant_q8 / len(categories_q8) if categories_q8 else 0
    
    for cat in sorted(set(categories_q4.keys()) | set(categories_q8.keys())):
        q4_time = categories_q4.get(cat, 0) / 1000
        q8_time = categories_q8.get(cat, 0) / 1000
        ratio = q8_time / q4_time if q4_time > 0 else 0
        
        q4_with_quant = q4_time + (quant_per_op_q4 / 1000)
        q8_with_quant = q8_time + (quant_per_op_q8 / 1000)
        new_ratio = q8_with_quant / q4_with_quant if q4_with_quant > 0 else 0
        
        print(f"{cat:<15} {q4_time:<12.2f} {q8_time:<12.2f} {ratio:<8.2f} {q4_with_quant:<12.2f} {q8_with_quant:<12.2f} {new_ratio:<10.2f}")
    
    # Quantization type analysis
    print(f"\n🔧 QUANTIZATION TYPE ANALYSIS:")
    
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
    
    print("W4A8:")
    for qtype, data in quant_types_q4.items():
        avg_time = data['total_time'] / data['count']
        print(f"  {qtype}: {data['count']} ops, {data['total_time']/1000:.2f}ms total, {avg_time:.1f}μs avg")
    
    print("W8A8:")
    for qtype, data in quant_types_q8.items():
        avg_time = data['total_time'] / data['count']
        print(f"  {qtype}: {data['count']} ops, {data['total_time']/1000:.2f}ms total, {avg_time:.1f}μs avg")
    
    # Create comprehensive visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Updated total time comparison with quantization
    methods = ['W4A8\n(Tensor Only)', 'W4A8\n(+Quantization)', 'W8A8\n(Tensor Only)', 'W8A8\n(+Quantization)']
    times = [total_tensor_q4/1000, total_combined_q4/1000, total_tensor_q8/1000, total_combined_q8/1000]
    colors = ['lightblue', 'blue', 'lightcoral', 'red']
    
    bars = ax1.bar(methods, times, color=colors)
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Total Execution Time: Impact of Quantization Overhead')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, time in zip(bars, times):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{time:.1f}ms', ha='center', va='bottom', fontweight='bold')
    
    # Add speedup annotations
    ax1.text(1.5, max(times) * 0.8, f'Tensor Only Speedup:\n{total_tensor_q8/total_tensor_q4:.2f}x slower',
             ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    ax1.text(1.5, max(times) * 0.6, f'With Quantization:\n{total_combined_q8/total_combined_q4:.2f}x slower',
             ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.7))
    
    # 2. Quantization overhead breakdown
    categories = ['W4A8', 'W8A8']
    tensor_times = [total_tensor_q4/1000, total_tensor_q8/1000]
    quant_times = [total_quant_q4/1000, total_quant_q8/1000]
    
    width = 0.6
    ax2.bar(categories, tensor_times, width, label='Tensor Operations', color=['lightblue', 'lightcoral'])
    ax2.bar(categories, quant_times, width, bottom=tensor_times, label='Quantization', color=['blue', 'red'], alpha=0.7)
    
    ax2.set_ylabel('Time (ms)')
    ax2.set_title('Quantization Overhead Breakdown')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add percentage labels
    for i, (tensor, quant) in enumerate(zip(tensor_times, quant_times)):
        total = tensor + quant
        quant_pct = quant / total * 100
        ax2.text(i, total + 50, f'{quant_pct:.2f}%\noverhead', ha='center', va='bottom', fontweight='bold')
    
    # 3. Quantization operations comparison
    quant_counts = [len(quant_q4), len(quant_q8)]
    ax3.bar(['W4A8', 'W8A8'], quant_counts, color=['lightblue', 'lightcoral'])
    ax3.set_ylabel('Number of Quantization Operations')
    ax3.set_title('Quantization Operation Count')
    ax3.grid(True, alpha=0.3)
    
    for i, count in enumerate(quant_counts):
        ax3.text(i, count + 10, str(count), ha='center', va='bottom', fontweight='bold')
    
    # 4. Per-operation category impact
    categories_list = sorted(set(categories_q4.keys()) | set(categories_q8.keys()))
    x = np.arange(len(categories_list))
    width = 0.35
    
    q4_times = [categories_q4.get(cat, 0)/1000 for cat in categories_list]
    q8_times = [categories_q8.get(cat, 0)/1000 for cat in categories_list]
    
    ax4.bar(x - width/2, q4_times, width, label='W4A8', color='lightblue')
    ax4.bar(x + width/2, q8_times, width, label='W8A8', color='lightcoral')
    
    ax4.set_xlabel('Operation Category')
    ax4.set_ylabel('Time (ms)')
    ax4.set_title('Operation Time by Category (Tensor Operations Only)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories_list, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('quantization_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📝 KEY FINDINGS:")
    print(f"1. W8A8 requires {len(quant_q8)} quantization operations vs {len(quant_q4)} for W4A8")
    print(f"2. Quantization overhead is minimal: {quant_percent_q4:.2f}% for W4A8, {quant_percent_q8:.2f}% for W8A8")
    print(f"3. Main performance difference comes from tensor operations, not quantization")
    print(f"4. W8A8 is {total_combined_q8/total_combined_q4:.2f}x slower than W4A8 (including quantization)")
    print(f"5. Quantization types: W4A8 uses mainly f32->f16, W8A8 uses mainly f32->q8_0")

if __name__ == "__main__":
    create_comprehensive_analysis() 