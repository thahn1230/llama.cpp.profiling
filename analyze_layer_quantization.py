import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def parse_sequential_operations(filename):
    """Parse operations in sequential order to track which quantization belongs to which operation"""
    operations = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    current_pos = 0
    for line in lines:
        line = line.strip()
        
        # Parse tensor operation
        tensor_match = re.match(r"\[\[\[\[Latency for Tensor\]\]\]\] '([^']+)' \(([^)]+)\): (\d+) us", line)
        if tensor_match:
            op_name, op_type, latency = tensor_match.groups()
            operations.append({
                'type': 'tensor',
                'name': op_name,
                'op_type': op_type,
                'latency_us': int(latency),
                'position': current_pos
            })
            current_pos += 1
        
        # Parse quantization operation
        quant_match = re.match(r"\[QUANT\] Thread \d+: ([0-9.]+) ms, (\d+) elements, ([0-9.]+) MB/s \(src1: (.+) -> (.+)\)", line)
        if quant_match:
            time_ms, elements, speed, src_type, dst_type = quant_match.groups()
            operations.append({
                'type': 'quantization',
                'time_ms': float(time_ms),
                'time_us': float(time_ms) * 1000,
                'elements': int(elements),
                'speed_mbps': float(speed),
                'src_type': src_type,
                'dst_type': dst_type,
                'position': current_pos
            })
            current_pos += 1
    
    return operations

def extract_layer_number(op_name):
    """Extract layer number from operation name"""
    # Look for patterns like "Qcur-0", "ffn_gate-1", etc.
    match = re.search(r'-(\d+)', op_name)
    if match:
        return int(match.group(1))
    return -1  # For operations without layer number

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

def analyze_layer_quantization():
    # Parse both files sequentially
    ops_q4 = parse_sequential_operations('q4_0-t1-q.txt')
    ops_q8 = parse_sequential_operations('q8_0-t1-q.txt')
    
    print("=== LAYER-BY-LAYER QUANTIZATION ANALYSIS ===\n")
    
    # Analyze layer distribution
    def analyze_by_layer(ops, name):
        print(f"\n{name} Analysis:")
        
        layer_data = {}
        layer_quant = {}
        
        for i, op in enumerate(ops):
            if op['type'] == 'tensor':
                layer = extract_layer_number(op['name'])
                category = categorize_operation(op['name'], op['op_type'])
                
                if layer not in layer_data:
                    layer_data[layer] = {}
                if category not in layer_data[layer]:
                    layer_data[layer][category] = {'tensor_time': 0, 'quant_time': 0, 'count': 0}
                
                layer_data[layer][category]['tensor_time'] += op['latency_us']
                layer_data[layer][category]['count'] += 1
                
                # Look for quantization operations that follow this tensor operation
                # Check next few operations for quantization
                for j in range(i+1, min(i+5, len(ops))):
                    if ops[j]['type'] == 'quantization':
                        layer_data[layer][category]['quant_time'] += ops[j]['time_us']
                    elif ops[j]['type'] == 'tensor':
                        break  # Stop at next tensor operation
            
            elif op['type'] == 'quantization':
                # Also track quantization by layer position estimation
                layer = -1  # Will try to infer from context
                if layer not in layer_quant:
                    layer_quant[layer] = {'total_time': 0, 'count': 0}
                layer_quant[layer]['total_time'] += op['time_us']
                layer_quant[layer]['count'] += 1
        
        # Print layer analysis
        print(f"Layers found: {sorted([l for l in layer_data.keys() if l >= 0])}")
        
        # Calculate totals by layer
        for layer in sorted(layer_data.keys()):
            if layer >= 0:  # Valid layers only
                print(f"\nLayer {layer}:")
                total_tensor = sum(cat['tensor_time'] for cat in layer_data[layer].values())
                total_quant = sum(cat['quant_time'] for cat in layer_data[layer].values())
                print(f"  Total Tensor Time: {total_tensor/1000:.2f} ms")
                print(f"  Total Quantization Time: {total_quant/1000:.2f} ms")
                print(f"  Quantization Overhead: {total_quant/total_tensor*100 if total_tensor > 0 else 0:.2f}%")
                
                for category, data in layer_data[layer].items():
                    if data['tensor_time'] > 0:
                        quant_percent = data['quant_time'] / data['tensor_time'] * 100 if data['tensor_time'] > 0 else 0
                        print(f"    {category}: {data['tensor_time']/1000:.2f}ms tensor + {data['quant_time']/1000:.2f}ms quant ({quant_percent:.1f}% overhead)")
        
        return layer_data
    
    layer_data_q4 = analyze_by_layer(ops_q4, "W4A8")
    layer_data_q8 = analyze_by_layer(ops_q8, "W8A8")
    
    # Detailed operation sequence analysis
    print(f"\n=== OPERATION SEQUENCE ANALYSIS ===")
    
    def print_operation_sequence(ops, name, max_ops=50):
        print(f"\n{name} - First {max_ops} operations:")
        print(f"{'#':<4} {'Type':<12} {'Operation':<25} {'Time (us)':<10} {'Category':<15}")
        print("-" * 80)
        
        for i, op in enumerate(ops[:max_ops]):
            if op['type'] == 'tensor':
                category = categorize_operation(op['name'], op['op_type'])
                layer = extract_layer_number(op['name'])
                op_display = f"{op['name']} (L{layer})" if layer >= 0 else op['name']
                print(f"{i:<4} {'Tensor':<12} {op_display:<25} {op['latency_us']:<10} {category:<15}")
            else:
                quant_display = f"{op['src_type']}->{op['dst_type']}"
                print(f"{i:<4} {'Quantization':<12} {quant_display:<25} {op['time_us']:<10.1f} {'Quant':<15}")
    
    print_operation_sequence(ops_q4, "W4A8")
    print_operation_sequence(ops_q8, "W8A8")
    
    # Quantization pattern analysis
    print(f"\n=== QUANTIZATION PATTERN ANALYSIS ===")
    
    def analyze_quant_patterns(ops, name):
        print(f"\n{name} Quantization Patterns:")
        
        tensor_ops = [op for op in ops if op['type'] == 'tensor']
        quant_ops = [op for op in ops if op['type'] == 'quantization']
        
        print(f"Total Tensor Operations: {len(tensor_ops)}")
        print(f"Total Quantization Operations: {len(quant_ops)}")
        print(f"Quantization Ratio: {len(quant_ops)/len(tensor_ops):.2f} quant ops per tensor op")
        
        # Group quantization by type
        quant_by_type = {}
        for op in quant_ops:
            key = f"{op['src_type']} -> {op['dst_type']}"
            if key not in quant_by_type:
                quant_by_type[key] = {'count': 0, 'total_time': 0, 'avg_time': 0}
            quant_by_type[key]['count'] += 1
            quant_by_type[key]['total_time'] += op['time_us']
        
        for key, data in quant_by_type.items():
            data['avg_time'] = data['total_time'] / data['count']
            print(f"  {key}: {data['count']} ops, {data['total_time']/1000:.2f}ms total, {data['avg_time']:.1f}us avg")
    
    analyze_quant_patterns(ops_q4, "W4A8")
    analyze_quant_patterns(ops_q8, "W8A8")
    
    # Create comprehensive visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
    
    # 1. Quantization timing distribution across operations
    tensor_ops_q4 = [op for op in ops_q4 if op['type'] == 'tensor'][:100]  # First 100 tensor ops
    quant_positions_q4 = []
    for op in ops_q4:
        if op['type'] == 'quantization':
            quant_positions_q4.append(op['position'])
    
    tensor_ops_q8 = [op for op in ops_q8 if op['type'] == 'tensor'][:100]
    quant_positions_q8 = []
    for op in ops_q8:
        if op['type'] == 'quantization':
            quant_positions_q8.append(op['position'])
    
    ax1.scatter(quant_positions_q4, [1]*len(quant_positions_q4), alpha=0.6, label='W4A8', s=20)
    ax1.scatter(quant_positions_q8, [2]*len(quant_positions_q8), alpha=0.6, label='W8A8', s=20)
    ax1.set_xlabel('Operation Position')
    ax1.set_ylabel('Method')
    ax1.set_title('Quantization Operation Positions in Sequence')
    ax1.set_yticks([1, 2])
    ax1.set_yticklabels(['W4A8', 'W8A8'])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Layer-wise quantization overhead
    valid_layers = [l for l in layer_data_q4.keys() if l >= 0][:10]  # First 10 layers
    if valid_layers:
        layer_overhead_q4 = []
        layer_overhead_q8 = []
        
        for layer in valid_layers:
            total_tensor_q4 = sum(cat['tensor_time'] for cat in layer_data_q4.get(layer, {}).values())
            total_quant_q4 = sum(cat['quant_time'] for cat in layer_data_q4.get(layer, {}).values())
            overhead_q4 = total_quant_q4 / total_tensor_q4 * 100 if total_tensor_q4 > 0 else 0
            layer_overhead_q4.append(overhead_q4)
            
            total_tensor_q8 = sum(cat['tensor_time'] for cat in layer_data_q8.get(layer, {}).values())
            total_quant_q8 = sum(cat['quant_time'] for cat in layer_data_q8.get(layer, {}).values())
            overhead_q8 = total_quant_q8 / total_tensor_q8 * 100 if total_tensor_q8 > 0 else 0
            layer_overhead_q8.append(overhead_q8)
        
        x = np.arange(len(valid_layers))
        width = 0.35
        
        ax2.bar(x - width/2, layer_overhead_q4, width, label='W4A8', color='lightblue')
        ax2.bar(x + width/2, layer_overhead_q8, width, label='W8A8', color='lightcoral')
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Quantization Overhead (%)')
        ax2.set_title('Quantization Overhead by Layer')
        ax2.set_xticks(x)
        ax2.set_xticklabels(valid_layers)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. Quantization operation frequency
    quant_ops_q4 = [op for op in ops_q4 if op['type'] == 'quantization']
    quant_ops_q8 = [op for op in ops_q8 if op['type'] == 'quantization']
    
    times_q4 = [op['time_us'] for op in quant_ops_q4]
    times_q8 = [op['time_us'] for op in quant_ops_q8]
    
    ax3.hist(times_q4, bins=20, alpha=0.7, label='W4A8', color='lightblue')
    ax3.hist(times_q8, bins=20, alpha=0.7, label='W8A8', color='lightcoral')
    ax3.set_xlabel('Quantization Time (us)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Quantization Operation Times')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Cumulative quantization time
    cumsum_q4 = np.cumsum([op['time_us'] if op['type'] == 'quantization' else 0 for op in ops_q4])
    cumsum_q8 = np.cumsum([op['time_us'] if op['type'] == 'quantization' else 0 for op in ops_q8])
    
    positions_q4 = list(range(len(ops_q4)))
    positions_q8 = list(range(len(ops_q8)))
    
    ax4.plot(positions_q4, cumsum_q4/1000, label='W4A8', linewidth=2)
    ax4.plot(positions_q8, cumsum_q8/1000, label='W8A8', linewidth=2)
    ax4.set_xlabel('Operation Position')
    ax4.set_ylabel('Cumulative Quantization Time (ms)')
    ax4.set_title('Cumulative Quantization Time Through Execution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('layer_quantization_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    analyze_layer_quantization() 