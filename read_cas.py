import re

def parse_fluent_msh(file_path):
    data = {
        'nodes': [],
        'faces': [],
        'cells': [],
        'zones': {},
        'comments': [],
        'output_prompts': [],
        'dimensions': 0
    }

    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    current_section = None
    current_zone = None

    # 正则表达式模式
    hex_pattern = re.compile(r'[0-9a-fA-F]+')
    node_section_pattern = re.compile(r'\(10 \(1')
    face_section_pattern = re.compile(r'\(13 \((\d+)')
    # cell_section_pattern = re.compile(r'\(12 \((\d+)\s*(\d+)\s*(\d+)')
    cell_section_pattern = re.compile(r'\(\s*12\s*\(\s*(\d+)\s+(\d+)\s+([0-9A-Fa-f]+)\s+(\d+)\s+(\d+)')   
    bc_pattern = re.compile(r'^\(\s*45\s+\(\s*(\d+)\s+([\w-]+)\s+([\w-]+)\s*\)\s*\(\s*\)\s*\)$')
    # cell_type_pattern = re.compile(r'^\((\d+)\s+\(([^)]*)\)\s*\(\s*$')
    
    for line in lines:
        # 处理注释和输出提示
        if line.startswith('(0'):
            data['comments'].append(line[2:].strip())
            continue
        elif line.startswith('(1 ') and not line.startswith('(10 ') and not line.startswith('(12 ') and not line.startswith('(13 '):
            data['output_prompts'].append(line[2:].strip())
            continue

        # 处理维度信息
        if line.startswith('(2 '):
            # 使用正则表达式提取所有数字
            numbers = re.findall(r'\d+', line)
            if len(numbers) >= 2:
                data['dimensions'] = int(numbers[1])
            else:
                raise ValueError(f"Invalid dimension line: {line}")
            continue

        # 处理节点数量
        if line.startswith('(10 (0'):
            data['node_count'] = int(line.split()[3], 16)
            continue

        # 处理面数量
        if line.startswith('(13 (0'):
            data['face_count'] = int(line.split()[3], 16)
            continue

        # 处理单元数量
        if line.startswith('(12 (0'):
            data['cell_count'] = int(line.split()[3], 16)
            continue

        # 处理节点坐标
        if node_section_pattern.match(line):
            current_section = 'nodes'
            continue

        # 处理面数据
        face_match = face_section_pattern.match(line)
        if face_match:
            current_section = 'faces'
            zone_id = int(face_match.group(1))
            current_zone = {
                'zone_id': zone_id,
                'type': 'faces',
                'data': [],
                'left_cells': [],
                'right_cells': []
            }
            data['zones'][f'zone_{zone_id}'] = current_zone
            continue

        # 处理单元数据
        cell_match = cell_section_pattern.match(line)
        if cell_match:
            current_section = 'cells'
            zone_id = int(cell_match.group(1))
            cell_start_idx = int(cell_match.group(2),16)
            cell_end_idx = int(cell_match.group(3), 16)
            cell_count = cell_end_idx - cell_start_idx + 1
            current_zone = {
                'zone_id': zone_id,
                'type': 'cells',
                'cell_type': [],
                'cell_count': cell_count,
                'data': []
            }
            data['zones'][f'zone_{zone_id}'] = current_zone
            continue

        # 处理边界条件
        if line.startswith('(45'):
            match = bc_pattern.match(line)
            if match:
                zone_id = int(match.group(1))
                bc_type = match.group(2)
                bc_name = match.group(3).strip() if match.group(3) else None
               
                # 清理边界名称中的多余字符
                if bc_name:
                    bc_name = bc_name.split(')', 1)[0].strip()

                if f'zone_{zone_id}' in data['zones']:
                    zone = data['zones'][f'zone_{zone_id}']
                    zone['bc_type'] = bc_type
                    zone['bc_name'] = bc_name
                else:
                    print(f"Warning: Zone {zone_id} not found for BC: {line}")
            else:
                print(f"Warning: Unparsed BC line: {line}")
            continue            
                 
        # 处理当前section的数据
        if current_section == 'nodes':
            if line == '))':
                current_section = None
            else:
                coords = list(map(float, line.split()))
                for i in range(0, len(coords), data['dimensions']):
                    data['nodes'].append(coords[i:i+data['dimensions']])
            continue

        if current_section == 'faces':
            if line == '))':
                current_section = None
            else:
                # 处理十六进制面数据
                hex_values = hex_pattern.findall(line)
                if len(hex_values) == 4:
                    face = {
                        'node1': int(hex_values[0], 16),
                        'node2': int(hex_values[1], 16),
                        'left_cell': int(hex_values[2], 16),
                        'right_cell': int(hex_values[3], 16)
                    }
                elif len(hex_values) == 5:
                    face = {
                        'nnodes':int(hex_values[0], 16),
                        'node1': int(hex_values[1], 16),
                        'node2': int(hex_values[2], 16),
                        'left_cell': int(hex_values[3], 16),
                        'right_cell': int(hex_values[4], 16)
                    }                    
                current_zone['data'].append(face)
            continue

        if current_section == 'cells':
            if line == '))':
                current_section = None
            else:        
                hex_values = line.split()
                # 分离单元类型和节点数据
                for h in hex_values:
                    cell_type = int(h)
                    current_zone['cell_type'].append(cell_type)
            continue
        
    # # 后处理补充信息
    # for zone_id, zone in data['zones'].items():
    #     if zone['type'] == 'cells':
    #         zone['cell_count'] = len(zone['data'])
    #         if zone['data']:
    #             zone['nodes_per_cell'] = len(zone['data'][0])
    return data