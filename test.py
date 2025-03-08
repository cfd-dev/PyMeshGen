from read_cas import parse_fluent_msh

file_path = './sample/tri.cas'
data = parse_fluent_msh(file_path)

print(f"Dimensions: {data['dimensions']}")
print(f"Nodes: {len(data['nodes'])}")
print(f"Zones: {len(data['zones'])}")

# 打印第一个面区域的信息
face_zone = data['zones'].get('zone_3')
if face_zone:
    print(f"\nFace Zone {face_zone['zone_id']} ({face_zone['bc_type'] if 'bc_type' in face_zone else 'interior'}):")
    print(f"First 2 faces:")
    for face in face_zone['data'][:2]:
        print(f"Nodes: {face['node1']}->{face['node2']}, Cells: {face['left_cell']}|{face['right_cell']}")