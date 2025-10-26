import sys
import os
import json

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'data_structure'))
sys.path.insert(0, os.path.join(project_root, 'utils'))

from data_structure.parameters import Parameters

# 加载配置文件
with open('test_config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

print('配置文件加载成功')
print('debug_level:', config['debug_level'])
print('input_file:', config['input_file'])
print('output_file:', config['output_file'])
print('mesh_type:', config['mesh_type'])
print('parts数量:', len(config['parts']))

if config['parts']:
    print('第一个部件名称:', config['parts'][0]['part_name'])
    print('第一个部件参数:', config['parts'][0]['part_params'])

# 测试Parameters类
params = Parameters("FROM_CASE_JSON", "test_config.json")
print('\nParameters对象创建成功')
print('params.debug_level:', params.debug_level)
print('params.input_file:', params.input_file)
print('params.part_params数量:', len(params.part_params))

if params.part_params:
    print('第一个part名称:', params.part_params[0].part_name)
    print('第一个part参数max_size:', params.part_params[0].part_params.max_size)
    print('第一个part参数first_height:', params.part_params[0].part_params.first_height)