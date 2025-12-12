import numpy as np
import json
import os

# 转换 .npy 文件到 JSON
npy_path = 'Result/gradient_analysis/gradients_last_round.npy'
json_path = 'Result/gradient_analysis/gradients_last_round.json'

try:
    # 加载 .npy 文件
    data = np.load(npy_path, allow_pickle=True).item()
    
    # 转换为可 JSON 序列化的格式
    json_data = {}
    
    # 处理恶意梯度
    if data['malicious']:
        json_data['malicious'] = [g.tolist() if g is not None else None for g in data['malicious']]
    else:
        json_data['malicious'] = []
    
    # 处理良性梯度
    if data['benign']:
        json_data['benign'] = [g.tolist() if g is not None else None for g in data['benign']]
    else:
        json_data['benign'] = []
    
    # 处理选中梯度
    if data['selected'] is not None:
        if isinstance(data['selected'], list):
            json_data['selected'] = [g.tolist() if g is not None else None for g in data['selected']]
        else:
            json_data['selected'] = data['selected'].tolist()
    else:
        json_data['selected'] = None
    
    # 保存为 JSON
    with open(json_path, 'w') as f:
        json.dump(json_data, f)
    
    print('✓ 转换成功！')
    print(f'  恶意梯度: {len(json_data["malicious"])} 个')
    print(f'  良性梯度: {len(json_data["benign"])} 个')
    print(f'  选中梯度: {json_data["selected"]}')
    print(f'  保存到: {json_path}')
    
except FileNotFoundError:
    print(f'✗ 文件不存在: {npy_path}')
except Exception as e:
    print(f'✗ 错误: {e}')
    import traceback
    traceback.print_exc()
