# PyMeshGen [![License](https://img.shields.io/badge/License-GPLv2+-brightgreen.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.html)

![Mesh Example](./docs/images/demo_mesh.png)

## 项目概述
开源的python非结构网格生成工具，专注于为计算流体力学或有限元分析(CFD/FEA)提供易用的二维网格生成解决方案，同时集成了多种主流网格生成算法供研究学习。

## 项目发起人：
- 王年华，nianhuawong@qq.com
  
## 主要特性
- **输入输出**
  - 支持导入Fluent `.cas` 网格格式
  - 导入导出VTK `.vtk` 可视化格式
  - 导入导出`.stl`网格格式
- **核心算法**
  - 二维阵面推进法（Advancing Front）
  - 边界层推进技术（Advancing Layer）
  - 四叉树背景网格尺寸控制
- **网格类型**
  - 三角形各向同性网格
  - 四边形边界层网格

## 快速开始
```bash
# 安装依赖
pip install -r requirements.txt

# 生成示例网格
python PyMeshGen.py --case "./config/30p30n.json"
```

## 开发团队
- **项目发起人**: Nianhua Wang <nianhuawong@qq.com>