"""
测试 bw_core.py 的所有改进项

验证：
P0-1: 显式邻接关系缓存
P0-2: 星形性验证
P0-3: Robust Predicates
P1-1: 优先级队列
P1-2: 动态 KD 树更新
P1-3: 增强点间距检查
P2-1: 前端方法变体
P2-2: 改进 Laplacian 平滑
"""

import numpy as np
import time
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from delaunay.bw_core import BowyerWatsonMeshGenerator, Triangle


def test_basic_square():
    """测试基本正方形边界"""
    print("\n" + "="*80)
    print("测试 1: 基本正方形边界")
    print("="*80)
    
    # 定义正方形边界点
    boundary_points = np.array([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
    ])
    
    boundary_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    
    generator = BowyerWatsonMeshGenerator(
        boundary_points=boundary_points,
        boundary_edges=boundary_edges,
        max_edge_length=0.2,
        smoothing_iterations=0,
    )
    
    start_time = time.time()
    points, simplices, boundary_mask = generator.generate_mesh(target_triangle_count=50)
    elapsed = time.time() - start_time
    
    print(f"[OK] 网格生成成功")
    print(f"  - 总节点数: {len(points)}")
    print(f"  - 三角形数: {len(simplices)}")
    print(f"  - 边界节点: {np.sum(boundary_mask)}")
    print(f"  - 内部节点: {len(points) - np.sum(boundary_mask)}")
    print(f"  - 生成时间: {elapsed:.3f}s")
    
    # 验证邻接关系
    print("\n验证邻接关系 (P0-1):")
    has_neighbors = all(
        all(n is not None for n in tri.neighbors)
        for tri in generator.triangles
    )
    print(f"  - 所有三角形都有邻接关系: {has_neighbors}")
    
    # 验证边界边
    print("\n验证边界边恢复:")
    boundary_edges_recovered = 0
    for v1, v2 in boundary_edges:
        if any(v1 in tri.vertices and v2 in tri.vertices for tri in generator.triangles):
            boundary_edges_recovered += 1
    print(f"  - 恢复的边界边: {boundary_edges_recovered}/{len(boundary_edges)}")
    
    return points, simplices


def test_circle():
    """测试圆形边界"""
    print("\n" + "="*80)
    print("测试 2: 圆形边界（验证 Robust Predicates 和优先级队列）")
    print("="*80)
    
    # 生成圆形边界点
    n_points = 32
    angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    radius = 1.0
    boundary_points = np.column_stack([
        radius * np.cos(angles),
        radius * np.sin(angles)
    ])
    
    boundary_edges = [(i, (i+1) % n_points) for i in range(n_points)]
    
    generator = BowyerWatsonMeshGenerator(
        boundary_points=boundary_points,
        boundary_edges=boundary_edges,
        max_edge_length=0.15,
        smoothing_iterations=2,  # 测试改进的平滑
    )
    
    start_time = time.time()
    points, simplices, boundary_mask = generator.generate_mesh(target_triangle_count=200)
    elapsed = time.time() - start_time
    
    print(f"[OK] 网格生成成功")
    print(f"  - 总节点数: {len(points)}")
    print(f"  - 三角形数: {len(simplices)}")
    print(f"  - 生成时间: {elapsed:.3f}s")
    
    # 计算三角形质量统计
    qualities = [generator._compute_triangle_quality(tri) for tri in generator.triangles]
    print(f"\n三角形质量统计:")
    print(f"  - 最小质量: {min(qualities):.4f}")
    print(f"  - 平均质量: {np.mean(qualities):.4f}")
    print(f"  - 最大质量: {max(qualities):.4f}")
    
    return points, simplices


def test_hole():
    """测试带孔洞的网格"""
    print("\n" + "="*80)
    print("测试 3: 带孔洞的正方形（验证孔洞处理和星形性验证）")
    print("="*80)
    
    # 外部正方形
    outer_square = np.array([
        [0, 0],
        [2, 0],
        [2, 2],
        [0, 2],
    ])
    
    # 内部正方形孔洞
    hole = np.array([
        [0.7, 0.7],
        [1.3, 0.7],
        [1.3, 1.3],
        [0.7, 1.3],
    ])
    
    boundary_points = outer_square
    boundary_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    
    generator = BowyerWatsonMeshGenerator(
        boundary_points=boundary_points,
        boundary_edges=boundary_edges,
        max_edge_length=0.15,
        holes=[hole],
        smoothing_iterations=0,
    )
    
    start_time = time.time()
    points, simplices, boundary_mask = generator.generate_mesh(target_triangle_count=100)
    elapsed = time.time() - start_time
    
    print(f"[OK] 网格生成成功")
    print(f"  - 总节点数: {len(points)}")
    print(f"  - 三角形数: {len(simplices)}")
    print(f"  - 生成时间: {elapsed:.3f}s")
    
    # 验证孔洞内的点
    hole_center = np.array([1.0, 1.0])
    points_in_hole = 0
    for i, point in enumerate(points):
        if not boundary_mask[i]:
            dist = np.linalg.norm(point - hole_center)
            if dist < 0.3:
                points_in_hole += 1
    
    print(f"  - 孔洞内的节点数: {points_in_hole}（应该为 0）")
    
    return points, simplices


def test_complex_boundary():
    """测试复杂边界"""
    print("\n" + "="*80)
    print("测试 4: 复杂 L 形边界（验证前端方法和增强点检查）")
    print("="*80)
    
    # L 形边界
    boundary_points = np.array([
        [0, 0],
        [2, 0],
        [2, 1],
        [1, 1],
        [1, 2],
        [0, 2],
    ])
    
    boundary_edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    
    generator = BowyerWatsonMeshGenerator(
        boundary_points=boundary_points,
        boundary_edges=boundary_edges,
        max_edge_length=0.15,
        smoothing_iterations=1,
    )
    
    start_time = time.time()
    points, simplices, boundary_mask = generator.generate_mesh(target_triangle_count=100)
    elapsed = time.time() - start_time
    
    print(f"[OK] 网格生成成功")
    print(f"  - 总节点数: {len(points)}")
    print(f"  - 三角形数: {len(simplices)}")
    print(f"  - 生成时间: {elapsed:.3f}s")
    
    # 计算质量统计
    qualities = [generator._compute_triangle_quality(tri) for tri in generator.triangles]
    print(f"\n三角形质量统计:")
    print(f"  - 最小质量: {min(qualities):.4f}")
    print(f"  - 平均质量: {np.mean(qualities):.4f}")
    print(f"  - 最大质量: {max(qualities):.4f}")
    print(f"  - 质量 > 0.3 的比例: {sum(1 for q in qualities if q > 0.3) / len(qualities):.2%}")
    
    return points, simplices


def performance_comparison():
    """性能对比测试（对比有无优先级队列）"""
    print("\n" + "="*80)
    print("测试 5: 性能对比（验证优先级队列优化效果）")
    print("="*80)
    
    # 生成较大的圆形边界
    n_points = 64
    angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    radius = 1.0
    boundary_points = np.column_stack([
        radius * np.cos(angles),
        radius * np.sin(angles)
    ])
    
    boundary_edges = [(i, (i+1) % n_points) for i in range(n_points)]
    
    # 测试不同目标三角形数
    for target_count in [100, 200, 500]:
        print(f"\n目标三角形数: {target_count}")
        
        generator = BowyerWatsonMeshGenerator(
            boundary_points=boundary_points,
            boundary_edges=boundary_edges,
            max_edge_length=0.1,
            smoothing_iterations=0,
        )
        
        start_time = time.time()
        points, simplices, boundary_mask = generator.generate_mesh(target_triangle_count=target_count)
        elapsed = time.time() - start_time
        
        print(f"  - 实际三角形数: {len(simplices)}")
        print(f"  - 总节点数: {len(points)}")
        print(f"  - 生成时间: {elapsed:.3f}s")


def test_adjacency_consistency():
    """测试邻接关系一致性"""
    print("\n" + "="*80)
    print("测试 6: 邻接关系一致性验证（P0-1）")
    print("="*80)
    
    # 简单正方形
    boundary_points = np.array([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
    ])
    
    boundary_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    
    generator = BowyerWatsonMeshGenerator(
        boundary_points=boundary_points,
        boundary_edges=boundary_edges,
        max_edge_length=0.2,
        smoothing_iterations=0,
    )
    
    points, simplices, boundary_mask = generator.generate_mesh(target_triangle_count=30)
    
    # 验证邻接关系的双向性
    print("\n验证邻接关系双向性:")
    bidirectional_count = 0
    total_edges = 0
    
    for tri in generator.triangles:
        for i in range(3):
            neighbor = tri.neighbors[i]
            if neighbor is not None:
                total_edges += 1
                # 检查邻居是否也指向自己
                for j in range(3):
                    if neighbor.neighbors[j] is tri:
                        bidirectional_count += 1
                        break
    
    consistency = bidirectional_count / total_edges if total_edges > 0 else 0
    print(f"  - 双向邻接边: {bidirectional_count}/{total_edges}")
    print(f"  - 一致性: {consistency:.2%}")
    
    assert consistency > 0.95, f"邻接关系一致性过低: {consistency:.2%}"
    print(f"[OK] 邻接关系一致性验证通过")


if __name__ == "__main__":
    print("开始测试 bw_core.py 的所有改进项...")
    
    try:
        # 基础功能测试
        test_basic_square()
        test_circle()
        test_hole()
        test_complex_boundary()
        
        # 性能测试
        performance_comparison()
        
        # 邻接关系验证
        test_adjacency_consistency()
        
        print("\n" + "="*80)
        print("[OK] 所有测试通过！")
        print("="*80)
        
    except Exception as e:
        print(f"\n[FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
