import torch
import trimesh
import numpy as np
import argparse
import time
from pathlib import Path


"""
This code using neural network to smooth the mesh. This is the implementation of the paper:
[1] Guo Y F, Wang C R, Ma Z, et al. A new mesh smoothing method based on a neural network[J]. 
Computational Mechanics, 2022, 69:425-438.

The opensource code is originally from the author's GitHub:
https://github.com/yfguo91/meshsmoothing

The code is modified by the author of this repository to make it more readable and usable.
"""


def run(input_file, output_file, opt_epoch):
    model_dir = Path(__file__).parent / "model"
    opt3 = torch.load(model_dir / "opt3.pkl", weights_only=False)
    opt4 = torch.load(model_dir / "opt4.pkl", weights_only=False)
    opt5 = torch.load(model_dir / "opt5.pkl", weights_only=False)
    opt6 = torch.load(model_dir / "opt6.pkl", weights_only=False)
    opt7 = torch.load(model_dir / "opt7.pkl", weights_only=False)
    opt8 = torch.load(model_dir / "opt8.pkl", weights_only=False)
    opt9 = torch.load(model_dir / "opt9.pkl", weights_only=False)

    def improveRing(ring):
        if ring.shape[0] == 6:
            opt = opt3
        if ring.shape[0] == 8:
            opt = opt4
        if ring.shape[0] == 10:
            opt = opt5
        if ring.shape[0] == 12:
            opt = opt6
        if ring.shape[0] == 14:
            opt = opt7
        if ring.shape[0] == 16:
            opt = opt8
        if ring.shape[0] == 18:
            opt = opt9
        ring = torch.from_numpy(ring).float()
        opoint = opt(ring).data.numpy()
        return opoint

    def laplacian(ring):
        newpoints = np.mean(ring, axis=0)
        return newpoints

    # 读入网格
    mesh = trimesh.load(input_file)  # 这个网格结构可以方便的寻找点面之间的邻接关系
    points = mesh.vertices  # 网格点的坐标
    faces = mesh.faces  # 网格的三角片

    # 提取边界点、点的邻接关系
    boundary_mask = trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)
    boundary_edges = mesh.edges[boundary_mask]  # 边界edges
    boundary_ids = np.unique(boundary_edges.ravel())  # 边界点的索引
    bpindex = np.zeros(len(points), dtype=bool)  # 记录边界点flag
    bpindex[boundary_ids] = True  # boundary point flag
    vvlist = mesh.vertex_neighbors  # 点的邻接关系

    # 打印网格信息
    print("Input file path: ", input_file)
    print("Mesh loaded:", not (mesh.is_empty))
    print("Mesh information:")
    print("Vertices:", len(mesh.vertices))
    print("Faces:", len(mesh.faces))
    print("Edges:", len(mesh.edges))
    print("Reading input mesh file..., DONE!")

    # mesh.show(
    #     wireframe=True,  # 启用线框模式
    #     wireframe_color=[0, 0, 0, 1],  # 黑色线框
    #     background=[1, 1, 1, 1],  # 白色背景
    # )

    # 优化网格
    start = time.time()
    for epoch in range(opt_epoch):
        for i in range(0, points.shape[0]):
            if bpindex[i] == True:
                continue

            vvpoints = points[vvlist[i]]
            # ring nodes数量小于3或者大于9，直接用laplacian光滑，否则，用神经网络光滑
            if vvpoints.shape[0] < 3 or vvpoints.shape[0] > 9:
                points[i] = laplacian(vvpoints)
            else:
                vvpoints = vvpoints.flatten()
                l = []
                for m in range(0, vvpoints.shape[0]):
                    if (m + 1) % 3 != 0:
                        l.append(m)
                vvpoints = vvpoints[l]

                ringx = vvpoints[0 : vvpoints.shape[0] : 2]
                ringy = vvpoints[1 : vvpoints.shape[0] : 2]

                ringxmin = ringx.min()
                ringxmax = ringx.max()
                ringymin = ringy.min()
                ringymax = ringy.max()

                if ringxmax - ringxmin > ringymax - ringymin:
                    max_length = ringxmax - ringxmin
                else:
                    max_length = ringymax - ringymin

                vvpoints[0 : vvpoints.shape[0] : 2] = (ringx - ringxmin) / (max_length)
                vvpoints[1 : vvpoints.shape[0] : 2] = (ringy - ringymin) / (max_length)

                points[i, 0:2] = improveRing(vvpoints) * max_length + [
                    ringxmin,
                    ringymin,
                ]

        end = time.time()
        print(f"epoch {epoch+1}, time elapsed= {(end - start):.3f}s")

    new_mesh = trimesh.Trimesh(vertices=points, faces=faces)
    new_mesh.export(output_file)
    print("Export output mesh file..., DONE!")
    print("Output file path: ", output_file)

    # new_mesh.show(
    #     wireframe=True,
    #     wireframe_color=[0, 0, 0, 1],
    #     background=[1, 1, 1, 1],
    # )

    return


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="optimize mesh by optimizer")
    parser.add_argument(
        "--input_file", help="The input file.", default=None, required=True
    )
    parser.add_argument(
        "--output_file", help="The output file.", default=None, required=True
    )
    parser.add_argument(
        "--opt_epoch", help="the epoch of smooth.", type=int, default=10
    )
    return parser.parse_args()


def predefined_examples(example_index):
    """Predefined examples."""
    example_dir = Path(__file__).parent / "example"
    examples = {
        1: {
            "input_file": example_dir / "first.stl",
            "output_file": example_dir / "first_opt.stl",
            "opt_epoch": 10,
        },
        2: {
            "input_file": example_dir / "second.stl",
            "output_file": example_dir / "second_opt.stl",
            "opt_epoch": 10,
        },
        3: {
            "input_file": example_dir / "third.stl",
            "output_file": example_dir / "third_opt.stl",
            "opt_epoch": 10,
        },
        4: {
            "input_file": example_dir / "fourth.stl",
            "output_file": example_dir / "fourth_opt.stl",
            "opt_epoch": 10,
        },
        5: {
            "input_file": example_dir / "fifth.stl",
            "output_file": example_dir / "fifth_opt.stl",
            "opt_epoch": 10,
        },
        6: {
            "input_file": example_dir / "sixth.stl",
            "output_file": example_dir / "sixth_opt.stl",
            "opt_epoch": 10,
        },
        7: {
            "input_file": example_dir / "rae2822_bad.stl",
            "output_file": example_dir / "rae2822_opt.stl",
            "opt_epoch": 10,
        },
        8: {
            "input_file": example_dir / "30p30n_bad.stl",
            "output_file": example_dir / "30p30n_opt.stl",
            "opt_epoch": 10,
        },
        9: {
            "input_file": example_dir / "naca0012_bad.stl",
            "output_file": example_dir / "naca0012_opt.stl",
            "opt_epoch": 10,
        },
    }
    if example_index in examples:
        return examples[example_index]
    else:
        raise ValueError(f"Example index {example_index} not found.")


# if __name__ == "__main__":
#     args = parse_args()
#     run(args.input_file, args.output_file, args.opt_epoch)

if __name__ == "__main__":
    example_index = 9  # 选择一个示例
    example_args = predefined_examples(example_index)
    run(
        example_args["input_file"],
        example_args["output_file"],
        example_args["opt_epoch"],
    )
