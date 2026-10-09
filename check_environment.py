"""Check SceneMI imports and actual CUDA extension execution."""
import importlib

import torch


def main():
    for name in (
        "train.train_diffusion_scenemib", "open3d", "mesh2sdf",
        "mesh_to_sdf", "human_body_prior.tools.rotation_tools",
        "common.utils", "kaolin",
    ):
        importlib.import_module(name)
        print("Import OK:", name)
    assert torch.cuda.is_available(), "CUDA is not available"
    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
    from pytorch3d.ops import knn_points
    from vit_pytorch import ViT
    from chamfer_distance import ChamferDistance
    from kaolin.ops.mesh import check_sign

    points = torch.randn(1, 16, 3, device="cuda", requires_grad=True)
    result = knn_points(points, points.detach() + 0.1, K=1)
    result.dists.sum().backward()
    assert torch.isfinite(points.grad).all()
    distances = ChamferDistance()(points.detach(), points.detach() + 0.1)
    assert torch.isfinite(distances[0]).all()
    vertices = torch.tensor(
        [[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]],
        device="cuda",
    )
    faces = torch.tensor([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], device="cuda")
    query = torch.tensor([[[0.1, 0.1, 0.1], [2., 2., 2.]]], device="cuda")
    assert check_sign(vertices, faces, query).tolist() == [[True, False]]
    model = ViT(image_size=32, patch_size=8, num_classes=16, dim=32,
                depth=1, heads=2, mlp_dim=64).cuda()
    model(torch.randn(2, 3, 32, 32, device="cuda")).sum().backward()
    torch.cuda.synchronize()
    print("CUDA forward/backward, KNN, ChamferDistance, mesh check_sign: OK")


if __name__ == "__main__":
    main()
