import torch
from trimesh.base import Trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector


def intersection_points_on_mesh(
    faces: torch.Tensor,
    vertices: torch.Tensor,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    return_each_ray: bool = True,
    no_intersection_point: float = 00.0
):
    """
    Calculate intersection points of rays with a 3D mesh.

    Args:
        faces (torch.Tensor):
            A tensor representing the faces of the mesh.
        vertices (torch.Tensor):
            A tensor representing the vertices of the mesh.
        rays_o (torch.Tensor):
            Origin points of the rays, shape (N, 3).
        rays_d (torch.Tensor):
            Direction vectors of the rays, shape (N, 3).
        return_each_ray (bool, optional):
            If True, return intersection points for each ray.
        no_intersection_point (float, optional):
            Value for rays with no intersections. Used only if
            return_each_ray is True.

    Returns:
        torch.Tensor
        either:
        If return_each_ray is:
        - True, returns intersection points for each ray (N, 3).
        - False, returns intersection points (M, 3). Where M is numbers of points.
    """

    faces = faces.cpu().detach().numpy()
    trimesh_obj = Trimesh(
        vertices=vertices.cpu().detach().numpy(),
        faces=faces
    )
    ray_mesh_intersection = RayMeshIntersector(trimesh_obj)
    out = ray_mesh_intersection.intersects_location(
        rays_o.cpu().detach().numpy(),
        rays_d.cpu().detach().numpy(),
        multiple_hits=False
    )
    ray_idxs_intersection_mash = out[1]
    if return_each_ray:
        n_ray = rays_d.shape[0]
        intersection_points_each_ray = torch.ones(
            n_ray, 3
        ) * no_intersection_point
        intersection_points_each_ray[ray_idxs_intersection_mash] = torch.tensor(
            out[0]).float().to(intersection_points_each_ray.device)
        return ray_idxs_intersection_mash, intersection_points_each_ray
    else:
        return ray_idxs_intersection_mash, torch.tensor(out[0]).float().to(rays_o.device)


def sample_extra_points_on_mesh(
        points: torch.Tensor,
        ray_directions: torch.Tensor,
        n_points: int,
        eps: float = 0.005
):
    """
    Generate extra points along rays for a 3D mesh. Near to points.

    Args:
        points (torch.Tensor): Tensor of initial points, shape (n, 3).
        ray_directions (torch.Tensor): Tensor of ray directions, shape (n, 3).
        n_points (int): Number of extra points to generate along each ray.
        eps (float, optional): Small perturbation factor. Default is 0.005.

    Returns:
        torch.Tensor: Generated points along rays, shape (n, m, 3), where n is the
        number of rays, and m is the number of extra points.
    """

    ray_directions_norm = ray_directions / torch.norm(
        ray_directions, dim=1, keepdim=True
    )  # Unit vector (vector) of e => ê

    # Define your epsilon tensor (n, m)
    epsilons = torch.normal(
        mean=0,
        std=torch.ones(ray_directions_norm.shape[0], n_points) * eps
    )

    # Expand dimensions to match the target shape (n, m, 3)
    ray_directions_norm = ray_directions_norm.unsqueeze(1)
    epsilons = epsilons.unsqueeze(2)

    # Add epsilon to ray_directions_norm using broadcasting
    result = ray_directions_norm * epsilons
    new_points = result + points.unsqueeze(1)
    return new_points


def transform_points_to_single_number_representation(
    ray_directions: torch.Tensor,
    ray_origin: torch.Tensor,
    points: torch.Tensor,
):
    result = (points - torch.unsqueeze(ray_origin, 1)) \
             / torch.unsqueeze(ray_directions, 1)
    return torch.nanmean(result, dim=2)
