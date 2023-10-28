from trimesh.base import Trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector

import torch
torch.autograd.set_detect_anomaly(True)


def _dot_product(Y, X, P, n_expanded):
    cross_product = torch.cross(Y - X, P - X, dim=-1)
    dot_product = (cross_product * n_expanded).sum(dim=-1)
    return dot_product


def intersection_points_on_mesh(
    ray_origins,
    ray_directions,
    faces,
    vertices
):
    """
        Find the intersection points between rays and a 3D mesh.

        Args:
            ray_origins (torch.Tensor):
                The origins of the rays as a tensor of shape (N, 3),
                where N is the number of rays.
            ray_directions (torch.Tensor):
                The directions of the rays as a tensor of shape (N, 3).
            faces (torch.Tensor):
                The faces of the 3D mesh as a tensor of shape (M, 3),
                where M is the number of faces.
            vertices (torch.Tensor):
                The vertices of the 3D mesh as a tensor of shape (K, 3),
                where K is the number of vertices.

        Returns:
            ray_idxs (torch.Tensor):
                A tensor containing the indices of the rays,
                for which there is any intersection point.
            pts (torch.Tensor):
                A tensor containing the intersection points
                for each ray, reshaped to (3, N).
    """

    """ray_idxs_obj, selected_face_idxs, _pts = intersection_points_on_mesh_trimesh_obj(
        faces=faces.detach().clone().cpu(),
        vertices=vertices.detach().clone().cpu(),
        ray_origins=ray_origins.detach().clone().cpu(),
        ray_directions=ray_directions.detach().clone().cpu(),
    )"""

    out = rays_triangles_intersection(
        ray_origins=ray_origins.double(),
        ray_directions=ray_directions.double(),
        triangle_vertices=vertices[faces.long()].double(),
    )

    intersection_pts = out['pts_nearest_each_ray'].float()
    pts = intersection_pts.unsqueeze(0).swapaxes(0, 1)

    ray_idxs = out["nearest_points_idx"][0]
    rs = ray_idxs.detach().cpu().numpy()

    triangle_idxs = out["nearest_points_idx"][1]
    ts = triangle_idxs.detach().cpu().numpy()

    # test
    #assert (rs == ray_idxs_obj).all()
    #if ~(ts == selected_face_idxs).all():
    #    torch.save(vertices, 'vertices.pt')
    #    torch.save(faces, 'faces.pt')
    #    torch.save(ray_directions, 'ray_directions.pt')
    #    torch.save(ray_origins, 'ray_origins.pt')

    #assert (ts == selected_face_idxs).all()

    """pts_diff = pts.detach() - _pts
    pts_diff_sum = pts_diff.sum()
    if pts_diff_sum > 0.005:
        print(pts_diff_sum)
        torch.save(vertices, 'vertices.pt')
        torch.save(faces, 'faces.pt')
        torch.save(ray_directions, 'ray_directions.pt')
        torch.save(ray_origins, 'ray_origins.pt')

    assert pts_diff_sum < 0.005
    """
    return ray_idxs, pts, None


def rays_triangles_intersection(
    ray_origins,
    ray_directions,
    triangle_vertices,
):
    num_rays = ray_origins.shape[0]
    num_triangles = triangle_vertices.shape[0]

    # Triangle
    A = triangle_vertices[:, 0]
    B = triangle_vertices[:, 1]
    C = triangle_vertices[:, 2]

    AB = B - A  # Oriented segment A to B
    AC = C - A  # Oriented segment A to C
    n = torch.cross(AB, AC)  # Normal vector
    n_ = n / torch.linalg.norm(n, dim=1, keepdim=True)  # Normalized normal

    # expand
    n_expanded = n_.expand(num_rays, num_triangles, 3)

    ray_origins_expanded = ray_origins.view(
        num_rays, 1, 3
    ).expand(num_rays, num_triangles, 3)

    ray_directions_norm = ray_directions / torch.linalg.norm(
        ray_directions, dim=1, keepdim=True
    )  # Unit vector (versor) of e => ê

    ray_directions_norm_expanded = ray_directions_norm.view(
        num_rays, 1, 3
    ).expand(num_rays, num_triangles, 3)

    A_expand = A.expand(num_rays, num_triangles, 3)
    B_expand = B.expand(num_rays, num_triangles, 3)
    C_expand = C.expand(num_rays, num_triangles, 3)

    # Using the point A to find d
    d = -(n_expanded * A_expand).sum(dim=-1)

    # Finding parameter t
    t = -((n_expanded * ray_origins_expanded).sum(dim=-1) + d)
    tt = (n_expanded * ray_directions_norm_expanded).sum(dim=-1)

    _tt = torch.where(tt == 0, 0.005, tt)

    t /= _tt

    # Finding P [num_rays, num_triangles, 3D point]
    pts = ray_origins_expanded + t.view(
        num_rays, num_triangles, 1
    ) * ray_directions_norm_expanded

    # pts[pts == float("Inf")] = 0

    # Get the resulting vector for each vertex
    # following the construction order
    Pa = _dot_product(B_expand, A_expand, pts, n_expanded)
    Pb = _dot_product(C_expand, B_expand, pts, n_expanded)
    Pc = _dot_product(A_expand, C_expand, pts, n_expanded)

    backface_intersection = torch.where(t < 0, 0, 1)

    eps = 10**(-9)
    valid_point = (Pa > -eps) & (Pb > -eps) & (Pc > -eps)  # [num_rays, num_triangles]

    _d = pts - ray_origins_expanded
    _d = (_d**2).sum(dim=2)

    d_valid = valid_point.int() * _d
    d_valid_inv = - torch.log(d_valid.abs())

    idx = d_valid_inv.abs().min(dim=1).indices
    nearest_valid_point_mask = torch.zeros_like(d_valid_inv)
    nearest_valid_point_mask[range(num_rays), idx] = 1
    nearest_valid_point_mask = (d_valid_inv != 0) * nearest_valid_point_mask

    idxs = torch.where(nearest_valid_point_mask == 1)
    pts_nearest = pts[idxs]

    nearest_points = nearest_valid_point_mask * valid_point
    nearest_points_idx = torch.where(nearest_points == 1)

    pts_nearest_each_ray = torch.zeros(num_rays, 3).double()
    pts_nearest_each_ray[nearest_points_idx[0].long()] = pts[nearest_points_idx].double()

    out = {
        'pts': pts,
        'backface_intersection': backface_intersection,
        'valid_point': valid_point,
        'nearest_valid_point_mask': nearest_valid_point_mask,
        'pts_nearest': pts_nearest,
        'nearest_points_idx': nearest_points_idx,
        'pts_nearest_each_ray': pts_nearest_each_ray
    }
    return out


def intersection_points_on_mesh_trimesh_obj(
    faces: torch.Tensor,
    vertices: torch.Tensor,
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
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
        ray_origins (torch.Tensor):
            Origin points of the rays, shape (N, 3).
        ray_directions (torch.Tensor):
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
        ray_origins.cpu().detach().numpy(),
        ray_directions.cpu().detach().numpy(),
        multiple_hits=False
    )
    ray_idxs_intersection_mash = out[1]
    face_idxs_intersection_mash = out[2]

    if return_each_ray:
        n_ray = ray_directions.shape[0]
        intersection_points_each_ray = torch.ones(
            n_ray, 3
        ) * no_intersection_point
        intersection_points_each_ray[ray_idxs_intersection_mash] = torch.tensor(
            out[0]).float().to(intersection_points_each_ray.device)

        intersection_points_each_ray = intersection_points_each_ray.unsqueeze(0).swapaxes(0, 1)
        return (ray_idxs_intersection_mash,
                face_idxs_intersection_mash,
                intersection_points_each_ray)
    else:
        return (ray_idxs_intersection_mash,
                face_idxs_intersection_mash,
                torch.tensor(out[0]).float().to(ray_origins.device))


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

    # epsilons = torch.rand(ray_directions_norm.shape[0], n_points) * 2 * eps - eps

    # Expand dimensions to match the target shape (n, m, 3)
    ray_directions_norm = ray_directions_norm.unsqueeze(1)
    epsilons = epsilons.unsqueeze(2)

    # Add epsilon to ray_directions_norm using broadcasting
    result = ray_directions_norm * epsilons
    new_points = result + points.unsqueeze(1)
    #new_points = new_points.detach() # new_points without grad

    dist_with_grad = (new_points - points.unsqueeze(1))/ray_directions_norm
    dist_with_grad = dist_with_grad[:, :, 0]
    return new_points, dist_with_grad


def transform_points_to_single_number_representation(
    ray_directions: torch.Tensor,
    ray_origin: torch.Tensor,
    points: torch.Tensor,
):
    result = (points - torch.unsqueeze(ray_origin, 1)) \
             / torch.unsqueeze(ray_directions, 1)
    return torch.nanmean(result, dim=2)