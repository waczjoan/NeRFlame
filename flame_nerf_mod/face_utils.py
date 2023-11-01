import sys

from torch.autograd.function import once_differentiable

sys.path.append('./FLAME/')
from pytorch3d import _C
import torch
from pytorch3d.structures import Meshes, Pointclouds
from torch.autograd import Function

_DEFAULT_MIN_TRIANGLE_AREA: float = 5e-4


class _PointFaceDistance(Function):
    """
    Torch autograd Function wrapper PointFaceDistance Cuda implementation
    """

    @staticmethod
    def forward(
        ctx,
        points,
        points_first_idx,
        tris,
        tris_first_idx,
        max_points,
        min_triangle_area=_DEFAULT_MIN_TRIANGLE_AREA,
    ):
        """
        Args:
            ctx: Context object used to calculate gradients.
            points: FloatTensor of shape `(P, 3)`
            points_first_idx: LongTensor of shape `(N,)` indicating the first point
                index in each example in the batch
            tris: FloatTensor of shape `(T, 3, 3)` of triangular faces. The `t`-th
                triangular face is spanned by `(tris[t, 0], tris[t, 1], tris[t, 2])`
            tris_first_idx: LongTensor of shape `(N,)` indicating the first face
                index in each example in the batch
            max_points: Scalar equal to maximum number of points in the batch
            min_triangle_area: (float, defaulted) Triangles of area less than this
                will be treated as points/lines.
        Returns:
            dists: FloatTensor of shape `(P,)`, where `dists[p]` is the squared
                euclidean distance of `p`-th point to the closest triangular face
                in the corresponding example in the batch
            idxs: LongTensor of shape `(P,)` indicating the closest triangular face
                in the corresponding example in the batch.

            `dists[p]` is
            `d(points[p], tris[idxs[p], 0], tris[idxs[p], 1], tris[idxs[p], 2])`
            where `d(u, v0, v1, v2)` is the distance of point `u` from the triangular
            face `(v0, v1, v2)`

        """
        dists, idxs = _C.point_face_dist_forward(
            points,
            points_first_idx,
            tris,
            tris_first_idx,
            max_points,
            min_triangle_area,
        )
        ctx.save_for_backward(points, tris, idxs)
        ctx.min_triangle_area = min_triangle_area
        return dists, idxs

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_dists, grad_idxs):
        grad_dists = grad_dists.contiguous()
        points, tris, idxs = ctx.saved_tensors
        min_triangle_area = ctx.min_triangle_area
        grad_points, grad_tris = _C.point_face_dist_backward(
            points, tris, idxs, grad_dists, min_triangle_area
        )
        return grad_points, None, grad_tris, None, None, None


point_face_distance = _PointFaceDistance.apply


def point_mesh_face_distance(
        meshes: Meshes,
        pcls: Pointclouds,
        min_triangle_area: float = _DEFAULT_MIN_TRIANGLE_AREA,
):
    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")
    N = len(meshes)

    # packed representation for pointclouds
    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
    max_tris = meshes.num_faces_per_mesh().max().item()

    # point to face distance: shape (P,)
    point_to_face, idx = point_face_distance(
        points, points_first_idx, tris, tris_first_idx, max_points, min_triangle_area
    )

    return point_to_face, idx


def distance_calculator(set_of_coordinates, mesh_points):
    return torch.cdist(set_of_coordinates, mesh_points)


def FLAME_based_alpha_calculator_3_face_version(
        set_of_coordinates, mesh_points, mesh_faces
):
    mesh_points = [mesh_points]
    mesh_faces = [mesh_faces]

    set_of_coordinates_size = set_of_coordinates.size()
    set_of_coordinates = [torch.flatten(set_of_coordinates, 0, 1)]

    p_mesh = Meshes(verts=mesh_points, faces=mesh_faces)
    p_points = Pointclouds(points=set_of_coordinates)

    dists, idxs = point_mesh_face_distance(p_mesh, p_points)
    dists = torch.sqrt(dists)
    dists, idxs = torch.reshape(dists, (set_of_coordinates_size[0], set_of_coordinates_size[1])), \
                  torch.reshape(idxs, (set_of_coordinates_size[0], set_of_coordinates_size[1]))

    return dists, idxs


def FLAME_based_alpha_calculator_f_relu(min_d, m, e):
    alpha = 1 - ((m(min_d) / e) - (m(min_d - e) / e))
    return alpha
