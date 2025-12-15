import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    Calculate Euclidean distance between each two points using only XY coordinates.

    src^T * dst = xn * xm + yn * ym;
    sum(src^2, dim=-1) = xn*xn + yn*yn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym;
    dist = (xn - xm)^2 + (yn - ym)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source gaussians, [B, N, 8] where first 2 dims are XY positions
        dst: target gaussians, [B, M, 8] where first 2 dims are XY positions
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    
    # Extract only XY coordinates
    src_xy = src[:, :, :2]  # [B, N, 2]
    dst_xy = dst[:, :, :2]  # [B, M, 2]
    
    # Calculate squared distance using only XY
    dist = -2 * torch.matmul(src_xy, dst_xy.permute(0, 2, 1))
    dist += torch.sum(src_xy ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst_xy ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input gaussian data, [B, N, 8]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, 8]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(gaussians, npoint):
    """
    Input:
        gaussians: gaussian data, [B, N, 8] where first 2 dims are XY positions
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = gaussians.device
    B, N, C = gaussians.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    
    # Extract only XY coordinates for distance calculation
    xy = gaussians[:, :, :2]  # [B, N, 2]
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xy[batch_indices, farthest, :].view(B, 1, 2)  # [B, 1, 2]
        dist = torch.sum((xy - centroid) ** 2, -1)  # Euclidean distance on XY only
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, gaussians, new_gaussians):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        gaussians: all points, [B, N, 8]
        new_gaussians: query points, [B, S, 8]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = gaussians.device
    B, N, C = gaussians.shape
    _, S, _ = new_gaussians.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_gaussians, gaussians)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, gaussians, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        gaussians: input points position data, [B, N, 8]
        points: input points data, [B, N, D]
    Return:
        new_gaussians: sampled points position data, [B, npoint, 8]
        new_points: sampled points data, [B, npoint, nsample, 8+D]
    """
    B, N, C = gaussians.shape
    S = npoint
    fps_idx = farthest_point_sample(gaussians, npoint) # [B, npoint]
    new_gaussians = index_points(gaussians, fps_idx)  # [B, npoint, 8]
    idx = query_ball_point(radius, nsample, gaussians, new_gaussians)  # [B, npoint, nsample]
    grouped_gaussians = index_points(gaussians, idx) # [B, npoint, nsample, 8]
    
    # Only normalize XY coordinates, keep other features unchanged
    grouped_gaussians_norm = grouped_gaussians.clone()
    grouped_gaussians_norm[:, :, :, :2] = grouped_gaussians[:, :, :, :2] - new_gaussians[:, :, None, :2]
    # Other features (scale, rot, feat) remain absolute, not relative

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_gaussians_norm, grouped_points], dim=-1) # [B, npoint, nsample, 8+D]
    else:
        new_points = grouped_gaussians_norm
    if returnfps:
        return new_gaussians, new_points, grouped_gaussians, fps_idx
    else:
        return new_gaussians, new_points

def sample_and_group_all(gaussians, points):
    """
    Input:
        gaussians: input points position data, [B, N, 8]
        points: input points data, [B, N, D]
    Return:
        new_gaussians: sampled points position data, [B, 1, 8]
        new_points: sampled points data, [B, 1, N, 8+D]
    """
    device = gaussians.device
    B, N, C = gaussians.shape
    
    # Compute centroid of XY coordinates
    centroid_xy = torch.mean(gaussians[:, :, :2], dim=1, keepdim=True)  # [B, 1, 2]
    
    # Create new_gaussians with centroid XY and zeros for other features
    new_gaussians = torch.zeros(B, 1, C).to(device)
    new_gaussians[:, :, :2] = centroid_xy
    
    # Group all gaussians and normalize XY relative to centroid
    grouped_gaussians = gaussians.view(B, 1, N, C).clone()
    grouped_gaussians[:, :, :, :2] = grouped_gaussians[:, :, :, :2] - centroid_xy.unsqueeze(2)
    # Other features (scale, rot, feat) remain absolute
    
    if points is not None:
        new_points = torch.cat([grouped_gaussians, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_gaussians
    return new_gaussians, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points) # type: ignore
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            # First layer gets full 8D gaussians (XY normalized + 6 features)
            last_channel = in_channel + 2  # in_channel is the additional features, +2 for XY
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, 2, N] (XY coordinates)
            points: input points data, [B, D, N] (additional features beyond XY)
        Return:
            new_xyz: sampled points position data, [B, 2, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)  # [B, N, 2]
        if points is not None:
            points = points.permute(0, 2, 1)  # [B, N, D]

        B, N, C = xyz.shape  # C=2 for XY
        S = self.npoint
        
        # Combine xyz and points for full gaussian representation
        if points is not None:
            full_gaussians = torch.cat([xyz, points], dim=-1)  # [B, N, 2+D]
        else:
            full_gaussians = xyz  # [B, N, 2]
        
        # Farthest point sampling on full gaussians (uses XY for distance)
        fps_idx = farthest_point_sample(full_gaussians, S)  # [B, S]
        new_xyz = index_points(xyz, fps_idx)  # [B, S, 2] - only XY positions
        new_full = index_points(full_gaussians, fps_idx)  # [B, S, 2+D] - full gaussians at sampled points
        
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            # Query based on full gaussians (but distance uses XY only in query_ball_point)
            group_idx = query_ball_point(radius, K, full_gaussians, new_full)  # [B, S, K]
            grouped_gaussians = index_points(full_gaussians, group_idx)  # [B, S, K, 2+D]
            
            # Normalize only XY coordinates relative to centroid
            grouped_gaussians_norm = grouped_gaussians.clone()
            grouped_gaussians_norm[:, :, :, :2] = grouped_gaussians[:, :, :, :2] - new_xyz.unsqueeze(2)  # [B, S, K, 2]
            # Other features remain absolute: grouped_gaussians_norm[:, :, :, 2:] unchanged

            # Permute for Conv2d: [B, S, K, 2+D] -> [B, 2+D, K, S]
            grouped_points = grouped_gaussians_norm.permute(0, 3, 2, 1)  # [B, 2+D, K, S]
            
            # Apply convolutions
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            
            # Max pooling over neighbors
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        # Extract only XY positions for output
        new_xyz = new_xyz.permute(0, 2, 1)  # [B, 2, S]
        new_points_concat = torch.cat(new_points_list, dim=1)  # [B, total_D', S]
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points