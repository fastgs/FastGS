#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
import json
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import cv2
import open3d as o3d
from scene.app_model import AppModel
import copy
from collections import deque
import time

def clean_mesh(mesh, min_len=1000):
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < min_len
    mesh_0 = copy.deepcopy(mesh)
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    return mesh_0

def post_process_mesh(mesh, cluster_to_keep=1):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy
    print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50) # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0

def render_set(model_path, name, iteration, views, scene, gaussians, pipeline, background, args,
               app_model=None, max_depth=5.0, volume=None, use_depth_filter=False):

    gts_path = os.path.join(model_path, name, f"ours_{iteration}", "gt")
    render_path = os.path.join(model_path, name, f"ours_{iteration}", "renders")
    render_depth_path = os.path.join(model_path, name, f"ours_{iteration}", "renders_depth")
    render_normal_path = os.path.join(model_path, name, f"ours_{iteration}", "renders_normal")

    os.makedirs(gts_path, exist_ok=True)
    os.makedirs(render_path, exist_ok=True)
    os.makedirs(render_depth_path, exist_ok=True)
    os.makedirs(render_normal_path, exist_ok=True)

    total_time = 0.0
    depths_tsdf_fusion = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        gt, _ = view.get_image()
        start_time = time.time()
        out = render(view, gaussians, pipeline, background, args, app_model=app_model)
        end_time = time.time()
        total_time += (end_time - start_time)
        rendering = out["render"].clamp(0.0, 1.0)
        _, H, W = rendering.shape

        depth = out["plane_depth"].squeeze()
        depth_tsdf = depth.clone()
        depth_np = depth.detach().cpu().numpy()
        depth_i = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-20)
        depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)

        normal = out["rendered_normal"].permute(1,2,0)
        normal = normal / (normal.norm(dim=-1, keepdim=True) + 1.0e-8)
        normal = normal.detach().cpu().numpy()
        normal = ((normal + 1) * 127.5).astype(np.uint8).clip(0, 255)

        if name == 'test':
            torchvision.utils.save_image(gt.clamp(0.0, 1.0),
                                         os.path.join(gts_path, view.image_name + ".png"))
            torchvision.utils.save_image(rendering,
                                         os.path.join(render_path, view.image_name + ".png"))
        else:
            rendering_np = (rendering.permute(1,2,0).clamp(0,1)[:,:,[2,1,0]] * 255
                            ).detach().cpu().numpy().astype(np.uint8)
            cv2.imwrite(os.path.join(render_path, view.image_name + ".jpg"), rendering_np)
        cv2.imwrite(os.path.join(render_depth_path, view.image_name + ".jpg"), depth_color)
        cv2.imwrite(os.path.join(render_normal_path, view.image_name + ".jpg"), normal)

        if use_depth_filter:
            view_dir = torch.nn.functional.normalize(view.get_rays(), p=2, dim=-1)
            depth_normal = out["depth_normal"].permute(1,2,0)
            depth_normal = torch.nn.functional.normalize(depth_normal, p=2, dim=-1)
            dot = torch.sum(view_dir * depth_normal, dim=-1).abs()
            angle = torch.acos(dot)
            mask = angle > (80.0 / 180 * np.pi)
            depth_tsdf[mask] = 0
        depths_tsdf_fusion.append(depth_tsdf.squeeze().cpu())

    # ===== TSDF 融合并生成 mesh =====
    if volume is not None:
        depths_tsdf_fusion = torch.stack(depths_tsdf_fusion, dim=0)
        for idx, view in enumerate(tqdm(views, desc="TSDF Fusion progress")):
            ref_depth = depths_tsdf_fusion[idx].cuda()
            if view.mask is not None:
                ref_depth[view.mask.squeeze() < 0.5] = 0
            ref_depth[ref_depth > max_depth] = 0
            ref_depth = ref_depth.detach().cpu().numpy()

            pose = np.identity(4)
            pose[:3, :3] = view.R.transpose(-1, -2)
            pose[:3, 3] = view.T
            color = o3d.io.read_image(os.path.join(render_path, view.image_name + ".jpg"))
            depth = o3d.geometry.Image((ref_depth * 1000).astype(np.uint16))

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color, depth, depth_scale=1000.0, depth_trunc=max_depth, convert_rgb_to_intensity=False)

            volume.integrate(
                rgbd,
                o3d.camera.PinholeCameraIntrinsic(W, H, view.Fx, view.Fy, view.Cx, view.Cy),
                pose)

        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        mesh_path_color = os.path.join(mesh_path, "mesh_color.ply")
        o3d.io.write_triangle_mesh(mesh_path_color, mesh)

        mesh_no_color = o3d.geometry.TriangleMesh(mesh) 
        mesh_no_color.vertex_colors = o3d.utility.Vector3dVector(np.zeros((len(mesh_no_color.vertices), 3)))
        mesh_no_color.compute_vertex_normals()
        mesh_path_nocolor = os.path.join(mesh_path, "mesh_nocolor.ply")
        o3d.io.write_triangle_mesh(mesh_path_nocolor, mesh_no_color)

    num_frames = len(views)
    avg_time = total_time / num_frames if num_frames > 0 else 0
    fps = 1.0 / avg_time if avg_time > 0 else 0
    print(f"[{name}] Rendered {num_frames} frames in {total_time:.2f}s. Average FPS: {fps:.2f}")

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool,
                 max_depth : float, voxel_size : float, num_cluster: int, use_depth_filter : bool, args):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        # app_model = AppModel()
        # app_model.load_weights(scene.model_path)
        # app_model.eval()
        # app_model.cuda()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=4.0*voxel_size,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), scene, gaussians, pipeline, background, args, 
                       max_depth=max_depth, volume=volume, use_depth_filter=use_depth_filter)
            print(f"extract_triangle_mesh")
            mesh = volume.extract_triangle_mesh()

            path = os.path.join(dataset.model_path, "mesh")
            os.makedirs(path, exist_ok=True)
            
            o3d.io.write_triangle_mesh(os.path.join(path, "tsdf_fusion.ply"), mesh, 
                                       write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)
            
            mesh = post_process_mesh(mesh, num_cluster)
            o3d.io.write_triangle_mesh(os.path.join(path, "tsdf_fusion_post.ply"), mesh, 
                                       write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), scene, gaussians, pipeline, background, args)

if __name__ == "__main__":
    torch.set_num_threads(8)
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--max_depth", default=5.0, type=float)
    parser.add_argument("--voxel_size", default=0.002, type=float)
    parser.add_argument("--num_cluster", default=1, type=int)
    parser.add_argument("--mult", type=float, default=0.5)
    parser.add_argument("--use_depth_filter", action="store_true")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    print(f"multi_view_num {model.multi_view_num}")
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.max_depth, args.voxel_size, args.num_cluster, args.use_depth_filter,args)