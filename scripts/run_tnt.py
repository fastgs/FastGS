import os

scenes = ['Truck', 'Caterpillar', 'Meetingroom', 'Ignatius']
data_devices = ['cuda', 'cuda', 'cuda', 'cuda']
# scenes = ['Barn']
# data_devices = ['cuda']
data_base_path='data/tnt_dataset/tnt'
out_base_path='output_tnt'
out_name='test'
gpu_id=0

for id, scene in enumerate(scenes):

    cmd = f'rm -rf {out_base_path}/{scene}/*'
    print(cmd)
    os.system(cmd)
    
    common_args = f"-r2 --ncc_scale 0.5 --opacity_cull_threshold 0.05 --exposure_compensation --densification_interval 500 --test_iterations 30000 --highfeature_lr 0.02 --grad_abs_thresh 0.0004 --mult 0.7 --eval"
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py -s {data_base_path}/{scene} -m {out_base_path}/{scene} {common_args}'
    print(cmd)
    os.system(cmd)
    
    # 0.0002梯度for barn
    # common_args = f"-r2 --ncc_scale 0.5 --opacity_cull_threshold 0.05 --exposure_compensation --densification_interval 500 --test_iterations 30000 --highfeature_lr 0.02 --grad_abs_thresh 0.0002 --mult 0.7 --eval"
    # cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py -s {data_base_path}/{scene} -m {out_base_path}/{scene} {common_args}'
    # print(cmd)
    # os.system(cmd)

    common_args = f"--num_cluster 1 --use_depth_filter --mult 0.7 --max_depth 10.0 --voxel_size 0.01"
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python scripts/render_tnt.py -m {out_base_path}/{scene} --data_device {data_devices[id]} {common_args}'
    print(cmd)
    os.system(cmd)

    cmd = f'python metrics.py -m {out_base_path}/{scene}'
    print(cmd)
    os.system(cmd)

    # require open3d==0.9
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python scripts/tnt_eval/run.py --dataset-dir {data_base_path}/{scene} --traj-path {data_base_path}/{scene}/{scene}_COLMAP_SfM.log --ply-path {out_base_path}/{scene}/mesh/tsdf_fusion_post.ply --out-dir {out_base_path}/{scene}/mesh'
    print(cmd)
    os.system(cmd)