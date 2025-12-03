# 24-view 1/4 resolution 可以
python train.py -s /wtc/ssd/datasets/mipnerf360/bicycle -m output/mip12/bicycle --eval  --n_views 24 --sh_lower --abs  --grad_abs_thresh 0.003 -r 4
python train.py -s /wtc/ssd/datasets/mipnerf360/garden -m output/mip12/garden --eval  --n_views 24  --sh_lower --abs --highfeature_lr 0.02 --grad_abs_thresh 0.003 -r 4 --lowfeature_lr 0.0001
python train.py -s /wtc/ssd/datasets/mipnerf360/stump -m output/mip12/stump --eval  --n_views 24 --sh_lower --abs  --grad_abs_thresh 0.003 -r 4 --lowfeature_lr 0.0005
python train.py -s /wtc/ssd/datasets/mipnerf360/room -m output/mip12/room --eval  --n_views 24 --sh_lower --abs --highfeature_lr 0.02 --grad_abs_thresh 0.003 -r 4 --lowfeature_lr 0.0005
python train.py -s /wtc/ssd/datasets/mipnerf360/counter -m output/mip12/counter --eval  --n_views 24 --sh_lower --abs --highfeature_lr 0.02 --grad_abs_thresh 0.003 -r 4
python train.py -s /wtc/ssd/datasets/mipnerf360/kitchen -m output/mip12/kitchen --eval  --n_views 24 --sh_lower --abs --highfeature_lr 0.02 --grad_abs_thresh 0.003 -r 4 --lowfeature_lr 0.0005
python train.py -s /wtc/ssd/datasets/mipnerf360/bonsai -m output/mip12/bonsai --eval  --n_views 24 --sh_lower --abs --highfeature_lr 0.02 --grad_abs_thresh 0.003 -r 4 --lowfeature_lr 0.0001
python render.py -m output/mip12/bicycle -r 4
python render.py -m output/mip12/garden -r 4
python render.py -m output/mip12/stump -r 4
python render.py -m output/mip12/room -r 4
python render.py -m output/mip12/counter -r 4
python render.py -m output/mip12/kitchen -r 4
python render.py -m output/mip12/bonsai -r 4
python metrics.py -m output/mip12/bicycle
python metrics.py -m output/mip12/garden
python metrics.py -m output/mip12/stump
python metrics.py -m output/mip12/room
python metrics.py -m output/mip12/counter
python metrics.py -m output/mip12/kitchen
python metrics.py -m output/mip12/bonsai

# python train.py -s /wtc/ssd/datasets/mipnerf360/bicycle -m output/mip12/bicycle --eval  --n_views 12 --sh_lower --abs  --grad_abs_thresh 0.002 -r 4 --lowfeature_lr 0.0001
# python train.py -s /wtc/ssd/datasets/mipnerf360/garden -m output/mip12/garden --eval  --n_views 12  --sh_lower --abs --highfeature_lr 0.02 --grad_abs_thresh 0.002 -r 4 --lowfeature_lr 0.0001
# python train.py -s /wtc/ssd/datasets/mipnerf360/stump -m output/mip12/stump --eval  --n_views 12 --sh_lower --abs  --grad_abs_thresh 0.002 -r 4 --lowfeature_lr 0.0001
# python train.py -s /wtc/ssd/datasets/mipnerf360/room -m output/mip12/room --eval  --n_views 12 --sh_lower --abs --highfeature_lr 0.02 --grad_abs_thresh 0.002 -r 4 --lowfeature_lr 0.0001
# python train.py -s /wtc/ssd/datasets/mipnerf360/counter -m output/mip12/counter --eval  --n_views 12 --sh_lower --abs --highfeature_lr 0.02 --grad_abs_thresh 0.002 -r 4
# python train.py -s /wtc/ssd/datasets/mipnerf360/kitchen -m output/mip12/kitchen --eval  --n_views 12 --sh_lower --abs --highfeature_lr 0.02 --grad_abs_thresh 0.002 -r 4 --lowfeature_lr 0.0005
# python train.py -s /wtc/ssd/datasets/mipnerf360/bonsai -m output/mip12/bonsai --eval  --n_views 12 --sh_lower --abs --highfeature_lr 0.02 --grad_abs_thresh 0.002 -r 4 --lowfeature_lr 0.0001
# python render.py -m output/mip12/bicycle -r 4
# python render.py -m output/mip12/garden -r 4
# python render.py -m output/mip12/stump -r 4
# python render.py -m output/mip12/room -r 4
# python render.py -m output/mip12/counter -r 4
# python render.py -m output/mip12/kitchen -r 4
# python render.py -m output/mip12/bonsai -r 4
# python metrics.py -m output/mip12/bicycle
# python metrics.py -m output/mip12/garden
# python metrics.py -m output/mip12/stump
# python metrics.py -m output/mip12/room
# python metrics.py -m output/mip12/counter
# python metrics.py -m output/mip12/kitchen
# python metrics.py -m output/mip12/bonsai

# exp_name='mip12'
# scenes=("bicycle" "bonsai" "counter" "garden" "kitchen" "room" "stump")
# dataset_path='/home/rensw/Downloads/GS_dataset/mip-nerf360'
# n_views=12

# for scene in "${scenes[@]}"
# do
#   echo "Training on $scene..."
#   python train.py -s $dataset_path/$scene/ \
#     -m output/$exp_name/$scene \
#     --eval -r 8 \
#     --n_views $n_views

#   echo "Rendering $scene..."
#   python render.py -m output/$exp_name/$scene -r 8
# done

# Compute metrics for all scenes
python metric.py --path output/mip12