# llff 3-views
python train.py -s ./data/nerf_llff_data/fern -m output/llff3/fern --eval  --n_views 3  --lowfeature_lr 0.0001 --highfeature_lr 0.005 --grad_abs_thresh 0.003 -r 8
python train.py -s ./data/nerf_llff_data/flower -m output/llff3/flower --eval  --n_views 3 --lowfeature_lr 0.0001  --grad_abs_thresh 0.003 -r 8
python train.py -s ./data/nerf_llff_data/fortress -m output/llff3/fortress --eval  --n_views 3 --lowfeature_lr 0.0001 --highfeature_lr 0.005 --grad_abs_thresh 0.003 -r 8
python train.py -s ./data/nerf_llff_data/horns -m output/llff3/horns --eval  --n_views 3 --lowfeature_lr 0.0001  --grad_abs_thresh 0.003 -r 8 
python train.py -s ./data/nerf_llff_data/leaves -m output/llff3/leaves --eval  --n_views 3 --lowfeature_lr 0.0001  --grad_abs_thresh 0.003 -r 8
python train.py -s ./data/nerf_llff_data/orchids -m output/llff3/orchids --eval  --n_views 3 --lowfeature_lr 0.0001  --grad_abs_thresh 0.003 -r 8
python train.py -s ./data/nerf_llff_data/room -m output/llff3/room --eval  --n_views 3 --lowfeature_lr 0.0001  --grad_abs_thresh 0.003 -r 8
python train.py -s ./data/nerf_llff_data/trex -m output/llff3/trex --eval  --n_views 3 --lowfeature_lr 0.0001  --grad_abs_thresh 0.003 -r 8
python render.py -m output/llff3/fern -r 8
python render.py -m output/llff3/flower -r 8
python render.py -m output/llff3/fortress -r 8
python render.py -m output/llff3/horns -r 8
python render.py -m output/llff3/leaves -r 8
python render.py -m output/llff3/orchids -r 8
python render.py -m output/llff3/room -r 8
python render.py -m output/llff3/trex -r 8
python metrics.py -m output/llff3/fern
python metrics.py -m output/llff3/flower
python metrics.py -m output/llff3/fortress
python metrics.py -m output/llff3/horns
python metrics.py -m output/llff3/leaves
python metrics.py -m output/llff3/orchids
python metrics.py -m output/llff3/room
python metrics.py -m output/llff3/trex

# python metric.py --path output/llff3

# llff 6-views
python train.py -s ./data/nerf_llff_data/fern -m output/llff6/fern --eval  --n_views 6  --highfeature_lr 0.005 --grad_abs_thresh 0.003 -r 8
python train.py -s ./data/nerf_llff_data/flower -m output/llff6/flower --eval  --n_views 6   --grad_abs_thresh 0.003 -r 8
python train.py -s ./data/nerf_llff_data/fortress -m output/llff6/fortress --eval  --n_views 6 --highfeature_lr 0.005 --grad_abs_thresh 0.003 -r 8
python train.py -s ./data/nerf_llff_data/horns -m output/llff6/horns --eval  --n_views 6   --grad_abs_thresh 0.003 -r 8 
python train.py -s ./data/nerf_llff_data/leaves -m output/llff6/leaves --eval  --n_views 6   --grad_abs_thresh 0.003 -r 8
python train.py -s ./data/nerf_llff_data/orchids -m output/llff6/orchids --eval  --n_views 6  --grad_abs_thresh 0.003 -r 8
python train.py -s ./data/nerf_llff_data/room -m output/llff6/room --eval  --n_views 6   --grad_abs_thresh 0.003 -r 8
python train.py -s ./data/nerf_llff_data/trex -m output/llff6/trex --eval  --n_views 6   --grad_abs_thresh 0.003 -r 8
python render.py -m output/llff6/fern -r 8
python render.py -m output/llff6/flower -r 8
python render.py -m output/llff6/fortress -r 8
python render.py -m output/llff6/horns -r 8
python render.py -m output/llff6/leaves -r 8
python render.py -m output/llff6/orchids -r 8
python render.py -m output/llff6/room -r 8
python render.py -m output/llff6/trex -r 8
python metrics.py -m output/llff6/fern
python metrics.py -m output/llff6/flower
python metrics.py -m output/llff6/fortress
python metrics.py -m output/llff6/horns
python metrics.py -m output/llff6/leaves
python metrics.py -m output/llff6/orchids
python metrics.py -m output/llff6/room
python metrics.py -m output/llff6/trex

# python metric.py --path output/llff6

# llff 9-views
python train.py -s ./data/nerf_llff_data/fern -m output/llff9/fern --eval  --n_views 9 --highfeature_lr 0.005  --grad_abs_thresh 0.003 -r 8
python train.py -s ./data/nerf_llff_data/flower -m output/llff9/flower --eval  --n_views 9  --grad_abs_thresh 0.003 -r 8
python train.py -s ./data/nerf_llff_data/fortress -m output/llff9/fortress --eval  --n_views 9  --highfeature_lr 0.005 --grad_abs_thresh 0.003 -r 8
python train.py -s ./data/nerf_llff_data/horns -m output/llff9/horns --eval  --n_views 9 --grad_abs_thresh 0.003 -r 8 
python train.py -s ./data/nerf_llff_data/leaves -m output/llff9/leaves --eval  --n_views 9  --grad_abs_thresh 0.001 -r 8
python train.py -s ./data/nerf_llff_data/orchids -m output/llff9/orchids --eval  --n_views 9  --grad_abs_thresh 0.003 -r 8
python train.py -s ./data/nerf_llff_data/room -m output/llff9/room --eval  --n_views 9  --grad_abs_thresh 0.001 -r 8
python train.py -s ./data/nerf_llff_data/trex -m output/llff9/trex --eval  --n_views 9  --grad_abs_thresh 0.003 -r 8
python render.py -m output/llff9/fern -r 8
python render.py -m output/llff9/flower -r 8
python render.py -m output/llff9/fortress -r 8
python render.py -m output/llff9/horns -r 8
python render.py -m output/llff9/leaves -r 8
python render.py -m output/llff9/orchids -r 8
python render.py -m output/llff9/room -r 8
python render.py -m output/llff9/trex -r 8
python metrics.py -m output/llff9/fern
python metrics.py -m output/llff9/flower
python metrics.py -m output/llff9/fortress
python metrics.py -m output/llff9/horns
python metrics.py -m output/llff9/leaves
python metrics.py -m output/llff9/orchids
python metrics.py -m output/llff9/room
python metrics.py -m output/llff9/trex

# python metric.py --path output/llff9