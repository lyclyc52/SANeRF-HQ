#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

SANERFHQ_WORKSPACE_ROOT="workspace"
SANERFHQ_DATA_ROOT="/ssddata/yliugu/data"
SANERFHQ_SCENE="bonsai"
SANERFHQ_DATA_PATH="${SANERFHQ_DATA_ROOT}/${SANERFHQ_SCENE}"
SANERFHQ_MASK_PATH="${SANERFHQ_WORKSPACE_ROOT}/sam_nerf/${SANERFHQ_SCENE}/object_masks"
SANERFHQ_WORK_PATH="${SANERFHQ_WORKSPACE_ROOT}/obj_nerf/${SANERFHQ_SCENE}"
SANERFHQ_INIT_CKPT="${SANERFHQ_WORKSPACE_ROOT}/rgb_nerf/${SANERFHQ_SCENE}/checkpoints/ngp_ep0019.pth"

python main.py ${SANERFHQ_DATA_PATH} \
--mask_root ${SANERFHQ_MASK_PATH} \
--test_view_path example_test_views.json \
--workspace ${SANERFHQ_WORK_PATH} \
--init_ckpt ${SANERFHQ_INIT_CKPT} \
--enable_cam_center \
--with_mask \
--data_type mip \
--num_rays 6000 \
--iters 200 \
--contract \
--val_type val_split \
--ray_pair_rgb_loss_weight 1 \
--ray_pair_rgb_threshold 0.1 \
--ray_pair_rgb_iter 150 \
--ray_pair_rgb_num_sample 8 \
--local_sample_patch_size 8 \
--num_local_sample 4 \
--mixed_sampling \
--random_image_batch \
--error_map 