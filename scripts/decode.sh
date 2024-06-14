#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

SANERFHQ_WORKSPACE_ROOT="workspace"
SANERFHQ_DATA_ROOT="/ssddata/yliugu/data"
SANERFHQ_SCENE="bonsai"
SANERFHQ_DATA_PATH="${SANERFHQ_DATA_ROOT}/${SANERFHQ_SCENE}"
SANERFHQ_WORK_PATH="${SANERFHQ_WORKSPACE_ROOT}/sam_nerf/${SANERFHQ_SCENE}"
SANERFHQ_INIT_CKPT="${SANERFHQ_WORKSPACE_ROOT}/rgb_nerf/${SANERFHQ_SCENE}/checkpoints/ngp_ep0019.pth"

python main.py ${SANERFHQ_DATA_PATH} \
--workspace ${SANERFHQ_WORK_PATH} \
--init_ckpt ${SANERFHQ_INIT_CKPT} \
--enable_cam_center \
--data_type mip \
--test \
--test_split val \
--val_type val_all \
--with_sam \
--num_rays 8192 \
--contract \
--sam_use_view_direction \
--feature_container cache \
--decode \
--use_point \
--point_file example_points.json \