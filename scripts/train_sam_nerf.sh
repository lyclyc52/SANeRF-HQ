#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

SANERFHQ_WORKSPACE_ROOT="workspace"
SANERFHQ_DATA_ROOT="/ssddata/yliugu/data"
SANERFHQ_SCENE="bonsai"
SANERFHQ_DATA_PATH="${SANERFHQ_DATA_ROOT}/${SANERFHQ_SCENE}"
SANERFHQ_WORK_PATH="${SANERFHQ_WORKSPACE_ROOT}/sam_nerf/${SANERFHQ_SCENE}"
SANERFHQ_INIT_CKPT="${SANERFHQ_WORKSPACE_ROOT}/rgb_nerf/${SANERFHQ_SCENE}/checkpoints/ngp_ep0019.pth"

python main.py ${SANERFHQ_DATA_PATH} \
--workspace ${SANERFHQ_WORK_PATH} \
--with_sam \
--init_ckpt ${SANERFHQ_INIT_CKPT} \
--data_type mip \
--iters 5000 \
--contract \
--feature_container cache \
--enable_cam_center \
--sam_use_view_direction \


