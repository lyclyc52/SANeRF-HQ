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
--test \
--test_split val \
--val_type val_split \
--data_type mip \
--contract \
--use_default_intrinsics \
--return_extra \
--render_mask_instance_id 1
