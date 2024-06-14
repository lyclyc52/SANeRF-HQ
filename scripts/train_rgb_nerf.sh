#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

SANERFHQ_WORKSPACE_ROOT="workspace"
SANERFHQ_DATA_ROOT="/ssddata/yliugu/data"
SANERFHQ_SCENE="bonsai"
SANERFHQ_DATA_PATH="${SANERFHQ_DATA_ROOT}/${SANERFHQ_SCENE}"
SANERFHQ_WORK_PATH="${SANERFHQ_WORKSPACE_ROOT}/rgb_nerf/${SANERFHQ_SCENE}"

python main.py ${SANERFHQ_DATA_PATH} \
--workspace ${SANERFHQ_WORK_PATH} \
--enable_cam_center \
--downscale 4 \
--data_type mip \
--iters 5000 \
--contract \
--random_image_batch