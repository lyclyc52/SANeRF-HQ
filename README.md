# SANeRF-HQ
[![arXiv](https://img.shields.io/badge/arXiv-2312.01531-f9f107.svg)](https://arxiv.org/abs/2312.01531) [<img src="https://img.shields.io/badge/Project-Page?logo=googledocs&logoColor=white&labelColor=gray">](https://lyclyc52.github.io/SANeRF-HQ/) [<img src="https://img.shields.io/badge/Cite-BibTex-orange">](#citation)

SANeRF-HQ: Segment Anything for NeRF in High Quality [CVPR 2024].

This is the official implementation of SANeRF-HQ.


## Set up
The code is based on [this repo](https://github.com/ashawkey/Segment-Anything-NeRF). 

First, install requirement packages
```bash
pip install -r requirements.txt
```
Also, you can build the extension (optional)
```bash
# install all extension modules
bash scripts/install_ext.sh

# if you want to install manually, here is an example:
cd raymarching
python setup.py build_ext --inplace # build ext only, do not install (only can be used in the parent directory)
pip install . # install to python path (you still need the raymarching/ folder, since this only install the built extension.)
```

## Dataset
We use the dataset from Mip-NeRF 360, LERF, LLFF, 3DFRONT, Panoptic Lifting and Contrastive Lift. You can download the dataset from their website by clicking the following hyperlinks. Also we provide one example [here](https://hkustconnect-my.sharepoint.com/:f:/g/personal/yliugu_connect_ust_hk/ElUwJv6ohvZFggKhp_ZHaKwBDlF0R1sLiqYJNlJjqesHvw?e=lhZNjO).

To switch different dataset, simply change the value of the flag `--data_type` during training.

 - [Mip-NeRF 360](https://jonbarron.info/mipnerf360/): `--data_type=colmap`.
 - [LERF](https://www.lerf.io/): `--data_type=colmap`. 
    
    Note: For LERF dataset, we do not obtain good NeRF reconstruction results by their camera poses (probably because of some hyper parameteres). Thus we use the colmap pose estimiation provided by [this](https://github.com/ashawkey/torch-ngp?tab=readme-ov-file#usage). Please following their instructions to run colmap first if you would like to test LERF. The corresponding scripts are also included in this repo.
 - [LLFF](https://jonbarron.info/mipnerf360/): `--data_type=llff`. We use the data provided by Mip-NeRF 360.
 - [3D-FRONT](https://github.com/lyclyc52/Instance_NeRF): `--data_type 3dfront`. We use the data provided by Instance NeRF
 - [Panoptic Lifting](https://github.com/nihalsid/panoptic-lifting) / [Contrastive Lift](https://github.com/yashbhalgat/Contrastive-Lift): `--data_type=others`.

For the evaluation masks we selected, you can download them [here](https://hkustconnect-my.sharepoint.com/:f:/g/personal/yliugu_connect_ust_hk/ElUwJv6ohvZFggKhp_ZHaKwBDlF0R1sLiqYJNlJjqesHvw?e=lhZNjO). Some datasets have ground truth segmentation (e.g. 3D-FRONT and Panoptic Lifting) so we directly use their annotation. For those without ground truth segmentation (e.g. Mip-NeRF 360), we randomly select some views and use [this](https://github.com/open-mmlab/playground) to obtain masks. Then, we pass the masks through [CascadePSP](https://github.com/hkchengrex/CascadePSP) for refinement if necessary. 


## Training

We provide some sample scripts to use our code. For the detailed description of each arguments, please refer to our code.

To train the RGB NeRF, run
```bash
bash scripts/train_rgb_nerf.sh
```


Then run the following script to obtain feature container.
```bash
bash scripts/train_sam_nerf.sh
```
You can change the container type by the flag`--feature_container`.


With the feature container, you can decode the object mask per image. 
```bash
bash scripts/decode.sh
```
In decoding, 3D points are required as input. To obtain 3D points, you can project 2D points onto 3D (The script is not provided but you can find the corresponding code in `test_step` in `nerf/train.py`) or use the GUI to select points.

To use the GUI, you should add `--gui` or you can run
```bash
bash scripts/gui.sh
```
Use you 


To train object field, run
```bash
bash scripts/train_obj_nerf.sh
```
Simply set `ray_pair_rgb_iter > iter` if you think that the ray pair rgb loss is slow or does not help in some cases. 


## Evaluation
To evaluate our results, you can run `scripts/test_obj_nerf.sh`. You can add `--use_default_intrinsics` in the test script to render mask with the default intrinsics. You can be download the evaluation views [here](https://hkustconnect-my.sharepoint.com/:f:/g/personal/yliugu_connect_ust_hk/ElUwJv6ohvZFggKhp_ZHaKwBDlF0R1sLiqYJNlJjqesHvw?e=lhZNjO)

## Other Results
In our paper, we demonstrate the potential of our pipeline to achieve various segmentation tasks. Here are some instructions about how we get those results.

### Text-prompt Segmentation
We use [Grounding-DINO](https://github.com/IDEA-Research/GroundingDINO) to generate the bounding box based on text and then use the bounding box as prompt for SAM to generate mask.

### Auto-segmentation and Dynamic Segmentation
We use [DEVA](https://github.com/hkchengrex/Tracking-Anything-with-DEVA) for a sequence of images in video. 

For static scene, you can first render a video from NeRF. You can utilize the 'save trajectory' function in GUI to store a sequence of camera poses. Click `start track` to start recoding the camera trajectory and click `save trajectory` to store it. Then put those frames into DEVA to help you obtain automatic segmentation results. Finally, you can use the code to train the object field. Remember to change `--n_inst` in multi-instance cases 

## Acknowledgement
- SAM and HQ-SAM
  ```bibtex
  @article{kirillov2023segany,
      title={Segment Anything},
      author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
      journal={arXiv:2304.02643},
      year={2023}
  }

  @inproceedings{sam_hq,
      title={Segment Anything in High Quality},
      author={Ke, Lei and Ye, Mingqiao and Danelljan, Martin and Liu, Yifan and Tai, Yu-Wing and Tang, Chi-Keung and Yu, Fisher},
      booktitle={NeurIPS},
      year={2023}
  }  
  ```

- torch-ngp
  ```bibtex
  @misc{torch-ngp,
      Author = {Jiaxiang Tang},
      Year = {2022},
      Note = {https://github.com/ashawkey/torch-ngp},
      Title = {Torch-ngp: a PyTorch implementation of instant-ngp}
  }
  ```

- CascadePSP
  ```bibtex
  @inproceedings{cheng2020cascadepsp,
    title={{CascadePSP}: Toward Class-Agnostic and Very High-Resolution Segmentation via Global and Local Refinement},
    author={Cheng, Ho Kei and Chung, Jihoon and Tai, Yu-Wing and Tang, Chi-Keung},
    booktitle={CVPR},
    year={2020}
  }
  ```

- OpenMMLab Playground: https://github.com/open-mmlab/playground


## Citation
If you find this repo or our paper useful, please :star: this repository and consider citing :pencil::
```bibtex
@article{liu2023sanerf,
  title={SANeRF-HQ: Segment Anything for NeRF in High Quality},
  author={Liu, Yichen and Hu, Benran and Tang, Chi-Keung and Tai, Yu-Wing},
  journal={arXiv preprint arXiv:2312.01531},
  year={2023}
}
```