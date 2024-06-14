import cv2
import numpy as np
import os
import os.path as path
import json


def get_image_name_ours_rgb(img_root, object_name, scene_name, data_type, img_id):
    img_name = os.path.join(img_root, f'{scene_name}-{object_name}-nerf-rgb', 'results', f'{img_id}_mask.npy')
    inference = np.load(img_name)
                    
    inference = inference.argmax(-1)
    return inference

def get_image_name_ours(img_root, object_name, scene_name, data_type, img_id):
    img_name = os.path.join(img_root, f'{scene_name}-{object_name}-nerf', 'results', f'{img_id}_mask.npy')
    inference = np.load(img_name)
                    
    inference = inference.argmax(-1)
    return inference
def get_image_name_ours_sam(img_root, object_name, scene_name, data_type, img_id):
    img_name = os.path.join(img_root, f'{scene_name}-{object_name}-sam', 'results', f'{img_id}_mask.npy')
    inference = np.load(img_name)
                    
    inference = inference.argmax(-1)
    return inference

def get_image_name_ours_hq_sam_nerf(img_root, object_name, scene_name, data_type, img_id):
    img_name = os.path.join(img_root, f'{scene_name}-{object_name}-hq_sam_nerf', 'results', f'{img_id}_mask.npy')
    inference = np.load(img_name)
                    
    inference = inference.argmax(-1)
    return inference

def get_image_name_ours_hq_sam(img_root, object_name, scene_name, data_type, img_id):
    img_name = os.path.join(img_root, f'{scene_name}-{object_name}-hq_sam', 'results', f'{img_id}_mask.npy')
    inference = np.load(img_name)
    inference = inference.argmax(-1)
    return inference

def get_image_name_sa3d(img_root, object_name, scene_name, data_type, img_id):

    root = 'nerf_unbounded'
    img_name = os.path.join(img_root, root, f'dvgo_{scene_name}', f'render_test_{object_name}', 'masked_img', f'rgb_{img_id}.png')
    masked_img = cv2.imread(img_name)
    
    img_name = os.path.join(img_root, root, f'dvgo_{scene_name}', f'render_test_{object_name}', 'ori_img', f'{img_id}.png')
    ori_img = cv2.imread(img_name)

    diff = np.abs(masked_img - ori_img * 0.3).sum(-1)
    
    # print(np.unique(diff))
    
    # cv2.imwrite('diff.png', diff)
    inference = diff > 5
    
    return inference

# def get_image_name_sa3d(img_root, object_name, scene_name, data_type, img_id):

#     root = 'nerf_unbounded'
#     img_name = os.path.join(img_root, root, f'dvgo_{scene_name}', f'render_test_{object_name}', 'masked_only_img', f'rgb_{img_id}.png')
#     print(img_name)
#     masked_img = cv2.imread(img_name)
    

    
#     inference = masked_img<128
    
#     return inference


def get_image_name_isrf(img_root, object_name, scene_name, data_type, img_id):

    img_name = os.path.join(img_root, f'{scene_name}_{object_name}', 'test', f'{img_id}.png')
    # print(img_name)
    inference = cv2.imread(img_name)[..., 0] 
    
    inference = inference  >  0
    
    return inference
get_name_fucntion_dict ={
    'ours': get_image_name_ours,
    'sa3d': get_image_name_sa3d,
    'isrf': get_image_name_isrf,
    'ours_rgb': get_image_name_ours_rgb,
    'ours_sam': get_image_name_ours_sam,
    'ours_hq_sam_nerf': get_image_name_ours_hq_sam_nerf,
    'ours_hq_sam': get_image_name_ours_hq_sam,
}

get_img_root_dict = {
    
    'ours': '/ssddata/yliugu/trial_model_final/mask_nerf',

    'ours_rgb': '/ssddata/yliugu/trial_model_final/mask_nerf_rgb',
    'ours_sam': '/ssddata/yliugu/trial_model_final/mask_nerf_sam',
    'ours_hq_sam': '/ssddata/yliugu/trial_model_final/mask_nerf_hq_sam',
    'ours_hq_sam_nerf': '/ssddata/yliugu/trial_model_final/mask_nerf_hq_sam_nerf',
    'sa3d': '/ssddata/yliugu/SegmentAnythingin3D/logs',
    # model_name = 
    
    'isrf': '/ssddata/yliugu/isrf_code/masks'
    # model_name = 
}

def main(model_root, model_name='ours'):
    
     
    mask_data_root = '/ssddata/yliugu/selected_masks'
    meta_path = '/ssddata/yliugu/Segment-Anything-NeRF/scenes_metadata_v2.json'
    scene_path = '/ssddata/yliugu/Segment-Anything-NeRF/scene_list.json'
    eval_scene_path = '/ssddata/yliugu/Segment-Anything-NeRF/scenes_test_view.json'
    
    
    get_name_fucntion = get_name_fucntion_dict[model_name]
    
    with open(scene_path) as f:
        scene_dict = json.load(f)

    with open(meta_path) as f:
        meta = json.load(f)
    
    with open(eval_scene_path) as f:
        eval_scene_json  = json.load(f)
        
        
    for data_type in list(scene_dict.keys()):
        data_type = 'llff'
        scene_list = scene_dict[data_type]
        total_acc = 0
        total_iou = 0
        
        obj_count = 0

        for scene_name in scene_list:
            
            
            # scene_name = 'ctr_lift_2'
            scene_data_root = path.join(mask_data_root, scene_name)
            
            for object_name in meta[scene_name]:     

                gt_mask_folder = path.join(scene_data_root, object_name)
                
                eval_img_names = eval_scene_json[scene_name][object_name]
                
                # if len(eval_img_names) < 10 and data_type != 'llff':
                #     print(scene_name, object_name)
                    
                    
                cur_iou = 0
                cur_acc = 0
                cur_count = 0
                
                cur_intersection = 0
                cur_union = 0   
                cur_correct = 0
                cur_total = 0
                img_root = get_img_root_dict[model_name]  
                
                 
                
                for eval_img in eval_img_names:
                    inference = get_name_fucntion(img_root,object_name,scene_name,data_type,img_id=eval_img)
                    
                    # non = lambda s: s if s<0 else None
                    # mom = lambda s: max(0,s)

                    # ox, oy = 0, -20
                    # shift_lena = np.zeros_like(inference)
                    # shift_lena[mom(oy):non(oy), mom(ox):non(ox)] = inference[mom(-oy):non(-oy), mom(-ox):non(-ox)]
                    # inference = shift_lena
                    
                    
                    # gt_path = path.join(gt_mask_folder, f'pred_mask_{eval_img}.png')
                    # if not os.path.isfile(gt_path):
                    #     print('yes')
                    gt_path = path.join(gt_mask_folder, f'{eval_img}_mask.png')
                    gt_img = cv2.imread(gt_path)[..., 0]
                    

                    # print(gt_img.shape)
                    # print(inference.shape)
                    # exit()
                    if inference.shape[0] != gt_img.shape[0]:
                        assert abs(inference.shape[0] / gt_img.shape[0] - inference.shape[1] / gt_img.shape[1]) < 0.1
                        gt_img = cv2.resize(gt_img, (inference.shape[1], inference.shape[0]))
                        
                    gt_img = gt_img > 128            
                    cur_intersection += (inference * gt_img).sum()
                    cur_union += ((inference + gt_img) > 0).sum()
                    inference_flatten = inference.reshape(-1)
                    gt_flatten = gt_img.reshape(-1)
                    false_pred = np.logical_xor(inference_flatten, gt_flatten).sum()
                    cur_total += inference_flatten.shape[0]
                    cur_correct += (inference_flatten.shape[0] - false_pred)
                    
                obj_acc = cur_correct / cur_total
                obj_iou = cur_intersection / cur_union
                
                # print()
                # print(f'{scene_name}_{object_name} acc: {(cur_correct / cur_total)}')
                # print(f'{scene_name}_{object_name} iou: {(cur_intersection / cur_union)}')
                
                
                # if model_name == 'ours_rgb': 
                #     cur_intersection = 0
                #     cur_union = 0   
                #     cur_correct = 0
                #     cur_total = 0
                #     for eval_img in eval_img_names:
                #         inference = get_name_fucntion_dict['ours'](get_img_root_dict['ours'] ,object_name,scene_name,data_type,img_id=eval_img)
                #         gt_path = path.join(gt_mask_folder, f'pred_mask_{eval_img}.png')
                        
                #         if not os.path.isfile(gt_path):
                #             print('yes')
                #             gt_path = path.join(gt_mask_folder, f'{eval_img}_mask.png')
                #         gt_img = cv2.imread(gt_path)[..., 0]
                #         if inference.shape[0] != gt_img.shape[0]:
                #             assert abs(inference.shape[0] / gt_img.shape[0] - inference.shape[1] / gt_img.shape[1]) < 0.1
                #             gt_img = cv2.resize(gt_img, (inference.shape[1], inference.shape[0]))
                            
                #         gt_img = gt_img > 128            
                #         cur_intersection += (inference * gt_img).sum()
                #         cur_union += ((inference + gt_img) > 0).sum()
                #         inference_flatten = inference.reshape(-1)
                #         gt_flatten = gt_img.reshape(-1)
                #         false_pred = np.logical_xor(inference_flatten, gt_flatten).sum()
                #         cur_total += inference_flatten.shape[0]
                #         cur_correct += (inference_flatten.shape[0] - false_pred)
                #     old_obj_acc = cur_correct / cur_total
                #     old_obj_iou = cur_intersection / cur_union
                    

               
                obj_count += 1
                total_acc += obj_acc
                total_iou += obj_iou
                print(f'{scene_name}_{object_name} acc: {obj_acc} iou: {obj_iou}')
        
            
        print(f'{data_type}:')
        print(f'acc: ', total_acc / obj_count)
        print(f'miou: ', total_iou / obj_count)
        break
    return

def eval_iou(inference, gt):


    intersection = (inference * gt).sum()
    union = ((inference + gt) > 0).sum()
    if union == 0:
        if intersection == 0:
            return 1
        else:
            return 0
        
    return intersection / union

def eval_acc(inference, gt):
    
    inference_flatten = inference.reshape(-1)
    gt_flatten = gt.reshape(-1)
    
    false_pred = np.logical_xor(inference_flatten, gt_flatten).sum()
    total = inference_flatten.shape[0]
    
    
    return 1. - false_pred/ total

if __name__ == '__main__':
    model_root = '/ssddata/yliugu/trial_model_final/mask_nerf'
    
    img_root = '/ssddata/yliugu/trial_model_final/mask_nerf'
    # model_name = 'ours'

    # img_root = '/ssddata/yliugu/trial_model_final/mask_nerf_rgb'
    # model_name = 'ours_rgb'
    
    # model_name = 'ours_sam'
    # model_name = 'ours_hq_sam'
    # model_name = 'ours_hq_sam_nerf'
    
    img_root = '/ssddata/yliugu/SegmentAnythingin3D/logs'
    model_name = 'sa3d'
    
    # img_root = '/ssddata/yliugu/isrf_code/masks'
    # model_name = 'isrf'
    
    main(model_root, model_name)
    
    
