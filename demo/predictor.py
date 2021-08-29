import os
import torch
import torch.nn.functional as F

from mmdet.apis import init_detector
from mmdet.core import tensor2im, save_image, mkdirs
import mmcv
from mmcv.parallel import collate, scatter

from mmdet.datasets.pipelines import Compose


class LoadImage(object):

    def __call__(self, results):
        if isinstance(results['img'], str):
            results['filename'] = results['img']
        else:
            results['filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


def save_results(outputs, img_meta, path=None):
    "save the testing completed results to disk"
    filename = img_meta[0].data[0][0]['filename']
    content = filename.split('/')
    scene_path = path+'/scene/'+content[-3]
    object_path = path+'/object/'+content[-3]
    mkdirs(scene_path)
    mkdirs(object_path)
    layer = 1
    number = 0
    for data in outputs:
        completed_scene, object_des = data
        sce_numpy = tensor2im(completed_scene)
        sce_name = content[-1].split('.')[0]
        name = sce_name + '_' + format(layer, '02d') + '.png'
        img_path = os.path.join(scene_path, name)
        save_image(sce_numpy, img_path)
        if object_des is not None:
            for object_de_index in object_des:
                if len(object_de_index) == 2:
                    object_de, index = object_de_index
                else:
                    object_de = object_de_index
                    index = number
                object_numpy = tensor2im(object_de)
                name = sce_name + '_' + format(index+1, '02d') + '.png'
                img_path = os.path.join(object_path, name)
                save_image(object_numpy, img_path)
                number +=1
        layer +=1


def show_results(model, outputs, data, path=None):
    """
    show or save the testing decomposition results
    :param outputs: final decomposition results
    :param data: original input data information
    :param path: save path
    :return:
    """
    file_name = data['img_meta'][0].data[0][0]['filename']
    content = file_name.split('/')
    if path is not None:
        mkdirs(path + '/decomposition/')
        out_file = path + '/decomposition/' + content[-3] + '_' + content[-1]
        model.show_result(data, outputs, score_thr=0.8, pre_order=True, show=False, out_file=out_file)
    else:
        model.show_result(data, outputs, score_thr=0.3, pre_order=True, show=True)


def compute_predictor(model, img, save_path='.'):
    """ predict the amodal instance and complete the occlusion
    :param model: The loaded model
    :param img: loaded images
    :param iteration: the largest iterations time
    :return: de_results: structured scene decomposition result
             co_results: completed individual objects and background in each step
    """

    cfg = model.cfg
    device = next(model.parameters()).device # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    input_data = scatter(data, [device])[0]
    # forward the model
    with torch.no_grad():
        # model.CLASSES = ('thing', 'stuff')
        de_result, co_result = model(return_loss=False, rescale=True, **input_data)
    if len(co_result) > 0:
        save_results(co_result, data['img_meta'], path=save_path+'/completion')
    if len(de_result) > 0:
        show_results(model, de_result, data, path=save_path)

    return de_result, co_result

config_file = '../configs/rgba/csd/lbl_completed_decomposition_htc_csd.py'
checkpoint_file ='/media/lyndon/2e91762c-97d9-40c9-9af1-6f318aca4771/program/mine/viv_v2/mmdetection/tools/work_dirs/final_models_tpami/lbl_completion_decomposition_htc_csd_finetune_merged/latest.pth'
save_path = '/media/lyndon/2e91762c-97d9-40c9-9af1-6f318aca4771/results/VIV_TPAMI/csd'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
# img = '/media/lyndon/c6f4bbbd-8d47-4dcb-b0db-d788fe2b2557/dataset/KINS/testing/image_2/001072.png'
img = '/media/lyndon/c6f4bbbd-8d47-4dcb-b0db-d788fe2b2557/dataset/coco/train2014/COCO_train2014_000000013720.jpg'
# img = '/media/lyndon/c6f4bbbd-8d47-4dcb-b0db-d788fe2b2557/dataset/suncg_data/sosc_new/017721df60943af0aa37f37fbd5a7af7/re_rgb/06.png'
de_result, co_result = compute_predictor(model, img, save_path=save_path)
# import glob
# img_list = sorted(glob.glob('/media/lyndon/2e91762c-97d9-40c9-9af1-6f318aca4771/dataset/image_depth/NYU_Test/testA/*.bmp'))
# for img in img_list:
#     de_result, co_result = compute_predictor(model, img, save_path=save_path)
