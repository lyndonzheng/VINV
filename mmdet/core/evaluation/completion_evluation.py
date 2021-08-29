import numpy as np
import argparse
from PIL import Image
import cv2
import glob
from skimage.measure import compare_ssim

from mmdet.datasets.csd import CSD
from pycocotools import mask as maskUtils

parser = argparse.ArgumentParser(description='Evaluation on the dataset')
parser.add_argument('--gt_path', type = str, default='/media/lyndon/c6f4bbbd-8d47-4dcb-b0db-d788fe2b2557/dataset/suncg_data',
                    help = 'path to original ground truth data')
parser.add_argument('--pre_path', type = str, default='/media/lyndon/2e91762c-97d9-40c9-9af1-6f318aca4771/results/VIV_TPAMI/lbl_completion_decomposition_htc_csd_pre_mask_gtde/completion/',
                    help='path to save the predicted dataset')
parser.add_argument('--num_test', type=int, default=1000,
                    help='how many images to load for each test')
args = parser.parse_args()


def computer_matric(img_gt, img_test, mask):
    """
    :param img_gt: original unmasked image
    :param img_test: generated test image
    :param mask: the object regions
    :return: mae, ssim, psnr
    """

    img_gt = img_gt.astype(np.float32) / 255.0
    img_test = img_test.astype(np.float32) /255.0

    # rmse for valid regions
    valid = np.count_nonzero(mask)
    if valid == 0:
        rmse = 0
    else:
        rmse = np.sqrt(np.sum(np.square(img_gt - img_test)) / valid)

    # psnr for valid regions
    if rmse < 1e-5:
        psnr = 100
    else:
        psnr = 20 * np.log10(1.0 / rmse)

    # ssim for valid regions
    ssim, S = compare_ssim(img_gt, img_test, multichannel=True, win_size=11, full=True)
    valid_ssim = np.where(mask > 0, S, np.zeros_like(S))
    if valid == 0:
        ssim = 1
    else:
        ssim = np.sum(valid_ssim) / valid

    return rmse, ssim, psnr


def completion_with_gt_f(sosc):
    imgIds = sosc.getImgIds()
    maes = []
    ssims = []
    psnrs = []
    for i in range(0, len(imgIds)):
        id = imgIds[i]
        annIds = sosc.getAnnIds(imgIds=id)
        anns = sosc.loadAnns(annIds)
        for ann in anns:
            # if ann['layer_order'] == 0:
            #     continue
            gt_img_name = ann['f_img_name']
            gt_object = Image.open(args.gt_path + '/sosc_new/' + gt_img_name)
            gt_numpy = np.array(gt_object)  # .astype(np.float32) / 255
            gt_object.close()
            v_mask_name = ann['v_mask_name']
            v_mask = Image.open(args.gt_path + '/sosc_new/' + v_mask_name)
            v_mask_numpy = np.array(v_mask)
            v_mask.close()
            content = gt_img_name.split('/')
            scene_name = content[0]
            img_content = content[-1].split('_')
            img_name = img_content[0]
            object_number = img_content[1]
            if 'BG' in content[-1]:
                pre_img_name = args.pre_path + '/scene/' + scene_name + '/' + img_name + '_' + object_number + '.png'
            else:
                pre_img_name = args.pre_path + '/object/' + scene_name + '/' + img_name + '_' + object_number + '.png'
            # pre_img_name = args.pre_path + '/'+ scene_name + '/' + img_name + '_' + object_number + '.png'
            try:
                pre_object = Image.open(pre_img_name).resize((512, 512))
            except:
                continue
            pre_numpy = np.array(pre_object)  # / 255
            pre_object.close()
            mask = gt_numpy[:, :, -1]
            dila = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
            dilation = cv2.erode(mask, dila, iterations=1)
            mask = dilation.repeat(3).reshape(gt_numpy.shape[0], gt_numpy.shape[1], 3)
            # v_mask_numpy = v_mask_numpy.reshape(gt_numpy.shape[0], gt_numpy.shape[1], 1)
            gt_numpy = np.where(mask > 0, gt_numpy[:, :, :-1], np.zeros_like(gt_numpy[:, :, :-1]))
            pre_numpy = np.where(mask > 0, pre_numpy[:, :, :-1], np.zeros_like(pre_numpy[:, :, :-1]))
            # pre_numpy = np.where(mask == 255, pre_numpy, np.zeros_like(pre_numpy))
            mse_temp, ssim_temp, psnr_temp = computer_matric(gt_numpy, pre_numpy, mask)
            print(mse_temp, ssim_temp, psnr_temp)

            if mse_temp != 100 and ssim_temp != 0 and psnr_temp != 0:
                maes.append(mse_temp)
                ssims.append(ssim_temp)
                psnrs.append(psnr_temp)

    print('{:>10},{:>10},{:>10}'.format('MAE', 'SSIM', 'PSNR'))
    print('{:10.4f},{:10.4f},{:10.4f}'.format(np.mean(maes), np.mean(ssims), np.mean(psnrs)))
    print('{:10.4f},{:10.4f},{:10.4f}'.format(np.var(maes), np.var(ssims), np.var(psnrs)))


def compltion_with_pred_f(sosc):
    imgIds = sosc.getImgIds()
    maes = []
    ssims = []
    psnrs = []
    for i in range(0, len(imgIds)):
        id = imgIds[i]
        # get the completed shape
        image = sosc.loadImgs(imgIds[i])[0]
        scene_name = image['img_name']
        content = scene_name.split('/')
        scene_name = content[0]
        img_content = content[-1].split('.')
        img_name = img_content[0]
        annIds = sosc.getAnnIds(imgIds=id)
        object_paths = sorted(glob.glob(args.pre_path + '/object/' + scene_name + '/' + img_name + '*.png'))
        # object_paths = sorted(glob.glob(args.pre_path + '/' + scene_name + '/' + img_name + '*.png'))
        pre_objects = []
        pre_rles = []
        for i, path in enumerate(object_paths):
            try:
                pre_object = Image.open(path).resize((512, 512))
            except:
                continue
            pre_numpy = np.array(pre_object)
            pre_object.close()
            mask = (pre_numpy[:,:,-1] > 0).astype(np.uint8)
            rle = maskUtils.encode(np.array(mask[:,:,np.newaxis], order='F'))[0]
            pre_rles.append(rle)
            pre_objects.append(pre_numpy)
        anns = sosc.loadAnns(annIds)
        for ann in anns:
            gt_img_name = ann['f_img_name']
            gt_object = Image.open(args.gt_path + '/sosc_new/' + gt_img_name)
            gt_numpy = np.array(gt_object)  # .astype(np.float32) / 255
            gt_object.close()
            gt_mask = (gt_numpy[:,:,-1]>0).astype(np.uint8)
            gt_rle = maskUtils.encode(np.array(gt_mask[:,:,np.newaxis], order='F'))
            # match the largest iou for testing
            if 'BG' in gt_img_name and 'SeGAN' not in args.pre_path:
                bg_paths = sorted(glob.glob(args.pre_path + '/scene/' + scene_name + '/' + img_name + '*.png'))
                pre_object = Image.open(bg_paths[-1]).resize((512, 512))
                pre_numpy = np.array(pre_object)
                pre_object.close()
            else:
                ious = maskUtils.iou(pre_rles, gt_rle, [0])
                max_ious_index = np.argmax(ious)
                pre_numpy = pre_objects[max_ious_index]
            dila = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
            dilation = cv2.erode(gt_mask, dila, iterations=1)
            gt_mask = dilation.repeat(3).reshape(gt_numpy.shape[0], gt_numpy.shape[1], 3)
            gt_numpy = np.where(gt_mask > 0, gt_numpy[:, :, :-1], np.zeros_like(gt_numpy[:, :, :-1]))
            if pre_numpy.shape[2] == 4:
                pre_numpy = np.where(gt_mask > 0, pre_numpy[:, :, :-1], np.zeros_like(pre_numpy[:, :, :-1]))
            else:
                pre_numpy = np.where(gt_mask > 0, pre_numpy, np.zeros_like(pre_numpy))
            mse_temp, ssim_temp, psnr_temp = computer_matric(gt_numpy, pre_numpy, gt_mask)

            if mse_temp != 0 and ssim_temp != 1 and psnr_temp != 100:
                print(mse_temp, ssim_temp, psnr_temp)
                maes.append(mse_temp)
                ssims.append(ssim_temp)
                psnrs.append(psnr_temp)

    print('{:>10},{:>10},{:>10}'.format('MAE', 'SSIM', 'PSNR'))
    print('{:10.4f},{:10.4f},{:10.4f}'.format(np.mean(maes), np.mean(ssims), np.mean(psnrs)))
    print('{:10.4f},{:10.4f},{:10.4f}'.format(np.var(maes), np.var(ssims), np.var(psnrs)))


if __name__ == "__main__":

    annFile = '{}/sosc_new_test_order.json'.format(args.gt_path)
    csd = CSD(annFile)

    completion_with_gt_f(csd)
    # compltion_with_pred_f(csd)