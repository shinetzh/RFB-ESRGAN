import os
import cv2
import time
import torch
import shutil
import logging
import argparse
import numpy as np
import os.path as osp
from tqdm import tqdm
from collections import OrderedDict


import utils.util as util
from data.util import bgr2ycbcr
import options.options as option
from models import create_model
from data import create_dataset, create_dataloader



def split_img(img):

    _, c, h, w, = img.shape
    h_half = h // 2
    w_half = w // 2
    part_1 = img[:,:,0:h_half+10, 0:w_half+10]
    part_2 = img[:,:,0:h_half+10, w_half - 10:w]
    part_3 = img[:,:,h_half - 10:h, 0:w_half+10]
    part_4 = img[:,:,h_half - 10:h, w_half - 10:w]

    return part_1, part_2, part_3, part_4

def concat_img(part_1, part_2, part_3, part_4):

    c,h_part, w_part = part_1.shape
    h = (h_part - 160) * 2
    w = (w_part - 160) * 2
    h_half = h // 2
    w_half = w // 2
    out = torch.zeros(3, h, w)
    # out = np.zeros(shape=(3,h, w))
    out[:,0:h_half, 0:w_half] = part_1[:,0:h_half, 0:w_half]
    out[:,0:h_half, w_half:w] = part_2[:,0:h_half, 160:w_part]
    out[:,h_half:h, 0:w_half] = part_3[:,160:h_part, 0:w_half]
    out[:,h_half:h, w_half:w] = part_4[:,160:h_part, 160:w_part]
    return out


def test(opt,logger):
    logger.info(option.dict2str(opt))

    #### Create test dataset and dataloader
    test_loaders = []
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt)
        logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
        test_loaders.append(test_loader)

    model = create_model(opt)
    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info('\nTesting [{:s}]...'.format(test_set_name))
        test_start_time = time.time()
        dataset_dir = osp.join(opt['path']['results_root'], test_set_name)
        util.mkdir(dataset_dir)

        test_results = OrderedDict()
        test_results['psnr'] = []
        test_results['ssim'] = []
        test_results['psnr_y'] = []
        test_results['ssim_y'] = []

        time_start = time.time()
        for data in test_loader:
            need_GT = False if test_loader.dataset.opt['dataroot_GT'] is None else True


            part_1, part_2, part_3, part_4 = split_img(data['LQ'])
            part_1_1, part_1_2, part_1_3, part_1_4 = split_img(part_1)
            part_2_1, part_2_2, part_2_3, part_2_4 = split_img(part_2)
            part_3_1, part_3_2, part_3_3, part_3_4 = split_img(part_3)
            part_4_1, part_4_2, part_4_3, part_4_4 = split_img(part_4)
            start_time = time.time()


            for i in range(1,5):
                for j in range(1,5):
                    input = locals()['part_'+str(i)+'_'+str(j)]
                    model.feed_data(input, need_GT=False)
                    img_path = data['GT_path'][0] if need_GT else data['LQ_path'][0]
                    img_name = osp.splitext(osp.basename(img_path))[0]

                    model.test()
                    visuals = model.get_current_visuals(need_GT=need_GT)
                    globals()['out_'+str(i)+'_'+str(j)]=visuals['rlt']

            out_1 = concat_img(out_1_1, out_1_2, out_1_3, out_1_4)
            out_2 = concat_img(out_2_1, out_2_2, out_2_3, out_2_4)
            out_3 = concat_img(out_3_1, out_3_2, out_3_3, out_3_4)
            out_4 = concat_img(out_4_1, out_4_2, out_4_3, out_4_4)
            out = concat_img(out_1, out_2, out_3, out_4)

            sr_img = util.tensor2img(out)  # uint8
            # save images
            suffix = opt['suffix']
            if suffix:
                save_img_path = osp.join(dataset_dir, img_name + suffix + '.png')
            else:
                save_img_path = osp.join(dataset_dir, img_name + '.png')

            util.save_img(sr_img, save_img_path)

            # calculate PSNR and SSIM
            if need_GT:
                gt_img = util.tensor2img(visuals['GT'])
                sr_img, gt_img = util.crop_border([sr_img, gt_img], opt['scale'])
                psnr = util.calculate_psnr(sr_img, gt_img)
                ssim = util.calculate_ssim(sr_img, gt_img)
                test_results['psnr'].append(psnr)
                test_results['ssim'].append(ssim)

                if gt_img.shape[2] == 3:  # RGB image
                    sr_img_y = bgr2ycbcr(sr_img / 255., only_y=True)
                    gt_img_y = bgr2ycbcr(gt_img / 255., only_y=True)

                    psnr_y = util.calculate_psnr(sr_img_y * 255, gt_img_y * 255)
                    ssim_y = util.calculate_ssim(sr_img_y * 255, gt_img_y * 255)
                    test_results['psnr_y'].append(psnr_y)
                    test_results['ssim_y'].append(ssim_y)
                    logger.info(
                        '{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}; PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}.'.
                        format(img_name, psnr, ssim, psnr_y, ssim_y))
                else:
                    logger.info('{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}.'.format(img_name, psnr, ssim))
            else:
                logger.info(img_name)

        if need_GT:  # metrics
            # Average PSNR/SSIM results
            ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
            ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
            logger.info(
                '----Average PSNR/SSIM results for {}----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}\n'.format(
                    test_set_name, ave_psnr, ave_ssim))
            if test_results['psnr_y'] and test_results['ssim_y']:
                ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
                ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
                logger.info(
                    '----Y channel, average PSNR/SSIM----\n\tPSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}\n'.
                    format(ave_psnr_y, ave_ssim_y))
        time_end = time.time()
        logger.info('per image cost time:{}s'.format((time_end-time_start)/100))

def crop(source_dir,target_dir):
    def crop_img(img_path,targetpath):
        img = cv2.imread(img_path)
        height, width, channels = img.shape
        y_center,x_center = height//2,width//2
        crop_img = img[y_center-500:y_center+500, x_center-500:x_center+500]
        cv2.imwrite(targetpath, crop_img)

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    for filename in tqdm(os.listdir(source_dir)):
        filepath = os.path.join(source_dir,filename)
        targetpath = os.path.join(target_dir,filename)
        crop_img(filepath,targetpath)

def pre_crop(source_dir,target_dir):
    def crop_img(img_path,targetpath):
        img = cv2.imread(img_path)
        height, width, channels = img.shape
        y_center,x_center = height//2,width//2
        crop_img = img[y_center-32:y_center+32, x_center-32:x_center+32]
        cv2.imwrite(targetpath, crop_img)

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    for filename in tqdm(os.listdir(source_dir)):
        filepath = os.path.join(source_dir,filename)
        targetpath = os.path.join(target_dir,filename)
        crop_img(filepath,targetpath)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_images', type=str, default='./NTIRE2020_testLR', help='the test images dir')
    args = parser.parse_args()

    #### options
    optpath = './options/test/configs.yml'
    opt = option.parse(optpath, is_train=False)
    opt = option.dict_to_nonedict(opt)
    opt['datasets']['test_1']['dataroot_LQ']=args.test_images
    base_root = '.'

    model_path = os.path.join('./pth/model.pth')
    print('model_path:{}'.format(model_path))

    save_dir = 'full_resolution_images'
    crop_dir = './result_images'

    # setup logger
    util.mkdirs(
        (path for key, path in opt['path'].items()
        if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
    util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                    screen=True, tofile=True)
    logger = logging.getLogger('base')
    step_img_size_list=[]

    crop_source_dir = osp.join(opt['path']['results_root'], save_dir)
    crop_target_dir = osp.join(opt['path']['results_root'], crop_dir)
    opt['path']['pretrain_model_G'] = model_path
    opt['datasets']['test_1']['name'] = save_dir
    ###
    test(opt,logger)
    crop(crop_source_dir,crop_target_dir)
    for root, dirs, files in os.walk("."):
        for dir in dirs:
            if dir == '__pycache__':
                dir_path = os.path.join(root,dir)
                shutil.rmtree(dir_path)
    print('finished!')
