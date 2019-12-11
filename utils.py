import torch
import os
import glob
import math
import numpy as np
from torchvision.utils import save_image
from PIL import Image


def make_patch(folder_path, patch_size, augmentation = False):
    if not(os.path.isdir('./patch_%d' % (patch_size))):
        os.makedirs('./patch_%d' % (patch_size))
    for f in glob.iglob(os.path.join(folder_path, '*')):
        im = Image.open(f)
        w, h = im.size
        num_row = w//patch_size
        num_col = h//patch_size
        for row in range(num_row):
            for col in range(num_col):
                im_crop = im.crop([patch_size*row, patch_size*col, patch_size*(row+1), patch_size*(col+1)])
                im_crop.save('./patch_%d/%d_%s' % (patch_size, row*num_col + col + 1, os.path.split(im.filename)[-1]))
                if augmentation == True:
                    im_hflip = im_crop.transpose(method=Image.FLIP_LEFT_RIGHT)
                    im_vflip = im_crop.transpose(method=Image.FLIP_TOP_BOTTOM)
                    im_rotate = im_crop.transpose(method=Image.ROTATE_90)
                    im_hflip.save('./patch_%d/%d_hflip_%s' % (patch_size, row*num_col + col + 1, os.path.split(im.filename)[-1]))
                    im_vflip.save('./patch_%d/%d_vflip_%s' % (patch_size, row * num_col + col + 1, os.path.split(im.filename)[-1]))
                    im_rotate.save('./patch_%d/%d_rotate_%s' % (patch_size, row * num_col + col + 1, os.path.split(im.filename)[-1]))
    print('all patches are generated')

def inference(model, testloader, name):
    model = model
    model.cuda()
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(testloader):
            input, label = data
            input = input.cuda()
            label = label.cuda()
            output = model(input)
            output = output.squeeze().cpu()
            label = label.squeeze().cpu()
            if not (os.path.isdir('./inference')):
                os.makedirs('./inference')
            if not (os.path.isdir('./inference_GT')):
                os.makedirs('./inference_GT')
            save_image(output, './inference/%s.png' % (os.path.splitext(name[i])[0]))
            save_image(label, './inference_GT/%s_GT.png' % (os.path.splitext(name[i]))[0])

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def psnr_in_folder(folder_path):
    pred_list = os.listdir(folder_path)
    pred_list.sort()
    images = []
    labels = []
    all_psnr = []

    for n in pred_list:
        f1 = os.path.join(folder_path, n)
        im = Image.open(f1)
        np_im=np.array(im, dtype='float32')
        np_im.astype('float32')

        images.append(np_im)
        f2 = os.path.join(folder_path + '_GT', os.path.splitext(n)[0] + '_GT' + os.path.splitext(n)[1])
        lb = Image.open(f2)
        np_lb = np.array(lb, dtype='float32')
        np_lb.astype('float32')
        labels.append(np_lb)

    for i in range(len(pred_list)):
        all_psnr.append(psnr(images[i], labels[i]))

    psnr_mean = sum(all_psnr)/len(pred_list)
    return all_psnr, psnr_mean
