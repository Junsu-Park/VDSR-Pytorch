import torch
from PIL import Image
import os
import torchvision.transforms as transforms
import random
from tqdm import tqdm

def vdsr_trainloader(folder_path, batch_size, transform, scale=2, shuffle=True):
    outputs = []
    input_batch = torch.tensor([])
    label_batch = torch.tensor([])
    file_list = os.listdir(folder_path)
    if shuffle==True:
        random.shuffle(file_list)
    for i, f in enumerate(file_list):
        im = Image.open(os.path.join(folder_path, f))
        w, h = im.size
        resized_im = transforms.Resize([h, w], Image.BICUBIC)(transforms.Resize([h//scale, w//scale], Image.BICUBIC)(im))
        im_tensor = transform(im).unsqueeze(0)
        resized_tensor = transform(resized_im).unsqueeze(0)
        input_batch = torch.cat([input_batch, resized_tensor], 0)
        label_batch = torch.cat([label_batch, im_tensor], 0)
        if (i % batch_size == batch_size - 1) | (i == len(file_list) - 1):
            outputs.append([input_batch, label_batch])
            input_batch = torch.tensor([])
            label_batch = torch.tensor([])
    return outputs

def multiscale_trainloader(folder_path, batch_size, transform, shuffle=True):
    outputs = []
    list = []
    input_batch = torch.tensor([])
    label_batch = torch.tensor([])
    file_list = os.listdir(folder_path)
    for f in file_list:
        im = Image.open(os.path.join(folder_path, f))
        w, h = im.size
        for scale in range(2,5):
            resized_im = transforms.Resize([h, w], Image.BICUBIC)(transforms.Resize([h // scale, w // scale], Image.BICUBIC)(im))
            list.append([resized_im, im])
    if shuffle == True:
        random.shuffle(list)
    for i, [input, label] in enumerate(list):
        input_tensor = transform(input).unsqueeze(0)
        label_tensor = transform(label).unsqueeze(0)
        input_batch = torch.cat([input_batch, input_tensor], 0)
        label_batch = torch.cat([label_batch, label_tensor], 0)
        if (i % batch_size == batch_size - 1) | (i == len(list) - 1):
            outputs.append([input_batch, label_batch])
            input_batch = torch.tensor([])
            label_batch = torch.tensor([])
    return outputs

def vdsr_testloader(folder_path, transform, scale=2):
    outputs = []
    file_list = os.listdir(folder_path)
    file_list.sort()
    for f in tqdm(file_list):
        im = Image.open(os.path.join(folder_path,f))
        w, h = im.size
        resized_im = transforms.Resize([h, w], Image.BICUBIC)(transforms.Resize([h//scale, w//scale], Image.BICUBIC)(im))
        outputs.append([transform(resized_im).unsqueeze(0), transform(im).unsqueeze(0)])

    return outputs, file_list