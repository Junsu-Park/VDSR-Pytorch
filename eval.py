import torch
import torch.nn as nn
import torchvision.transforms as transforms
from VDSR_dataloader import vdsr_testloader
from utils import *

#parameters
scale = 4

testloader, v_list = vdsr_testloader(folder_path='./test', transform=transforms.ToTensor(), scale=scale)
model = torch.load('./3by3/vdsr_multiscale_3by3_batchnorm_100epoch.pth')
model.eval().cpu()
criterion = nn.MSELoss()

val_loss = 0.0
highest_PSNR = 0.0

with torch.no_grad():
    for j, v_data in enumerate(testloader):
        v_input, v_label = v_data
        v_output = model(v_input)
        v_loss = criterion(v_output, v_label)
        val_loss += v_loss.item()
    print('validation loss : %f' % (val_loss / len(testloader)))
    inference(model, testloader, v_list)
    all_psnr, psnr_mean = psnr_in_folder('./inference')
    for i,p in enumerate(all_psnr):
        print('%s PSNR : %f' % (v_list[i], p))
    print('validation PSNR : %f' % (psnr_mean))