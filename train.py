import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from VDSR_dataloader import *
from VDSR_batchnorm import vdsr_bn
from VDSR_original import vdsr
from utils import *

#parameters
scale = 4
multiscale = True
mode = 'js'
patch_making = False
patch_size = 48
optimizer = 'Adam'
lr = 0.001
batch_size = 32
show_period = 500
train_3by3 = False

#patch making
if patch_making == True:
    make_patch('./train', patch_size, True)

# define model
if mode=='batchnorm':
    model = vdsr_bn()
elif mode=='original':
    model = vdsr()
elif mode=='js':
    model = JS_SR()
model.cuda()

# set optimizer
if optimizer == 'SGD':
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
elif optimizer =='Adam':
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)

criterion = nn.MSELoss()

running_loss = 0.0
val_loss = 0.0

if multiscale == False:
    # write loss, PSNR in txt file
    fid_val = open('./val_x%d.txt' % (scale), 'wt')
    fid_train = open('./train_x%d.txt' % (scale), 'wt')
    fid_psnr = open('./psnr_x%d.txt' % (scale), 'wt')
    fid_val.write('epoch\tval_loss\tPSNR\n')
    fid_train.write('epoch\titer\ttrain_loss\n')
    fid_psnr.write('All PSNR for epoch\n')
    fid_val.close()
    fid_train.close()
    fid_psnr.close()
    highest_PSNR = 0.0
    highest_epoch = 0.0
    trainloader = vdsr_trainloader(folder_path='./patch_%d' % (patch_size), batch_size=batch_size, transform=transforms.ToTensor(), scale=scale, shuffle=True)
    testloader, v_list = vdsr_testloader(folder_path='./test', transform=transforms.ToTensor(), scale=scale)
    v_list.sort()
    for ep in range(1000):
        # validation
        model.eval()
        fid_val = open('./val_x%d.txt' % (scale), 'at')
        fid_psnr = open('./psnr_x%d.txt' % (scale), 'at')
        with torch.no_grad():
            for j, v_data in enumerate(testloader):
                v_input, v_label = v_data
                v_input = v_input.cuda()
                v_label = v_label.cuda()
                v_output = model(v_input)
                v_loss = criterion(v_output, v_label)
                val_loss += v_loss.item()
            print('%d epoch validation loss : %f' % (ep, val_loss / len(testloader)))
            inference(model, testloader, v_list)
            all_psnr, psnr_mean = psnr_in_folder('./inference')
            print('%d epoch validation PSNR : %f' % (ep, psnr_mean))
            fid_val.write('%d\t%f\t%f\n' % (ep, val_loss/len(testloader), psnr_mean))
            fid_psnr.write('%d epoch\n' % (ep))
            val_loss = 0.0
            for l in range(len(v_list)):
                fid_psnr.write('%s\t%f\n' % (v_list[l], all_psnr[l]))
            if psnr_mean > highest_PSNR :
                highest_PSNR = psnr_mean
                highest_epoch = ep
            print('highest PSNR : %f, highest_epoch : %d' % (highest_PSNR, highest_epoch))
        fid_val.close()
        fid_psnr.close()

        # train
        model.train()
        for i, data in enumerate(trainloader):
            input, label = data
            input = Variable(input.cuda())
            label = label.cuda()
            output = model(input)
            if train_3by3 == True:
                output = output[:, :, (patch_size // 2 - 1):(patch_size // 2 + 1),
                           (patch_size // 2 - 1):(patch_size + 1)]
                label = label[:, :, (patch_size // 2 - 1):(patch_size // 2 + 1),
                          (patch_size // 2 - 1):(patch_size + 1)]
            optim.zero_grad()
            loss = criterion(output,label)
            loss.backward()
            if (mode == 'original') | (mode == 'js'):
                nn.utils.clip_grad_norm_(model.parameters(), 0.4)
            optim.step()
            running_loss += loss.item()
            if i%show_period == show_period-1:
                fid_train = open('train_x%d.txt' % (scale), 'at')
                print('%d epoch, %d iter loss : %f' % (ep+1, i+1, running_loss/show_period))
                fid_train.write('%d\t%d\t%f\n' % (ep+1, i+1, running_loss/show_period))
                running_loss = 0.0
                fid_train.close()
        if not(os.path.isdir('./%s_x%d' % (mode, scale))):
            os.makedirs('./%s_x%d' % (mode, scale))
        # save model
        torch.save(model, './%s_x%d/vdsr_%s_x%d_%depoch.pth' % (mode, scale, mode, scale, ep+1))

else:
    # write loss, PSNR in txt file
    for scale in range(2,5):
        fid_val = open('./val_x%d.txt' % (scale), 'wt')
        fid_psnr = open('./psnr_x%d.txt' % (scale), 'wt')
        fid_val.write('epoch\tval_loss\tPSNR\n')
        fid_psnr.write('All PSNR for epoch\n')
        fid_val.close()
        fid_psnr.close()
    fid_train = open('./train_multiscale.txt', 'wt')
    fid_train.write('epoch\titer\ttrain_loss\n')
    fid_train.close()
    trainloader = multiscale_trainloader(folder_path='./patch_%d' % (patch_size), batch_size=batch_size, transform=transforms.ToTensor(), shuffle=True)
    highest_PSNR = {'x2' : 0.0, 'x3' : 0.0, 'x4' : 0.0 }
    highest_epoch = {'x2' : 0, 'x3' : 0, 'x4' : 0}
    for ep in range(1000):
        # validation
        model.eval()
        for scale in range(2,5):
            fid_val = open('./val_x%d.txt' % (scale), 'at')
            fid_psnr = open('./psnr_x%d.txt' % (scale), 'at')
            testloader, v_list = vdsr_testloader(folder_path='./test', transform=transforms.ToTensor(), scale=scale)
            v_list.sort()
            with torch.no_grad():
                for j, v_data in enumerate(testloader):
                    v_input, v_label = v_data
                    v_input = v_input.cuda()
                    v_label = v_label.cuda()
                    v_output = model(v_input)
                    v_loss = criterion(v_output, v_label)
                    val_loss += v_loss.item()
                print('x%d scale %d epoch validation loss : %f' % (scale, ep, val_loss / len(testloader)))
                inference(model, testloader, v_list)
                all_psnr, psnr_mean = psnr_in_folder('./inference')
                print('x%d scale %d epoch validation PSNR : %f' % (scale, ep, psnr_mean))
                fid_val.write('%d\t%f\t%f\n' % (ep, val_loss/len(testloader), psnr_mean))
                fid_psnr.write('%d epoch\n' % (ep))
                val_loss = 0.0
                for l in range(len(v_list)):
                    fid_psnr.write('%s\t%f\n' % (v_list[l], all_psnr[l]))
                if psnr_mean > highest_PSNR['x%d' % (scale)] :
                    highest_PSNR['x%d' % (scale)] = psnr_mean
                    highest_epoch['x%d' % (scale)] = ep
                print('x%d scale highest PSNR : %f, highest_epoch : %d' % (scale, highest_PSNR['x%d' % (scale)], highest_epoch['x%d' % (scale)]))
            fid_val.close()
            fid_psnr.close()

        # train
        model.train()
        for i, data in enumerate(trainloader):
            input, label = data
            input = Variable(input.cuda())
            label = label.cuda()
            output = model(input)
            optim.zero_grad()
            if train_3by3 == True:
                output = output[:, :, (patch_size // 2 - 1):(patch_size // 2 + 1),
                           (patch_size // 2 - 1):(patch_size + 1)]
                label = label[:, :, (patch_size // 2 - 1):(patch_size // 2 + 1),
                          (patch_size // 2 - 1):(patch_size + 1)]
            loss = criterion(output,label)
            loss.backward()
            if (mode=='original') | (mode == 'js'):
                nn.utils.clip_grad_norm_(model.parameters(), 0.4)
            optim.step()
            running_loss += loss.item()
            if i%show_period == show_period-1:
                fid_train = open('train_multiscale.txt', 'at')
                print('%d epoch, %d iter loss : %f' % (ep+1, i+1, running_loss/show_period))
                fid_train.write('%d\t%d\t%f\n' % (ep+1, i+1, running_loss/show_period))
                running_loss = 0.0
                fid_train.close()
        if not(os.path.isdir('./multiscale_%s' % (mode))):
            os.makedirs('./multiscale_%s' % (mode))
        # save model
        torch.save(model, './js/js_multiscale_3by3_%s_%depoch.pth' % (mode, ep+1))