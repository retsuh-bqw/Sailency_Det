import argparse
import logging
from tqdm import tqdm
import os
import time
import yaml
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2

from utils.dataset import Salicon
from utils.loss import ModMSELoss
from test import test
from torchsummary import summary

from MLNet import MLNet

logger = logging.getLogger(__name__)

def train(model, train_data, val_data, args):

    criterion = ModMSELoss(shape_r_gt=60, shape_c_gt=80)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.epoch * 0.8, args.epoch * 0.9])
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.0001, step_size_up=10, step_size_down=3900, mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=True, base_momentum=0.8, max_momentum=0.9, last_epoch=- 1, verbose=False)


    device = next(model.parameters()).device
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=8)
    bilinearup = torch.nn.UpsamplingBilinear2d(scale_factor=8)
    min_test_loss = math.inf
    for epoch in range(args.epoch):
        train_loss = 0
        train_acc = 0
        train_n = 0
        model = model.train()
        with tqdm(train_loader) as loader:
            for ori_img, data, label in loader:
                loader.set_description(f"Epoch {epoch+1}")
                
                optimizer.zero_grad()
                data, label= data.to(device), label.to(device)

                outputs = model(data)     
                loss = criterion(outputs, label, model.prior.clone())
                
                # Backward and optimize
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * label.size(0)
                train_n += label.shape[0]
                loader.set_postfix(loss=loss.item())

                # i = 1
                # plt.figure(dpi=300,figsize=(24,8))
                # for file_name in ori_img:
                #     image_test = cv2.imread(file_name)
                #     print(image_test.shape)
                    
                #     saliency_map = bilinearup(outputs)[i-1].cpu().detach().numpy() * 255#.astype('uint8')
                #     gt_map = bilinearup(label[0])[0].cpu().numpy()
                #     print(saliency_map.shape)
                #     # _, saliency_map = cv2.threshold(saliency_map.squeeze(0), 0 , 255, cv2.THRESH_OTSU)
                #     print(saliency_map.shape)

                #     plt.subplot(1, 3, i), plt.imshow(cv2.cvtColor(image_test, cv2.COLOR_BGR2RGB))
                #     plt.title('Original', fontsize=30)
                #     plt.subplot(1, 3, 1+i), plt.imshow(gt_map.squeeze(0), cmap='gray')
                #     plt.title('GT',  fontsize=30)
                #     plt.subplot(1, 3, 2+i), plt.imshow(saliency_map.squeeze(0), cmap='gray')
                #     plt.title('Output',  fontsize=30)
                    
                #     # plt.title('Saliency Map')
                #     i += 3
                # plt.tight_layout()
                # plt.savefig('MLNet_7.png')
                # os._exit(1)

        cur_test_loss = test(model, val_data, args)
        if cur_test_loss < min_test_loss:
            min_test_loss = cur_test_loss
            saved_name = '{0}-{1}-{2}.pt'.format('Best-', 'LMNet', str(epoch))
            torch.save(model.state_dict(), os.path.join(args.save_path, saved_name))

        
    localtime = time.asctime( time.localtime(time.time()) )
    saved_name = '{0}-{1}-{2}.pt'.format('Last-', 'LMNet_', localtime)
    torch.save(model.state_dict(), os.path.join(args.save_path, saved_name))
        
        
        


if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='Adversarial Test')
    parser.add_argument('--save_path', type=str, default='./checkpoint', help='saved weight path')
    parser.add_argument('--pretrained', type=str, default=None, help='pretrained model path')
    parser.add_argument('--data_root', type=str, default='./data', help='dataset path')
    parser.add_argument('--epoch', type=int, default=10, help='epoch size')
    parser.add_argument('--batch_size', type=int, default=8, help='mini-batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='weight decay')
    parser.add_argument('--out_dir', type=str, default='./train_log/', help='log output dir')
    args = parser.parse_args()

    localtime = time.asctime( time.localtime(time.time()) )
    logfile = os.path.join(args.out_dir, 'MLNet' + localtime + '.log')
    if os.path.exists(logfile):
        os.remove(logfile)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=logfile)
    logger.info(args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = Salicon(data='train')
    val_data = Salicon(data='val')

    model = MLNet((6,8)).cuda()
    if args.pretrained != None:
        model.load_state_dict(torch.load(args.pretrained), strict=True)

    freeze_layers = 5
    for i,param in enumerate(model.parameters()):
        if i < freeze_layers:
            param.requires_grad = False
    # summary(model, (13,1450))

    train(model, data, val_data, args)