from tqdm import tqdm
import os
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.dataset import Salicon
from utils.loss import ModMSELoss
from torchsummary import summary

from MLNet import MLNet


def test(model, val_data, args):

    model.eval()
    criterion = ModMSELoss(shape_r_gt=60, shape_c_gt=80)

    device = next(model.parameters()).device
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=True, num_workers=8)

    val_loss = 0
    val_n = 0
    with tqdm(val_loader) as loader:
        for ori_img, data, label in loader:

            data, label= data.to(device), label.to(device)

            outputs = model(data)     
            loss = criterion(outputs, label, model.prior.clone())
            

            val_loss += loss.item() * label.size(0)
            loader.set_postfix(loss=loss.item())
    
    return val_loss