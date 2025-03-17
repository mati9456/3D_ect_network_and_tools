import scipy
import torch
import numpy as np
import time
from grokfast import gradfilter_ma, gradfilter_ema
import torch.nn as nn           


def load_mask(reduce=1, device="cuda"):
    mask_flat = scipy.io.loadmat("fov_ix.mat")['fov_ix']
    mask = np.zeros(64**3)
    mask[mask_flat] = 1
    mask = mask.reshape(64, 64, 64)

    mask = mask[::reduce, ::reduce, ::reduce]

    mask = np.rot90(mask, 1, (0, 2))
    #plot_volume(mask, 0.5, reduce=4)
    mask = torch.tensor(mask.copy()).to(torch.float32).to(device)
    
    return mask


def initialize_weights(model, init_type="xavier"):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
            if init_type == "xavier":
                nn.init.xavier_uniform_(m.weight)
            elif init_type == "kaiming":
                nn.init.kaiming_uniform_(m.weight)
            elif init_type == "normal":
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif init_type == "uniform":
                nn.init.uniform_(m.weight, -0.1, 0.1)
            elif init_type == "zero":
                nn.init.constant_(m.weight, 0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    

def train_model(model, device, mask, num_epochs, train_dataloader, val_dataloader, criterion, optimizer, scheduler, batch_aggregation=1, disp_freq=100, eval=True, psize=64):
    model.train().to(device)
    grads = None
    
    training_loss_arr = []
    valid_loss_arr = []
    lr_arr = []

    for epoch in range(num_epochs):
        print('Starting epoch {}...'.format(epoch+1))

        model.train()
        epoch_start_time = time.time()
        running_loss_g1 = 0.0
        minibatch_start_time = time.time()
        for i, (x_batch, y_batch) in enumerate(train_dataloader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device).view(-1, 1, psize, psize, psize)
                    
            fake_images = model(x_batch)

            # only calculate gradient for effective volume
            mask_batch = mask.unsqueeze(0).expand(y_batch.shape[0], 1, psize, psize, psize)

            g_loss = criterion(fake_images*mask_batch, y_batch*mask_batch)*2.1
            g_loss.backward()

            running_loss_g1 += g_loss.item()

            if (i+1)%batch_aggregation == 0:
                grads = gradfilter_ema(model, grads=grads, alpha=0.98, lamb=2.0)
                optimizer.step()
                optimizer.zero_grad()

            # print statistics
            if i % disp_freq == disp_freq-1:
                print('[%d, %5d] loss g: %.5f' %(epoch + 1, i + 1, running_loss_g1 / disp_freq))
                miniBatch_time = time.time() - minibatch_start_time
                print('minibatch finished, took {:.5f}s'.format(miniBatch_time))
                training_loss_arr.append(running_loss_g1 / disp_freq)
                running_loss_g1 = 0.0
                minibatch_start_time = time.time()

        if eval:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_batch, y_batch in val_dataloader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device).view(-1, 1, psize, psize, psize)
                    fake_images = model(x_batch)

                    mask_batch = mask.unsqueeze(0).expand(x_batch.shape[0], 1, psize, psize, psize)
                    loss = criterion(fake_images*mask_batch, y_batch*mask_batch)*2.1
                    
                    val_loss += loss.item()

            val_loss /= len(val_dataloader)
            valid_loss_arr.append(val_loss)
            print(f"Eval loss is: {val_loss}")

            if scheduler is not None:
                scheduler.step(val_loss)
                
            lr_arr.append(optimizer.param_groups[0]['lr'])

        print('Epoch finished, took {:.5f}s'.format(time.time() - epoch_start_time))
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")
        
    return model, training_loss_arr, valid_loss_arr, lr_arr


def train_model_once_cycle(model, device, mask, num_epochs, train_dataloader, val_dataloader, criterion, optimizer, batch_aggregation=1, disp_freq=100, eval=True, psize=64, eval_mod=1, disable_1cycle=False):
    model.train().to(device)
    if disable_1cycle is False:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=int(len(train_dataloader)/batch_aggregation), epochs=num_epochs)
    grads = None
    
    training_loss_arr = []
    valid_loss_arr = []
    lr_arr = []

    for epoch in range(num_epochs):
        print('Starting epoch {}...'.format(epoch+1))

        model.train()
        epoch_start_time = time.time()
        running_loss_g1 = 0.0
        minibatch_start_time = time.time()
        for i, (x_batch, y_batch) in enumerate(train_dataloader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device).view(-1, 1, psize, psize, psize)
                    
            fake_images = model(x_batch)

            # only calculate gradient for effective volume
            mask_batch = mask.unsqueeze(0).expand(y_batch.shape[0], 1, psize, psize, psize)

            g_loss = criterion(fake_images*mask_batch, y_batch*mask_batch)*2.1
            g_loss.backward()

            running_loss_g1 += g_loss.item()

            if (i+1)%batch_aggregation == 0:
                grads = gradfilter_ema(model, grads=grads, alpha=0.98, lamb=2.0)
                optimizer.step()
                optimizer.zero_grad()
                if disable_1cycle is False:
                    scheduler.step()
                lr_arr.append(optimizer.param_groups[0]['lr'])

            # print statistics
            if i % disp_freq == disp_freq-1:
                print('[%d, %5d] loss g: %.5f' %(epoch + 1, i + 1, running_loss_g1 / disp_freq))
                miniBatch_time = time.time() - minibatch_start_time
                print('minibatch finished, took {:.5f}s'.format(miniBatch_time))
                training_loss_arr.append(running_loss_g1 / disp_freq)
                running_loss_g1 = 0.0
                minibatch_start_time = time.time()

        if eval and (epoch+1)%eval_mod==0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_batch, y_batch in val_dataloader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device).view(-1, 1, psize, psize, psize)
                    fake_images = model(x_batch)

                    mask_batch = mask.unsqueeze(0).expand(x_batch.shape[0], 1, psize, psize, psize)
                    loss = criterion(fake_images*mask_batch, y_batch*mask_batch)*2.1
                    
                    val_loss += loss.item()

            val_loss /= len(val_dataloader)
            valid_loss_arr.append(val_loss)
            print(f"Eval loss is: {val_loss}")

            lr_arr.append(optimizer.param_groups[0]['lr'])

        print('Epoch finished, took {:.5f}s'.format(time.time() - epoch_start_time))
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")
        
    return model, training_loss_arr, valid_loss_arr, lr_arr