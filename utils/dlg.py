import torch
from .dataproc import data_lowrank, data_withnoise
from skimage import metrics
import numpy as np
import torchvision
from .utils import *


def DLGAttack(images, labels, net, criterion, original_dy_dx, ep):
    # generate dummy data and label
    images_lowrank, _ = data_lowrank( images )
    images_residual = images - images_lowrank
    images_noisy = data_withnoise( images_residual, 0.1)
    dummy_data = images_noisy.clone().requires_grad_(True)

    optimizer = torch.optim.Adam([dummy_data], lr=0.01)

    for i in range(10000):
        optimizer.zero_grad()
        x = dummy_data
        dummy_pred = net(x)
        dummy_loss = criterion(dummy_pred, labels)
        dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

        grad_diff = 0
        cur_grad = []
        for gx in dummy_dy_dx:
            cur_grad.append(torch.flatten(gx))
        cur_grad = torch.cat(cur_grad)
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        grad_diff = 1 - cos(cur_grad, original_dy_dx) + 0.001 * total_variation(dummy_data)
        # grad_diff = ((cur_grad - original_dy_dx) ** 2).sum()
        grad_diff.backward()
        # print(cos(cur_grad, original_dy_dx))
        # writer.add_scalar( 'Loss/loss',  grad_diff, i )
        optimizer.step()
        if i % 10000 == 9999:
            psnr = 0
            for k in range(images.size(0)):
                psnr1, _ = cal_metrics(images[k:k+1], torch.clip(dummy_data[k:k+1],0,1))
                psnr2, _ = cal_metrics(images[k:k+1], 1-torch.clip(dummy_data[k:k+1],0,1))
                psnr += max(psnr1, psnr2)
                torchvision.utils.save_image(images[k] * 255, f"./attack/image_{k}_{ep}.png")
                torchvision.utils.save_image((1-torch.clip(dummy_data[k:k+1],0,1)) * 255, f"./attack/recon_{k}_{ep}.png")
            print(grad_diff,  total_variation(dummy_data), psnr / images.size(0), images[k].size())


def cal_metrics( orig, recon ):
    """Calculate metrics such as PSNR and SSIM
    :param orig original data
    :param recon reconstructed data from the attack model
    """
    n_batch = orig.size( 0 )
    orig = orig.cpu().detach().numpy()
    orig = np.transpose( orig, ( 0,2,3,1 ) )
    recon = recon.cpu().detach().numpy()
    recon = np.transpose( recon, ( 0,2,3,1 ) )

    val_psnr, val_ssim = 0.0, 0.0
    for b in range( n_batch ):
        orig_i = orig[ b ]
        recon_i = recon[ b ]
        val_psnr += metrics.peak_signal_noise_ratio( orig_i, recon_i )
        val_ssim += metrics.structural_similarity( orig_i, recon_i, multichannel=True )

    return val_psnr/n_batch, val_ssim/n_batch