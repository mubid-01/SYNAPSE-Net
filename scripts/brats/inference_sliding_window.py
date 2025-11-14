import os
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from tqdm import tqdm

from src.models.SYNAPSE-Net_4mod import SYNAPSE-Net_4mod as SYNAPSENet

def inference_sliding_window(model_path, input_nifti, output_nifti, roi_size=(208,208), device='cpu'):
    model = LACMAFNet(chs=(64,96,128,256,384), token_dim=256, bottleneck_heads=8, aux_outs=2,
                         bottleneck_window=8, pre_swin_layers=4, dropout_rate=0.4,
                         num_modalities=4, num_classes=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    vol = nib.load(input_nifti)
    img = vol.get_fdata(dtype=np.float32)
    if img.ndim == 4 and img.shape[2] == 4:
        pass

    pred = np.zeros_like(img[...,0], dtype=np.uint8)
    with torch.no_grad():
        for s in tqdm(range(img.shape[2]), desc='Slices'):
            slice_img = img[:,:,s,:] if img.ndim==4 else img[:,:,s]

            t = torch.from_numpy(slice_img.astype(np.float32)).permute(2,0,1).unsqueeze(0).to(device)
            r = F.interpolate(t, size=roi_size, mode='bilinear', align_corners=False)
            logits,_,_ = model(r)
            logits = F.interpolate(logits, size=t.shape[-2:], mode='bilinear', align_corners=False)
            preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            pred[:,:,s] = preds.squeeze().cpu().numpy()

    out = nib.Nifti1Image(pred, vol.affine, vol.header)
    nib.save(out, output_nifti)

if __name__ == '__main__':
    import argparse, torch
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    inference_sliding_window(args.model, args.input, args.output, device=args.device)
