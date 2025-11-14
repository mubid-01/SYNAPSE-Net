import os
import glob
import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm
import torch.nn.functional as F

from src.models.SYNAPSE-Net_4mod import SYNAPSE-Net_4mod as SYNAPSENet
from src.utils.helpers.brats_helpers import post_process_volume

def run_final_test_inference(model_path, data_root, output_dir, device='cpu'):
    os.makedirs(output_dir, exist_ok=True)
    model = LACMAFNet(chs=(64,96,128,256,384), token_dim=256, bottleneck_heads=8, aux_outs=2,
                         bottleneck_window=8, pre_swin_layers=4, dropout_rate=0.4,
                         num_modalities=4, num_classes=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_subjects = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
    with torch.no_grad():
        for subj in tqdm(all_subjects, desc='Final Inference'):
            flair = glob.glob(os.path.join(data_root, subj, '*-t2f.nii*'))
            if not flair: continue
            vol = nib.load(flair[0]); img = vol.get_fdata(dtype=np.float32)
            pred_vol = np.zeros_like(img[...,0], dtype=np.uint8)
            for s in range(img.shape[2]):
                img_slice = img[:,:,s,:] if img.ndim==4 else img[:,:,s]
 
                t = torch.from_numpy(img_slice.astype(np.float32)).permute(2,0,1).unsqueeze(0).to(device)
                r = F.interpolate(t, size=(208,208), mode='bilinear', align_corners=False)
                logits,_,_ = model(r)
                logits = F.interpolate(logits, size=t.shape[-2:], mode='bilinear', align_corners=False)
                preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
                pred_vol[:,:,s] = preds.squeeze().cpu().numpy()

            for cls in [1,2,3]: pred_vol = post_process_volume(pred_vol, class_idx=cls, min_size=50)
            out_nii = nib.Nifti1Image(pred_vol, vol.affine, vol.header)
            nib.save(out_nii, os.path.join(output_dir, f"{subj}_pred.nii"))

if __name__ == '__main__':
    import argparse, torch
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--out', default='./brats_final_preds')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    run_final_test_inference(args.model, args.data_root, args.out, args.device)
