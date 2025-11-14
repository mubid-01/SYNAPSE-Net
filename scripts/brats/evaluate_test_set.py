import os
import glob
import numpy as np
import nibabel as nib
from tqdm import tqdm

from src.models.SYNAPSENet_4mod import SYNAPSENet_4mod as SYNAPSENet
from src.data_loaders.dataset_brats import BraTS_dataset
from src.utils.helpers.brats_helpers import compute_brats_metrics_detailed, post_process_volume

def evaluate_test_set(model_path, data_root, output_dir, device='cpu'):
    os.makedirs(output_dir, exist_ok=True)
    model = SYNAPSENet(chs=(64,96,128,256,384), token_dim=256, bottleneck_heads=8, aux_outs=2,
                         bottleneck_window=8, pre_swin_layers=4, dropout_rate=0.4,
                         num_modalities=4, num_classes=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    ds = BraTS_dataset(base_dir=data_root, split='val', transform=None, num_subjects=None)
    all_metrics = []
    with torch.no_grad():
        for i in tqdm(range(len(ds)), desc='Evaluating'):
            vols = ds._load_volume(i)
            mask_vol = vols['mask']
            mask_vol[mask_vol == 4] = 3
            pred_vol = np.zeros_like(mask_vol, dtype=np.uint8)
            for s in range(mask_vol.shape[2]):
                img_slice = np.stack([vols['t1c'][:,:,s], vols['t1n'][:,:,s], vols['t2f'][:,:,s], vols['t2w'][:,:,s]], axis=-1)
                
                img_tensor = torch.from_numpy(img_slice.astype(np.float32)).permute(2,0,1).unsqueeze(0).to(device)
                resized = F.interpolate(img_tensor, size=(208,208), mode='bilinear', align_corners=False)
                logits, _, _ = model(resized)
                logits_rs = F.interpolate(logits, size=img_tensor.shape[-2:], mode='bilinear', align_corners=False)
                preds = torch.argmax(torch.softmax(logits_rs, dim=1), dim=1)
                pred_vol[:,:,s] = preds.squeeze().cpu().numpy()

            for cls in [1,2,3]: pred_vol = post_process_volume(pred_vol, class_idx=cls, min_size=50)
            metrics = compute_brats_metrics_detailed(pred_vol, mask_vol)
            all_metrics.append(metrics)

    import pandas as pd
    df = pd.DataFrame(all_metrics)
    summary = df.mean().to_dict()
    print('Evaluation summary:', summary)
    return summary

if __name__ == '__main__':
    import torch
    import torch.nn.functional as F
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--out', default='./brats_eval')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    evaluate_test_set(args.model, args.data_root, args.out, args.device)
