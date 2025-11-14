import os
import glob
import argparse
import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm
import torch.nn.functional as F

from src.models.SYNAPSENet_2mod import SYNAPSENet_2mod as SYNAPSENet


def generate_validation_probs(data_root=None, model_path=None, output_dir=None, roi_size=None, device=None, start_idx=None, end_idx=None):
    # Environment / default-aware config
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    default_work = os.path.join(repo_root, 'work')
    default_data = os.path.join(repo_root, 'data', 'isles')

    cfg = type('C', (), {})()
    cfg.data_root = data_root or os.environ.get('DATA_ROOT', default_data)
    cfg.model_path = model_path or os.environ.get('MODEL_PATH', os.path.join(default_work, 'models', 'best_model.pth'))
    cfg.output_dir = output_dir or os.environ.get('OUTPUT_DIR', os.path.join(default_work, "validation_probabilities"))
    cfg.chs = (64, 96, 128, 192, 256)
    cfg.token_dim = 256; cfg.swin_mlp_ratio = 2.0
    cfg.dropout_rate = 0.5; cfg.drop_path_rate = 0.3
    # roi_size may be passed as a tuple or as a comma-separated string from CLI
    if roi_size:
        if isinstance(roi_size, str):
            cfg.roi_size = tuple(map(int, roi_size.split(',')))
        else:
            cfg.roi_size = tuple(roi_size)
    else:
        cfg.roi_size = tuple(map(int, os.environ.get('ROI_SIZE', '208,208').split(',')))
    cfg.device = device or os.environ.get('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
    # optional slicing of subjects for chunked processing
    cfg.start_idx = int(start_idx) if start_idx is not None else int(os.environ.get('START_IDX', 0))
    cfg.end_idx = int(end_idx) if end_idx is not None else (int(os.environ.get('END_IDX', -1)) if os.environ.get('END_IDX') is not None else None)

    os.makedirs(cfg.output_dir, exist_ok=True)

    print(f"Loading model from: {cfg.model_path}")
    model = SYNAPSENet(
        chs=cfg.chs, token_dim=cfg.token_dim, swin_mlp_ratio=cfg.swin_mlp_ratio,
        dropout_rate=cfg.dropout_rate, drop_path_rate=cfg.drop_path_rate
    ).to(cfg.device)
    model.load_state_dict(torch.load(cfg.model_path, map_location=cfg.device))
    model.eval()

    all_subject_dirs = sorted([d for d in os.listdir(cfg.data_root) if d.startswith('sub-')])
    # apply optional start/end slicing
    test_subjects = all_subject_dirs[cfg.start_idx:cfg.end_idx]
    print(f"Found {len(test_subjects)} subjects for validation/testing (sliced {cfg.start_idx}:{cfg.end_idx}).")

    with torch.no_grad():
        for sub_dir in tqdm(test_subjects, desc="Generating Validation Probabilities"):
            try:
                # use a flexible glob to match .nii and .nii.gz variations
                dwi_pattern = os.path.join(cfg.data_root, sub_dir, 'ses-0001', 'dwi', '*_dwi.nii*')
                adc_pattern = os.path.join(cfg.data_root, sub_dir, 'ses-0001', 'dwi', '*_adc.nii*')
                dwi_path_list = sorted(glob.glob(dwi_pattern))
                adc_path_list = sorted(glob.glob(adc_pattern))

                if not dwi_path_list or not adc_path_list:
                    print(f"Warning: Missing DWI or ADC file for {sub_dir}. Skipping.")
                    continue

                dwi_path = dwi_path_list[0]
                adc_path = adc_path_list[0]

                original_nii = nib.load(dwi_path)
                dwi_vol = original_nii.get_fdata(dtype=np.float32)
                adc_vol = nib.load(adc_path).get_fdata(dtype=np.float32)
                
                pred_prob_vol = np.zeros_like(dwi_vol, dtype=np.float32)

                for s in range(dwi_vol.shape[2]):
                    img_slice = np.stack([dwi_vol[:,:,s], adc_vol[:,:,s]], axis=-1)
                    original_shape = img_slice.shape[:2]
                    
                    img_tensor = torch.from_numpy(img_slice.copy()).permute(2,0,1).float().unsqueeze(0).to(cfg.device)
                    resized_tensor = F.interpolate(img_tensor, size=cfg.roi_size, mode='bilinear', align_corners=False)

                    logits_original, _, _ = model(resized_tensor)
                    logits_flipped, _, _ = model(torch.flip(resized_tensor, dims=[3]))
                    main_logits = (logits_original + torch.flip(logits_flipped, dims=[3])) / 2.0
                    
                    resized_logits = F.interpolate(main_logits, size=original_shape, mode='bilinear', align_corners=False)
                    probs = torch.sigmoid(resized_logits)
                    pred_prob_vol[:,:,s] = probs.squeeze().cpu().numpy()

                output_filename = os.path.join(cfg.output_dir, f"{sub_dir}_prob.nii.gz")
                prob_nii = nib.Nifti1Image(pred_prob_vol.astype(np.float32), original_nii.affine, original_nii.header)
                nib.save(prob_nii, output_filename)

            except Exception as e:
                print(f"Skipping subject {sub_dir} due to an unexpected error: {e}")
            
    print(f"\nInference complete. High-quality probability maps saved to: {cfg.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate ISLES validation probability maps')
    parser.add_argument('--data_root', type=str, help='Path to ISLES dataset root')
    parser.add_argument('--model_path', type=str, help='Path to trained model file')
    parser.add_argument('--output_dir', type=str, help='Directory to save probability maps')
    parser.add_argument('--roi_size', type=str, help='ROI size, comma separated e.g. 208,208')
    parser.add_argument('--device', type=str, help='Device to run on (cpu or cuda)')
    args = parser.parse_args()

    generate_validation_probs(data_root=args.data_root, model_path=args.model_path, output_dir=args.output_dir, roi_size=args.roi_size, device=args.device)
