import os
import glob
import argparse
import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm
import torch.nn.functional as F

from src.models.SYNAPSENet_2mod import SYNAPSENet_2mod as SYNAPSENet
from src.utils.utils import post_process_volume


def run_final_test_inference(data_root=None, model_path=None, output_dir=None, threshold=None, min_lesion_size=None, roi_size=None, device=None):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    default_work = os.path.join(repo_root, 'work')
    default_data = os.path.join(repo_root, 'data', 'isles')

    cfg = type('C', (), {})()
    cfg.OPTIMAL_THRESHOLD = threshold or float(os.environ.get('OPTIMAL_THRESHOLD', 0.45))
    cfg.OPTIMAL_MIN_LESION_SIZE = min_lesion_size or int(os.environ.get('OPTIMAL_MIN_LESION_SIZE', 15))
    cfg.data_root = data_root or os.environ.get('DATA_ROOT', default_data)
    cfg.model_path = model_path or os.environ.get('MODEL_PATH', os.path.join(default_work, 'models', 'best_model.pth'))
    cfg.output_dir = output_dir or os.environ.get('OUTPUT_DIR', os.path.join(default_work, "final_test_segmentations"))
    cfg.chs = (64, 96, 128, 192, 256); cfg.token_dim = 256; cfg.swin_mlp_ratio = 2.0
    cfg.dropout_rate = 0.5; cfg.drop_path_rate = 0.3
    cfg.roi_size = roi_size or tuple(map(int, os.environ.get('ROI_SIZE', '208,208').split(',')))
    cfg.device = device or os.environ.get('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(cfg.output_dir, exist_ok=True)
    
    print(f"Loading model from: {cfg.model_path}")
    model = SYNAPSENet(
        chs=cfg.chs, token_dim=cfg.token_dim, swin_mlp_ratio=cfg.swin_mlp_ratio,
        dropout_rate=cfg.dropout_rate, drop_path_rate=cfg.drop_path_rate
    ).to(cfg.device)
    model.load_state_dict(torch.load(cfg.model_path, map_location=cfg.device))
    model.eval()

    all_subject_dirs = sorted([d for d in os.listdir(cfg.data_root) if d.startswith('sub-')])
    test_subjects = all_subject_dirs[150:]
    print(f"Found {len(test_subjects)} subjects for final testing.")

    with torch.no_grad():
        for sub_dir in tqdm(test_subjects, desc="Processing Final Test Set"):
            try:
                dwi_path_list = glob.glob(os.path.join(cfg.data_root, sub_dir, 'ses-0001/dwi/*_dwi.nii', '*.nii'))
                adc_path_list = glob.glob(os.path.join(cfg.data_root, sub_dir, 'ses-0001/dwi/*_adc.nii', '*.nii'))

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

                pred_binary = (pred_prob_vol > cfg.OPTIMAL_THRESHOLD).astype(np.uint8)
                pred_processed = post_process_volume(pred_binary, min_lesion_size=cfg.OPTIMAL_MIN_LESION_SIZE)

                output_filename = os.path.join(cfg.output_dir, f"{sub_dir}_pred.nii")
                final_nii = nib.Nifti1Image(pred_processed, original_nii.affine, original_nii.header)
                nib.save(final_nii, output_filename)
                
            except Exception as e:
                print(f"Skipping subject {sub_dir} during final inference due to an error: {e}")

    print(f"\nFinal segmentations saved to: {cfg.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run final ISLES test inference')
    parser.add_argument('--data_root', type=str, help='ISLES dataset root')
    parser.add_argument('--model_path', type=str, help='Trained model path')
    parser.add_argument('--output_dir', type=str, help='Output directory for segmentations')
    parser.add_argument('--threshold', type=float, help='Threshold for binarization')
    parser.add_argument('--min_lesion_size', type=int, help='Min lesion size for post-processing')
    parser.add_argument('--roi_size', type=str, help='ROI size e.g. 208,208')
    parser.add_argument('--device', type=str, help='Device to run on')
    args = parser.parse_args()

    run_final_test_inference(data_root=args.data_root, model_path=args.model_path, output_dir=args.output_dir, threshold=args.threshold, min_lesion_size=args.min_lesion_size, roi_size=args.roi_size, device=args.device)
