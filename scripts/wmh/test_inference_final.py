import os
import glob
import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm
import torch.nn.functional as F
from monai.inferers import sliding_window_inference

from src.models.SYNAPSENet_2mod import SYNAPSENet_2mod as SYNAPSENet
from src.utils.utils import post_process_volume

# --- SUMMARY OF CHANGES ---
# REMOVED: The incorrect slice-level min-max normalization loop. This script
# now correctly processes the Z-scored data, ensuring consistency with training.


def run_final_inference():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    default_work = os.path.join(repo_root, 'work')
    default_data = os.path.join(repo_root, 'data', 'wmh_split_data')

    class Config:
        # Paths (env overrides or repo-relative defaults)
        test_data_root = os.getenv('TEST_DATA_ROOT', os.path.join(os.getenv('DATA_ROOT', default_data), 'test'))
        base_work_dir = os.getenv('WORK_DIR', os.getenv('BASE_WORK_DIR', default_work))
        model_path = os.getenv('MODEL_PATH', os.path.join(base_work_dir, 'models', 'best_model.pth'))
        output_dir = os.getenv('OUTPUT_DIR', os.path.join(base_work_dir, 'predictions'))

        # Post-processing / inference params
        OPTIMAL_THRESHOLD = float(os.getenv('OPTIMAL_THRESHOLD', 0.25))
        OPTIMAL_MIN_LESION_SIZE = int(os.getenv('OPTIMAL_MIN_LESION_SIZE', 5))
        roi_size = tuple(map(int, os.getenv('ROI_SIZE', '208,208').split(',')))
        sw_batch_size = int(os.getenv('SW_BATCH_SIZE', 4))
        overlap = float(os.getenv('OVERLAP', 0.5))

        # Model architecture hyperparams (defaults mirror training)
        chs = (64, 96, 128, 192, 256)
        token_dim = 256
        swin_mlp_ratio = 2.0
        dropout_rate = 0.4
        drop_path_rate = 0.25

        device = os.getenv('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')

    cfg = Config()

    os.makedirs(cfg.output_dir, exist_ok=True)

    print(f"Loading model from: {cfg.model_path}")
    model = SYNAPSENet(chs=cfg.chs, token_dim=cfg.token_dim, swin_mlp_ratio=cfg.swin_mlp_ratio,
                         dropout_rate=cfg.dropout_rate, drop_path_rate=cfg.drop_path_rate).to(cfg.device)

    model.load_state_dict(torch.load(cfg.model_path, map_location=cfg.device))

    model.eval()

    flair_dir = os.path.join(cfg.test_data_root, 'flair')
    t1_dir = os.path.join(cfg.test_data_root, 't1')
    test_files = sorted(glob.glob(os.path.join(flair_dir, "*.nii*")))
    print(f"Found {len(test_files)} subjects in the test directory.")

    with torch.no_grad():
        for flair_path in tqdm(test_files, desc="Processing Test Set"):
            base_name = os.path.basename(flair_path).replace('_flair.nii.gz', '').replace('_flair.nii', '')
            t1_path = os.path.join(t1_dir, f"{base_name}_t1.nii.gz")
            if not os.path.exists(t1_path): t1_path = t1_path[:-3]
            if not os.path.exists(t1_path): continue

            original_nii = nib.load(flair_path)
            flair_vol = original_nii.get_fdata(dtype=np.float32)
            t1_vol = nib.load(t1_path).get_fdata(dtype=np.float32)

            input_vol_np = np.stack([flair_vol, t1_vol], axis=0).astype(np.float32)
            input_vol_np = np.transpose(input_vol_np, (0, 3, 1, 2))

            input_tensor = torch.from_numpy(input_vol_np).unsqueeze(0).to(cfg.device)

            def predictor_2d(patch_batch):
                patch_2d = patch_batch.squeeze(2)
                logits_2d, _, _ = model(patch_2d)
                return logits_2d.unsqueeze(2)

            pred_logits = sliding_window_inference(
                inputs=input_tensor,
                roi_size=(1, *cfg.roi_size),
                sw_batch_size=cfg.sw_batch_size,
                predictor=predictor_2d,
                overlap=cfg.overlap,
                mode="gaussian"
            )

            pred_probs_tensor = torch.sigmoid(pred_logits)
            pred_prob_vol = pred_probs_tensor.squeeze(0).squeeze(0).cpu().numpy().transpose(1, 2, 0)

            pred_binary = (pred_prob_vol > cfg.OPTIMAL_THRESHOLD).astype(np.uint8)
            pred_processed = post_process_volume(pred_binary, min_lesion_size=cfg.OPTIMAL_MIN_LESION_SIZE)

            output_filename = os.path.join(cfg.output_dir, f"{base_name}_wmh_pred.nii.gz")
            final_nii = nib.Nifti1Image(pred_processed.astype(np.uint8), original_nii.affine, original_nii.header)
            nib.save(final_nii, output_filename)

    print(f"\nFinal segmentations saved to: {cfg.output_dir}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Model checkpoint path')
    parser.add_argument('--data_root', help='Base data root')
    parser.add_argument('--test_data_root', help='Test data root')
    parser.add_argument('--output_dir', help='Output directory for predictions')
    parser.add_argument('--device', help='Device string')
    args = parser.parse_args()
    if args.model: os.environ['MODEL_PATH'] = args.model
    if args.data_root: os.environ['DATA_ROOT'] = args.data_root
    if args.test_data_root: os.environ['TEST_DATA_ROOT'] = args.test_data_root
    if args.output_dir: os.environ['OUTPUT_DIR'] = args.output_dir
    if args.device: os.environ['DEVICE'] = args.device
    run_final_inference()
