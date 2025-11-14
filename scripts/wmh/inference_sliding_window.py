import os
import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm
from monai.inferers import sliding_window_inference
import torch.nn.functional as F

from src.models.SYNAPSENet_2mod import SYNAPSENet_2mod as SYNAPSENet
from src.data_loaders.dataset_wmh import WMH_dataset


def generate_validation_probs():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    default_work = os.path.join(repo_root, 'work')
    default_data = os.path.join(repo_root, 'data', 'wmh_split_data')

    class Config:
        base_work_dir = os.getenv('WORK_DIR', os.getenv('BASE_WORK_DIR', default_work))
        validation_data_root = os.getenv('VALIDATION_DATA_ROOT', os.path.join(os.getenv('DATA_ROOT', default_data), 'train', 'validation'))
        model_path = os.getenv('MODEL_PATH', os.path.join(base_work_dir, 'models', 'best_model.pth'))
        output_dir = os.getenv('OUTPUT_DIR', os.path.join(base_work_dir, "validation_probabilities"))
        chs = (64, 96, 128, 192, 256); token_dim = 256; swin_mlp_ratio = 2.0
        dropout_rate = 0.4; drop_path_rate = 0.25
        roi_size = (208, 208); sw_batch_size = 4; overlap = 0.5
        device = os.getenv('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')

    cfg = Config()
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    print(f"Loading model from: {cfg.model_path}")
    model = SYNAPSENet(chs=cfg.chs, token_dim=cfg.token_dim, swin_mlp_ratio=cfg.swin_mlp_ratio,
                         dropout_rate=cfg.dropout_rate, drop_path_rate=cfg.drop_path_rate).to(cfg.device)

    model.load_state_dict(torch.load(cfg.model_path, map_location=cfg.device))
    
    model.eval()

    val_ds = WMH_dataset(base_dir=cfg.validation_data_root, split='val')
    print(f"Found {len(val_ds)} validation subjects in: {cfg.validation_data_root}")

    with torch.no_grad():
        for i in tqdm(range(len(val_ds)), desc="Generating Validation Probabilities"):
            trip = val_ds.file_triplets[i]
            original_nii = nib.load(trip['flair'])
            vols = val_ds._load_volume(i)
            flair_vol, t1_vol = vols['flair'], vols['t1']
            
            input_vol_np = np.stack([flair_vol, t1_vol], axis=0).astype(np.float32)
            input_vol_np = np.transpose(input_vol_np, (0, 3, 1, 2))
            
            input_tensor = torch.from_numpy(input_vol_np).unsqueeze(0).to(cfg.device)
            
            def predictor_2d(volume_patch_batch):
                patch_2d = volume_patch_batch.squeeze(2)
                logits_2d, _, _ = model(patch_2d)
                return logits_2d.unsqueeze(2)

            pred_logits = sliding_window_inference(
                inputs=input_tensor,
                roi_size=(1, cfg.roi_size[0], cfg.roi_size[1]),
                sw_batch_size=cfg.sw_batch_size,
                predictor=predictor_2d,
                overlap=cfg.overlap,
                mode="gaussian"
            )
            
            pred_probs_tensor = torch.sigmoid(pred_logits)
            pred_prob_vol = pred_probs_tensor.squeeze(0).squeeze(0).cpu().numpy().transpose(1, 2, 0)

            output_filename = os.path.join(cfg.output_dir, f"{trip['name']}_prob.nii.gz")
            prob_nii = nib.Nifti1Image(pred_prob_vol, original_nii.affine, original_nii.header)
            nib.save(prob_nii, output_filename)
            
    print(f"\nInference complete. High-quality probability maps saved to: {cfg.output_dir}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Path to model checkpoint')
    parser.add_argument('--data_root', help='Validation data root')
    parser.add_argument('--output_dir', help='Directory to save probability maps')
    parser.add_argument('--device', help='device string (cpu or cuda)')
    args = parser.parse_args()
    if args.model: os.environ['MODEL_PATH'] = args.model
    if args.data_root: os.environ['VALIDATION_DATA_ROOT'] = args.data_root
    if args.output_dir: os.environ['OUTPUT_DIR'] = args.output_dir
    if args.device: os.environ['DEVICE'] = args.device
    generate_validation_probs()
