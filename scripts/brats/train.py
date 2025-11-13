import os, sys, random, time
import numpy as np
from datetime import datetime
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import nibabel as nib
import pandas as pd

from src.models.SYNAPSE-Net_4mod import SYNAPSE-Net_4mod as SYNAPSENet
from src.data_loaders.dataset_brats import BraTS_dataset, RandomGenerator
from src.utils.helpers.brats_helpers import (
    post_process_volume,
    FocalLoss,
    DiceLoss,
    downsample_mask,
    compute_brats_metrics_detailed
)
class Config:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    default_data = os.path.join(repo_root, 'data', 'brats')
    default_work = os.path.join(repo_root, 'work')

    data_root = os.getenv('DATA_ROOT', default_data)
    work_dir = os.getenv('WORK_DIR', f'brats4_diceloss_final_stable_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    resume_from_checkpoint = None
    
    num_subjects = 50
    val_split_fraction = 0.2
    input_h, input_w = 208, 208
    batch_size = 8
    num_workers = 2
    oversample_factor = 2
    seed = 42
    
    num_classes = 4
    chs=(64, 96, 128, 256, 384)
    token_dim=256
    bottleneck_heads=8
    bottleneck_window=8
    pre_swin_layers=2
    dropout_rate = 0.4
    
    epochs = 300
    amp = True
    lr = 5e-5 
    weight_decay = 1e-4
    early_stop = 30
    
    aux_weights = [0.4, 0.2]
    lesion_weight = 0.25
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def prepare_dataloaders(cfg):
    train_transform = RandomGenerator(output_size=(cfg.input_h, cfg.input_w))
    train_ds = BraTS_dataset(
        base_dir=cfg.data_root, split='train', transform=train_transform,
        oversample_factor=cfg.oversample_factor, num_subjects=cfg.num_subjects,
        val_split_fraction=cfg.val_split_fraction, seed=cfg.seed
    )
    val_ds = BraTS_dataset(
        base_dir=cfg.data_root, split='val', transform=None,
        num_subjects=cfg.num_subjects, val_split_fraction=cfg.val_split_fraction, seed=cfg.seed
    )
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())
    return train_loader, val_ds

def prepare_model(cfg):
    model = SYNAPSENet(
        chs=cfg.chs, token_dim=cfg.token_dim, bottleneck_heads=cfg.bottleneck_heads,
        aux_outs=len(cfg.aux_weights), bottleneck_window=cfg.bottleneck_window,
        pre_swin_layers=cfg.pre_swin_layers, dropout_rate=cfg.dropout_rate,
        num_modalities=4, num_classes=cfg.num_classes
    ).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-7)
    return model, optimizer, scheduler

def train_epoch(epoch, model, loader, optimizer, scaler, cfg):
    model.train()
    running_loss = 0.0
    valid_steps = 0
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Train E{epoch}", leave=False)
    seg_loss_obj = DiceLoss(squared_pred=True, smooth_dr=1e-5, smooth_nr=0.0)
    lesion_loss_obj = FocalLoss()
    
    for i, batch in pbar:
        imgs, tgts = batch['image'].to(cfg.device), batch['label'].to(cfg.device)
        optimizer.zero_grad(set_to_none=True)
        
        with autocast(device_type=cfg.device, dtype=torch.float16, enabled=cfg.amp):
            main_logits, auxs, lesion_logits = model(imgs)
            loss_main = seg_loss_obj(main_logits, tgts)
            loss_aux = sum(w * seg_loss_obj(a, tgts) for w, a in zip(cfg.aux_weights, auxs))
            lesion_target = (tgts > 0).float()
            lesion_target_ds = downsample_mask(lesion_target.unsqueeze(1), lesion_logits)
            loss_lesion = lesion_loss_obj(lesion_logits, lesion_target_ds)
            loss = loss_main + loss_aux + cfg.lesion_weight * loss_lesion
        
        if torch.isfinite(loss):
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item())
            valid_steps += 1
        else:
            print(f"\nWARNING: Skipping step {i} due to non-finite loss.")

        if valid_steps > 0:
             pbar.set_postfix({'loss': f'{running_loss / valid_steps:.4f}'})
        
    return running_loss / valid_steps if valid_steps > 0 else 0.0

def validate(model, val_ds, cfg):
    model.eval()
    all_metrics_list = []
    case_losses = []
    seg_loss_obj = DiceLoss(squared_pred=True, smooth_dr=1e-5, smooth_nr=0.0)
    with torch.no_grad():
        for i in range(len(val_ds)):
            vols = val_ds._load_volume(i); mask_vol = vols['mask']
            mask_vol[mask_vol == 4] = 3
            pred_vol = np.zeros_like(mask_vol, dtype=np.uint8)
            vol_slice_losses = []
            
            for s in range(mask_vol.shape[2]):
                img_slice = np.stack([vols['t1c'][:,:,s], vols['t1n'][:,:,s], vols['t2f'][:,:,s], vols['t2w'][:,:,s]], axis=-1)

                img_tensor = torch.from_numpy(img_slice).permute(2,0,1).float().unsqueeze(0).to(cfg.device)
                mask_tensor = torch.from_numpy(mask_vol[:,:,s].copy()).long().unsqueeze(0).to(cfg.device)
                
                with autocast(device_type=cfg.device, dtype=torch.float16, enabled=cfg.amp):
                    resized_tensor = F.interpolate(img_tensor, size=(cfg.input_h, cfg.input_w), mode='bilinear', align_corners=False)
                    main_logits, _, _ = model(resized_tensor)
                    resized_logits = F.interpolate(main_logits, size=img_tensor.shape[-2:], mode='bilinear', align_corners=False)
                    val_loss = seg_loss_obj(resized_logits, mask_tensor)
                
                if torch.isfinite(val_loss):
                    vol_slice_losses.append(float(val_loss.item()))
                
                preds = torch.argmax(torch.softmax(resized_logits, dim=1), dim=1)
                pred_vol[:,:,s] = preds.squeeze().cpu().numpy()
            
            if vol_slice_losses:
                case_losses.append(np.mean(vol_slice_losses))

            pred_vol_pp = pred_vol.copy()
            for cls in [1, 2, 3]: pred_vol_pp = post_process_volume(pred_vol_pp, class_idx=cls, min_size=50)
            all_metrics_list.append(compute_brats_metrics_detailed(pred_vol_pp, mask_vol))
    
    metrics_df = pd.DataFrame(all_metrics_list)
    mean_metrics = metrics_df.mean().to_dict()
    mean_metrics['Val_Loss'] = np.mean(case_losses) if case_losses else 0.0
    return mean_metrics

def main():
    cfg = Config()
    os.makedirs(cfg.work_dir, exist_ok=True)
    set_seed(cfg.seed)
    log_file_path = os.path.join(cfg.work_dir, "training_log.txt")
    
    print("Preparing dataloaders..."); train_loader, val_ds = prepare_dataloaders(cfg)
    print("Preparing model..."); model, optimizer, scheduler = prepare_model(cfg)
    
    scaler = GradScaler(enabled=cfg.amp)
    best_val_dice_mean = -1.0
    start_epoch = 1
    epochs_no_improve = 0

    with open(log_file_path, 'a') as log_file:
        start_msg = f"\nStarting new training for {cfg.epochs} epochs on {cfg.device}"
        print(start_msg); log_file.write(start_msg + '\n')
        
        header = (f"{'Epoch':<5} | {'Train Loss':<10} | {'Val Loss':<10} | "
                  f"{'D_WT':<7} | {'H_WT':<7} | {'D_TC':<7} | {'H_TC':<7} | "
                  f"{'D_ET':<7} | {'H_ET':<7} | {'Mean Dice':<10}")
        separator = "-" * len(header)
        print(separator); log_file.write(separator + '\n'); print(header); log_file.write(header + '\n'); print(separator); log_file.write(header + '\n')
        log_file.flush()

        for epoch in range(start_epoch, cfg.epochs + 1):
            print(f"--- Epoch {epoch}/{cfg.epochs}, LR: {optimizer.param_groups[0]['lr']:.2e} ---")
            train_loss = train_epoch(epoch, model, train_loader, optimizer, scaler, cfg)
            val_metrics = validate(model, val_ds, cfg)
            
            val_dice_mean = (val_metrics['Dice_WT'] + val_metrics['Dice_TC'] + val_metrics['Dice_ET']) / 3
            
            scheduler.step()

            log_str = (f"{epoch:<5} | {train_loss:<10.4f} | {val_metrics['Val_Loss']:<10.4f} | "
                       f"{val_metrics['Dice_WT']:<7.4f} | {val_metrics['HD95_WT']:<7.2f} | "
                       f"{val_metrics['Dice_TC']:<7.4f} | {val_metrics['HD95_TC']:<7.2f} | "
                       f"{val_metrics['Dice_ET']:<7.4f} | {val_metrics['HD95_ET']:<7.2f} | "
                       f"{val_dice_mean:<10.4f}")
            print(log_str); log_file.write(log_str + '\n')

            if val_dice_mean > best_val_dice_mean:
                best_val_dice_mean = val_dice_mean
                epochs_no_improve = 0
                torch.save(model.state_dict(), os.path.join(cfg.work_dir, 'best_model.pth'))
                save_msg = f"  -> New best model saved with Mean Dice: {best_val_dice_mean:.4f}"
                print(save_msg); log_file.write(save_msg + '\n')
            else:
                epochs_no_improve += 1

            log_file.flush()
            
            if epochs_no_improve >= cfg.early_stop:
                stop_msg = f"\nEarly stopping triggered after {cfg.early_stop} epochs with no improvement."
                print(stop_msg)
                log_file.write(stop_msg + '\n')
                break

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, help='Path to BraTS data root')
    parser.add_argument('--work_dir', type=str, help='Working directory to save logs and models')
    parser.add_argument('--resume', type=str, help='Checkpoint path to resume from')
    parser.add_argument('--num_subjects', type=int, help='Override number of subjects to use')
    args = parser.parse_args()
    if args.data_root:
        Config.data_root = args.data_root
    if args.work_dir:
        Config.work_dir = args.work_dir
    if args.resume:
        Config.resume_from_checkpoint = args.resume
    if args.num_subjects:
        Config.num_subjects = args.num_subjects
    main()
