import os
import sys
import random
import time
import numpy as np
from datetime import datetime
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import nibabel as nib

from src.models.lacmafnet_2mod import LACMAFNet_2mod as LACMAFNet
from src.data_loaders.dataset_wmh import WMH_dataset, RandomGeneratorFromScratch
from src.utils.helpers.wmh_helpers import (
    FocalTverskyLoss, FocalLoss, BoundaryLoss,
)
from src.utils.utils import (
    compute_case_metrics, 
    downsample_mask,
    counts_from_logits
)

class Config:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    default_data = os.path.join(repo_root, 'data', 'wmh_split_data')
    default_work = os.path.join(repo_root, 'work')

    data_root = os.getenv('DATA_ROOT', default_data)
    
    work_dir = os.getenv('WORK_DIR', f'lacmafnet_WMH_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    resume_from_checkpoint = None
    input_h, input_w = 208, 208
    amp = True
    seed = 42
    
    batch_size = 18
    num_workers = 2
    epochs = 150
    early_stop = 20
    
    lr = 1e-4
    warmup_epochs = 15
    weight_decay = 1.5e-4

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print_interval = 20
    k_folds = 5
    
    chs = (64, 96, 128, 192, 256)
    token_dim = 256
    swin_mlp_ratio = 2.0
    dropout_rate = 0.4
    drop_path_rate = 0.3
    
    use_tta = True
    oversample_factor = 5     
    difficulty_percentile = 30 
    difficulty_bonus = 4      
    
    aux_weights = [0.4, 0.2]
    lesion_weight = 0.5
    boundary_loss_weight = 1.0 
    
    ft_loss_wf = 0.3           
    ft_loss_wt = 0.7           
    focal_alpha = 0.25
    focal_gamma = 2.0
    tversky_beta = 0.7         

def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def prepare_dataloaders(cfg):
    # Note: For k-fold CV we will construct datasets outside this function.
    train_transform = RandomGeneratorFromScratch(output_size=(cfg.input_h, cfg.input_w))
    from torch.utils.data import DataLoader
    # keep signature simple for the non-folded case: caller should pass dataset objects
    def make_loader(dataset, shuffle=True):
        return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=shuffle,
                          num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())

    return train_transform, make_loader

def prepare_model(cfg):
    model = LACMAFNet(chs=cfg.chs, token_dim=cfg.token_dim, swin_mlp_ratio=cfg.swin_mlp_ratio,
                         dropout_rate=cfg.dropout_rate, drop_path_rate=cfg.drop_path_rate).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(cfg.epochs - cfg.warmup_epochs), eta_min=1e-7)
    return model, optimizer, scheduler

def train_epoch(epoch, model, loader, optimizer, scaler, cfg):
    model.train()
    running_loss = 0.0
    total_TP, total_FP, total_FN, _ = 0.0, 0.0, 0.0, 0.0
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Train {epoch}")
    
    seg_loss_obj = FocalTverskyLoss(wf=cfg.ft_loss_wf, wt=cfg.ft_loss_wt, alpha=cfg.focal_alpha, gamma=cfg.focal_gamma, tversky_beta=cfg.tversky_beta)
    boundary_loss_obj = BoundaryLoss()
    lesion_loss_obj = FocalLoss(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma)
    
    for i, batch in pbar:
        imgs, tgts = batch['image'].to(cfg.device), batch['label'].to(cfg.device)
        optimizer.zero_grad(set_to_none=True)
        
        with autocast(device_type=cfg.device, dtype=torch.float16, enabled=cfg.amp):
            main_logits, auxs, lesion_logits = model(imgs)
            
            loss_main_seg = seg_loss_obj(main_logits, tgts)
            loss_main_boundary = boundary_loss_obj(torch.sigmoid(main_logits), tgts)
            loss_main = loss_main_seg + (cfg.boundary_loss_weight * loss_main_boundary)
            
            loss_aux = sum(w * seg_loss_obj(a, tgts) for w, a in zip(cfg.aux_weights, auxs))
            
            lesion_target = downsample_mask(tgts, lesion_logits)
            loss_lesion = lesion_loss_obj(lesion_logits, lesion_target)
            
            loss = loss_main + loss_aux + (cfg.lesion_weight * loss_lesion)
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += float(loss.item())
        tp, fp, fn, _ = counts_from_logits(main_logits.detach(), tgts.detach(), thr=0.5)
        total_TP += tp
        total_FP += fp
        total_FN += fn
        
        if (i + 1) % cfg.print_interval == 0 or (i + 1) == len(loader):
            dice_so_far = (2 * total_TP) / (2 * total_TP + total_FP + total_FN + 1e-8)
            pbar.set_postfix({'a_loss': f'{running_loss / (i + 1):.4f}', 'g_dice': f'{dice_so_far:.4f}', 'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'})
            
    epoch_loss = running_loss / len(loader)
    epoch_dice = (2 * total_TP) / (2 * total_TP + total_FP + total_FN + 1e-8)
    return epoch_loss, float(epoch_dice)

def validate(model, val_ds, cfg):
    model.eval()
    all_metrics = []
    case_losses = []
    seg_loss_obj = FocalTverskyLoss(wf=cfg.ft_loss_wf, wt=cfg.ft_loss_wt, alpha=cfg.focal_alpha, gamma=cfg.focal_gamma, tversky_beta=cfg.tversky_beta)
    
    with torch.no_grad():
        for i in tqdm(range(len(val_ds)), desc="Validating", leave=False):
            trip = val_ds.file_triplets[i]
            flair_vol = nib.load(trip['flair']).get_fdata(dtype=np.float32)
            t1_vol = nib.load(trip['t1']).get_fdata(dtype=np.float32)
            mask_vol = nib.load(trip['mask']).get_fdata(dtype=np.float32)
            
            pred_vol = np.zeros_like(mask_vol, dtype=np.float32)
            vol_slice_losses = []
            
            for s in range(mask_vol.shape[2]):
                img_slice = np.stack([flair_vol[:,:,s], t1_vol[:,:,s]], axis=-1)
                
                img_tensor = torch.from_numpy(img_slice.copy()).permute(2,0,1).float().unsqueeze(0).to(cfg.device)
                mask_tensor = torch.from_numpy(mask_vol[:,:,s].copy()).float().unsqueeze(0).unsqueeze(0).to(cfg.device)
                resized_tensor = F.interpolate(img_tensor, size=(cfg.input_h, cfg.input_w), mode='bilinear', align_corners=False)
                
                if cfg.use_tta:
                    logits_original, _, _ = model(resized_tensor)
                    logits_flipped, _, _ = model(torch.flip(resized_tensor, dims=[3]))
                    main_logits = (logits_original + torch.flip(logits_flipped, dims=[3])) / 2.0
                else:
                    main_logits, _, _ = model(resized_tensor)
                
                resized_logits_for_loss = F.interpolate(main_logits, size=img_tensor.shape[-2:], mode='bilinear', align_corners=False)
                slice_loss = seg_loss_obj(resized_logits_for_loss, mask_tensor)
                vol_slice_losses.append(float(slice_loss.item()))

                resized_logits_for_pred = F.interpolate(main_logits, size=img_tensor.shape[-2:], mode='bilinear', align_corners=False)
                probs = torch.sigmoid(resized_logits_for_pred)
                pred_vol[:,:,s] = probs.squeeze().cpu().numpy()

            case_losses.append(float(np.mean(vol_slice_losses)))
            all_metrics.append(compute_case_metrics(pred_vol, mask_vol))
            
    dices, hd95s, recalls, accs, avds, f1s = zip(*all_metrics)
    return (np.mean(dices), np.std(dices), np.mean(hd95s), np.std(hd95s), np.mean(recalls), np.mean(accs), np.mean(avds), np.mean(f1s), np.mean(case_losses))

def main():
    cfg = Config()
    os.makedirs(cfg.work_dir, exist_ok=True)
    set_seed(cfg.seed)
    
    log_file_path = os.path.join(cfg.work_dir, "training_log.txt")
    train_transform, make_loader = prepare_dataloaders(cfg)
    scaler = GradScaler(enabled=cfg.amp)
    # For k-fold cross-validation we will split the available training subjects into folds
    # Load the full WMH_dataset to get file_triplets and slice_map logic
    full_train_ds = WMH_dataset(base_dir=cfg.data_root, split='train', transform=None,
                                oversample_factor=cfg.oversample_factor,
                                difficulty_percentile=cfg.difficulty_percentile,
                                difficulty_bonus=cfg.difficulty_bonus)

    # Build list of subject indices
    subject_count = len(full_train_ds.file_triplets)
    indices = list(range(subject_count))
    random.shuffle(indices)

    k = cfg.k_folds if getattr(cfg, 'k_folds', 1) and cfg.k_folds > 1 else 1
    fold_size = subject_count // k

    # helper to create subset dataset (keeps WMH_dataset behaviour but restricts file_triplets)
    class WMHSubset(WMH_dataset):
        def __init__(self, base_dir, split, transform, subset_subject_idxs, parent_triplets, **kwargs):
            # ...existing code... reuse parent lists
            self.split = split
            self.transform = transform
            self.root = base_dir
            # shallow copy of parent triplets filtered by indices
            self.file_triplets = [parent_triplets[i] for i in subset_subject_idxs]
            # build slice_map similar to WMH_dataset for train split
            self.slice_map = []
            if self.split == 'train':
                for i, trip in enumerate(self.file_triplets):
                    try:
                        mask_vol = nib.load(trip['mask']).get_fdata()
                        for s in range(mask_vol.shape[2]):
                            lesion_sum = mask_vol[:, :, s].sum()
                            has_lesion = lesion_sum > 0
                            self.slice_map.append({'subject_idx': i, 'slice_idx': s, 'has_lesion': has_lesion})
                            if has_lesion:
                                # oversample using same oversample_factor as parent
                                for _ in range(cfg.oversample_factor - 1):
                                    self.slice_map.append({'subject_idx': i, 'slice_idx': s, 'has_lesion': True})
                    except Exception:
                        continue

    # Outer loop over folds
    fold_metrics = []
    for fold in range(k):
        print(f"Starting fold {fold+1}/{k}")
        # compute validation fold indices
        start = fold * fold_size
        end = start + fold_size if fold < k - 1 else subject_count
        val_subjects = indices[start:end]
        train_subjects = [i for i in indices if i not in val_subjects]

        # Create subset datasets
        train_ds = WMHSubset(base_dir=cfg.data_root, split='train', transform=train_transform,
                              subset_subject_idxs=train_subjects, parent_triplets=full_train_ds.file_triplets)
        val_ds = WMHSubset(base_dir=cfg.data_root, split='val', transform=None,
                            subset_subject_idxs=val_subjects, parent_triplets=full_train_ds.file_triplets)

        train_loader = make_loader(train_ds, shuffle=True)

        # prepare fresh model, optimizer and scheduler per-fold
        model, optimizer, scheduler = prepare_model(cfg)

        start_epoch = 1
        best_val_dice = -1.0
        best_val_loss = float('inf')
        epochs_no_improve = 0
        base_lr = cfg.lr

        # open per-fold log
        fold_log_path = os.path.join(cfg.work_dir, f"training_log_fold_{fold+1}.txt")
        with open(fold_log_path, 'a') as log_file:
            if start_epoch == 1:
                start_msg = f"\nStarting fold {fold+1}/{k} training for {cfg.epochs} epochs on {cfg.device}"
                print(start_msg)
                log_file.write(start_msg + '\n')
                separator = "-" * 150
                print(separator)
                log_file.write(separator + '\n')
                header = (f"{ 'Epoch':<7} | {'Train Loss':<12} | {'Train Dice':<12} | {'Val Loss':<10} | {'Val Dice (STD)':<18} | {'Val HD95 (STD)':<18} | {'Val Recall':<12} | {'Val F1':<10}")
                print(header)
                log_file.write(header + '\n')
                print(separator)
                log_file.write(separator + '\n')
                log_file.flush()

            for epoch in range(start_epoch, cfg.epochs + 1):
                if epoch <= cfg.warmup_epochs:
                    current_lr = base_lr * (epoch / cfg.warmup_epochs)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = current_lr

                train_loss, train_dice = train_epoch(epoch, model, train_loader, optimizer, scaler, cfg)
                val_dice, val_dice_std, val_hd95, val_hd95_std, val_recall, val_acc, val_avd, val_f1, val_loss = validate(model, val_ds, cfg)

                if epoch > cfg.warmup_epochs:
                    scheduler.step()

                log_str = (f"{epoch:<7} | {train_loss:<12.4f} | {train_dice:<12.4f} | {val_loss:<10.4f} | "
                           f"{val_dice:<7.4f} ({val_dice_std:<7.4f}) | "
                           f"{val_hd95:<7.2f} ({val_hd95_std:<7.2f}) | "
                           f"{val_recall:<12.4f} | {val_f1:<10.4f}")
                print(log_str)
                log_file.write(log_str + '\n')

                if val_dice > best_val_dice:
                    best_val_dice = val_dice
                    torch.save(model.state_dict(), os.path.join(cfg.work_dir, f'best_model_fold_{fold+1}.pth'))
                    save_msg = f"  -> New best model saved with Val Dice: {best_val_dice:.4f}"
                    print(save_msg)
                    log_file.write(save_msg + '\n')

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                checkpoint_path = os.path.join(cfg.work_dir, f'last_checkpoint_fold_{fold+1}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_val_dice': best_val_dice,
                    'best_val_loss': best_val_loss,
                    'epochs_no_improve': epochs_no_improve,
                }, checkpoint_path)
                log_file.flush()

                if epochs_no_improve >= cfg.early_stop:
                    stop_msg = f"\nEarly stopping triggered after {cfg.early_stop} epochs based on validation loss for fold {fold+1}."
                    print(stop_msg)
                    log_file.write(stop_msg + '\n')
                    break

            final_msg = f"\nFold {fold+1} finished. Best validation Dice: {best_val_dice:.4f}"
            print(final_msg)
            log_file.write(final_msg + '\n')
            fold_metrics.append({'fold': fold+1, 'best_val_dice': best_val_dice, 'best_val_loss': best_val_loss})

    # after all folds
    print("\nCross-validation finished. Summary per-fold:")
    for m in fold_metrics:
        print(f"Fold {m['fold']}: Best Dice={m['best_val_dice']:.4f}, Best Loss={m['best_val_loss']:.4f}")
        log_file.write(final_msg + '\n')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, help='Path to WMH data root')
    parser.add_argument('--work_dir', type=str, help='Working directory to save logs and models')
    parser.add_argument('--resume', type=str, help='Checkpoint path to resume from')
    parser.add_argument('--k_folds', type=int, default=None, help='Number of folds for cross-validation (overrides Config.k_folds)')
    args = parser.parse_args()
    # Override config defaults if provided
    if args.data_root:
        Config.data_root = args.data_root
    if args.work_dir:
        Config.work_dir = args.work_dir
    if args.resume:
        Config.resume_from_checkpoint = args.resume
    if args.k_folds:
        Config.k_folds = args.k_folds
    main()
