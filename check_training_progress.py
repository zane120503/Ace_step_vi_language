#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de kiem tra training progress va so steps da chay
"""

import os
import glob
from pathlib import Path
from datetime import datetime

def check_training_progress():
    log_base = Path("./exps/logs/vi_lora/lightning_logs")
    
    if not log_base.exists():
        print("[ERROR] Khong tim thay folder logs!")
        return
    
    # Tim folder training moi nhat
    folders = sorted(log_base.glob("*vi_lora_small"), key=os.path.getmtime, reverse=True)
    
    if not folders:
        print("[ERROR] Khong tim thay folder training nao!")
        return
    
    latest_folder = folders[0]
    mtime = datetime.fromtimestamp(os.path.getmtime(latest_folder))
    print(f"[OK] Folder training moi nhat: {latest_folder.name}")
    print(f"     Thoi gian cap nhat: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Kiem tra events file
    events_files = list(latest_folder.glob("events*"))
    if events_files:
        events_file = events_files[0]
        size = events_file.stat().st_size
        etime = datetime.fromtimestamp(events_file.stat().st_mtime)
        print(f"[INFO] Events file: {events_file.name}")
        print(f"       Kich thuoc: {size:,} bytes")
        print(f"       Thoi gian: {etime.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # File nho co nghia la training moi bat dau hoac da dung som
        if size < 5000:
            print(f"       [WARNING] File qua nho - training co the moi bat dau hoac chua co nhieu log")
        else:
            print(f"       [OK] File co du lieu - training da chay duoc mot luc")
    else:
        print("[ERROR] Khong tim thay events file!")
    
    print()
    
    # Kiem tra checkpoints
    checkpoints_dir = latest_folder / "checkpoints"
    if checkpoints_dir.exists():
        checkpoints = list(checkpoints_dir.glob("*"))
        if checkpoints:
            print(f"[OK] Tim thay {len(checkpoints)} checkpoint(s):")
            for ckpt in sorted(checkpoints, key=os.path.getmtime, reverse=True):
                ctime = datetime.fromtimestamp(os.path.getmtime(ckpt))
                print(f"     - {ckpt.name} (cap nhat: {ctime.strftime('%Y-%m-%d %H:%M:%S')})")
        else:
            print("[WARNING] Folder checkpoints ton tai nhung trong")
    else:
        print("[INFO] Chua co checkpoint nao duoc luu")
        print("       -> Checkpoint duoc luu moi 500 steps")
    
    print()
    
    # Kiem tra hparams
    hparams_file = latest_folder / "hparams.yaml"
    if hparams_file.exists():
        print("[INFO] Hyperparameters:")
        with open(hparams_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if 'max_steps' in line or 'every_n_train_steps' in line or 'learning_rate' in line:
                    print(f"       {line}")
    
    print()
    print("[TIP] De xem chi tiet training, chay:")
    print(f"      tensorboard --logdir {log_base}")
    
    # Kiem tra process Python co dang chay khong
    import subprocess
    try:
        result = subprocess.run(['powershell', '-Command', 'Get-Process | Where-Object { $_.ProcessName -like "*python*" } | Select-Object ProcessName, Id, StartTime | Format-Table -AutoSize'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and 'python' in result.stdout.lower():
            print()
            print("[INFO] Co process Python dang chay:")
            print(result.stdout)
    except:
        pass

if __name__ == "__main__":
    check_training_progress()

if __name__ == "__main__":
    check_training_progress()

