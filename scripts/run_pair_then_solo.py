"""
Run ATCNet + EEGTCNet in parallel, then EEGConformer solo.
"""
import subprocess, sys, os

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:10808'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:10808'
os.chdir('D:/EEGnet/external/TCFormer')

BASE_CMD = [sys.executable, 'train_pipeline.py', '--dataset', 'bcic2a',
            '--loso', '--gpu_id', '0', '--seed', '42']

# Phase 1: ATCNet + EEGTCNet in parallel
print("=" * 60)
print("Phase 1: ATCNet + EEGTCNet (parallel)")
print("=" * 60)

import threading

def run_model(model, tag):
    log = open(f'D:/EEGnet/results/logs/{tag}_loso.log', 'w')
    p = subprocess.run(BASE_CMD + ['--model', model], stdout=log, stderr=subprocess.STDOUT)
    log.close()
    print(f"{tag} exit: {p.returncode}")

os.makedirs('D:/EEGnet/results/logs', exist_ok=True)

t1 = threading.Thread(target=run_model, args=('atcnet', 'atcnet'))
t2 = threading.Thread(target=run_model, args=('eegtcnet', 'eegtcnet'))
t1.start(); t2.start()
t1.join(); t2.join()

# Phase 2: EEGConformer solo
print("=" * 60)
print("Phase 2: EEGConformer (solo)")
print("=" * 60)
run_model('eegconformer', 'eegconformer')
print("ALL DONE!")
