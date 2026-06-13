"""
Wait for LOSO to finish, then start Sub-Dep training.
"""
import subprocess, time, sys, os

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:10808'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:10808'
os.chdir('D:/EEGnet/external/TCFormer')

# Wait for LOSO process to finish
loso_pid = 30944
print(f"Waiting for LOSO (PID {loso_pid}) to finish...")
while True:
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x0400, False, loso_pid)  # SYNCHRONIZE
        if handle:
            kernel32.CloseHandle(handle)
            time.sleep(30)
        else:
            break
    except:
        break

print("LOSO finished! Starting Sub-Dep training...")
subprocess.run([
    sys.executable, "train_pipeline.py",
    "--model", "tcformer",
    "--dataset", "bcic2a",
    "--interaug",
    "--gpu_id", "0",
    "--seed", "42",
])
print("Sub-Dep training completed!")
