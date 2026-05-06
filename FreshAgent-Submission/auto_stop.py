import torch
import time
import os
import subprocess

resume_path = "models/resume_checkpoint.pth"
print("[AutoStop] Watcher started. Waiting for Epoch 3 to complete...")

while True:
    if os.path.exists(resume_path):
        try:
            ckpt = torch.load(resume_path, map_location="cpu")
            epoch = ckpt.get("epoch", 0)
            print(f"[AutoStop] Checkpoint detected: epoch={epoch}", flush=True)
            if epoch >= 3:
                print("[AutoStop] Epoch 3 reached! Stopping training...", flush=True)
                break
        except Exception as e:
            pass  # File mid-write, retry
    time.sleep(30)

# Kill train.py
print("[AutoStop] Terminating training process...")
kill_cmd = "Get-CimInstance Win32_Process | Where-Object {$_.CommandLine -like '*train.py*'} | Stop-Process -Force"
subprocess.run(["powershell", "-Command", kill_cmd])
time.sleep(5)

# ONNX Export — use correct --checkpoint argument
if os.path.exists("models/best_model.pth"):
    print("[AutoStop] Exporting to ONNX...")
    result = subprocess.run([
        "venv\\Scripts\\python.exe", "training\\export.py",
        "--checkpoint", "models\\best_model.pth",
        "--output",     "models\\freshagent.onnx"
    ])
    if result.returncode == 0:
        print("[AutoStop] ✅ ONNX export complete!")
    else:
        print("[AutoStop] ❌ ONNX export failed.")
else:
    print("[AutoStop] ❌ best_model.pth not found — cannot export.")

# Plot results
print("[AutoStop] Generating plots...")
subprocess.run(["venv\\Scripts\\python.exe", "training\\plot_results.py", "--output_dir", "models"])

print("\n[AutoStop] ==========================================")
print("[AutoStop]   ALL DONE! Check models/ folder.")
print("[AutoStop] ==========================================")
