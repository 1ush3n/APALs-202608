import torch

def get_available_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        try:
            free_mem, total_mem = torch.cuda.mem_get_info()
            free_gb = free_mem / (1024 ** 3)
            if free_gb < 1.0:
                print(f"[Device] 警告：GPU显存不足（剩余 {free_gb:.1f} GB），可能中途出现OOM。")
            print(f"[Device] 使用设备: {device} (GPU显存剩余: {free_gb:.1f} GB)")
        except Exception:
            print(f"[Device] 使用设备: {device}")
    else:
        device = torch.device("cpu")
        print("[Device] 使用设备: CPU（无可用GPU）")
    return device

def clear_torch_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
