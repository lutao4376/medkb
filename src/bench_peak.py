import os,sys,time
sys.stdout.reconfigure(encoding="utf-8")
import torch
print("电源:", )
import subprocess
try:
    b=subprocess.run(["powershell","-NoProfile","-Command",
      "(Get-CimInstance Win32_Battery).BatteryStatus; (Get-CimInstance Win32_ComputerSystem).PCSystemType"],
      capture_output=True,text=True,timeout=30).stdout.strip()
    print("  电池状态/机型:", b.replace("\n"," | "), "(2=接电源)")
except Exception as e: print("  查询失败",e)

for dt in (torch.float16, torch.float32):
    a=torch.randn(4096,4096,device="cuda",dtype=dt); b=torch.randn(4096,4096,device="cuda",dtype=dt)
    for _ in range(3): a@b
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(20): a@b
    torch.cuda.synchronize(); el=(time.time()-t0)/20
    print(f"  裸 matmul 4096^3 {dt}: {el*1000:.2f} ms = {2*4096**3/el/1e12:.1f} TFLOPS")

from transformers import AutoModel, AutoTokenizer
m=AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B", dtype=torch.float16).to("cuda").eval()
p=next(m.parameters())
print(f"\n模型: dtype={p.dtype} device={p.device} 参数={sum(x.numel() for x in m.parameters())/1e6:.0f}M")
print("config.torch_dtype:", m.config.torch_dtype)
