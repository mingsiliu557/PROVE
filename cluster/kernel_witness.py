"""Seeded small CUDA/FlashAttention/vLLM kernel witness; no base-model load."""
import json,os,time
from pathlib import Path
import torch
from flash_attn import flash_attn_func
assert os.environ.get('SLURM_JOB_ID') or os.environ.get('CLUSTER_VERIFIED_JOB_ID')
assert torch.cuda.is_available() and torch.cuda.device_count()==1
expected=os.environ.get('TTRV_ASSIGNED_GPU_UUIDS')
actual=str(getattr(torch.cuda.get_device_properties(0),'uuid',''))
if expected and actual:assert actual.removeprefix('GPU-')==expected.removeprefix('GPU-'),(actual,expected)
seed=1729;torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
x=torch.randn(16,64,device='cuda',dtype=torch.float16);w=torch.ones(64,device='cuda',dtype=torch.float16)
import vllm._C
out=torch.empty_like(x);torch.ops._C.rms_norm(out,x,w,1e-6)
reference=(x.float()*torch.rsqrt(x.float().pow(2).mean(-1,keepdim=True)+1e-6)).half()
torch.testing.assert_close(out,reference,atol=.003,rtol=.003)
q=torch.randn(1,32,4,32,device='cuda',dtype=torch.float16,requires_grad=True);k=torch.randn_like(q,requires_grad=True);v=torch.randn_like(q,requires_grad=True)
a=flash_attn_func(q,k,v,causal=True);b=torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),is_causal=True).transpose(1,2)
torch.testing.assert_close(a,b,atol=.004,rtol=.004);a.float().square().mean().backward();assert torch.isfinite(q.grad).all()
net=torch.nn.Sequential(torch.nn.Linear(64,32),torch.nn.GELU(),torch.nn.Linear(32,8)).cuda();before=net[0].weight.detach().clone();optim=torch.optim.AdamW(net.parameters(),lr=.001)
y=torch.randn(4,64,device='cuda');loss=net(y).square().mean();loss.backward();optim.step();assert torch.isfinite(loss) and not torch.equal(before,net[0].weight)
r={'status':'passed','scope':'small CUDA kernels and synthetic optimizer step; not full checkpoint or RL training','seed':seed,'job_id':os.environ.get('SLURM_JOB_ID',os.environ.get('CLUSTER_VERIFIED_JOB_ID')),'transport':os.environ.get('TTRV_TRANSPORT','slurm-step'),'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],'gpu_name':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'flash_attention_forward_backward':'passed','vllm_cuda_rms_norm':'passed','synthetic_optimizer_step':'passed','loss':loss.item(),'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'timestamp':time.time()}
Path(os.environ['TTRV_KERNEL_REPORT']).write_text(json.dumps(r,indent=2));print(json.dumps(r))
