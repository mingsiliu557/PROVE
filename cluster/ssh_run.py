"""Run on an already allocated node via SSH, without sbatch or srun."""
import argparse,json,os,re,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--job',required=True,type=int);p.add_argument('--mode',choices=['kernel','gpu-smoke','train-smoke','train'],default='kernel');p.add_argument('--gpu-count',type=int,default=1);p.add_argument('overrides',nargs='*');a=p.parse_args()
here=Path(__file__).resolve().parent;repo=here.parent;root=Path(os.environ.get('PROJECT_ROOT',str(repo.parent))).expanduser().resolve()
assert repo.parent==root and repo.name in ['TTRV','TTRV-intuitor-baseline','TTRV-qwen25vl7b']
s=subprocess.check_output(['scontrol','show','job','-dd',str(a.job)],text=True)
assert re.search(r'UserId=[^ ]+\('+str(os.getuid())+r'\)',s) and 'JobState=RUNNING' in s
node=os.uname().nodename.split('.')[0]
m=re.search(r'Nodes=('+re.escape(node)+r') CPU_IDs=([0-9,-]+) Mem=(\d+) GRES=gpu:[^\n]*?\(IDX:([0-9,-]+)\)',s)
assert m, 'This node is not in the verified single-node GPU allocation'
def expand(v):
 out=[]
 for part in v.split(','):
  a,sep,b=part.partition('-');out.extend(range(int(a),int(b)+1) if sep else [int(a)])
 return out
cpus=expand(m[2]);gpus=expand(m[4]);assert 1<=a.gpu_count<=len(gpus)
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True)
by_index={int(x.split(',')[0]):[v.strip() for v in x.split(',')] for x in rows.splitlines()}
selected=gpus[:a.gpu_count]
assert all(int(by_index[g][2])<512 for g in selected),'Selected allocated GPU is busy; do not interfere'
os.sched_setaffinity(0,cpus[:min(4,len(cpus))])
env=os.environ.copy()
for k in list(env):
 if k.startswith('SLURM_') or k in ['http_proxy','https_proxy','all_proxy','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY']:env.pop(k,None)
env.update(PROJECT_ROOT=str(root),CLUSTER_VERIFIED_JOB_ID=str(a.job),TTRV_TRANSPORT='direct-ssh-tmux',TTRV_REQUESTED_MEMORY_MB=m[3],TTRV_ALLOCATED_CPUS=str(min(4,len(cpus))),CUDA_VISIBLE_DEVICES=','.join(str(g) for g in selected),TTRV_ASSIGNED_GPU_UUIDS=','.join(by_index[g][1] for g in selected),CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='2',PYTHONNOUSERSITE='1',PYTHONPATH=str(repo/'verl'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
j=root/'logs/ttrv-training-setup-20261025';j.mkdir(parents=True,exist_ok=True)
tag=f'{repo.name}-{a.mode}-ssh-{os.getpid()}'
env.update(TTRV_GUARD_REPORT=str(j/(tag+'-guard.json')),TTRV_KERNEL_REPORT=str(j/(tag+'-kernel.json')))
(j/(tag+'-allocation.json')).write_text(json.dumps({'job':a.job,'host':node,'allocated_gpu_indices':gpus,'selected_gpu_indices':selected,'visible_gpu_uuids':env['TTRV_ASSIGNED_GPU_UUIDS'],'cuda_visible_devices':env['CUDA_VISIBLE_DEVICES'],'allocated_cpu_ids':cpus,'selected_cpu_ids':sorted(os.sched_getaffinity(0)),'requested_memory_mb':int(m[3]),'transport':'direct-ssh-tmux'},indent=2))
py=str(root/'envs/ttrv/bin/python')
command=[py,str(here/'kernel_witness.py')] if a.mode=='kernel' else ['bash',str(here/'run.sh'),a.mode,*a.overrides]
print(json.dumps({'tag':tag,'job':a.job,'mode':a.mode,'gpu_indices':selected,'new_slurm_job_or_step':False}),flush=True)
os.execve(py,[py,str(here/'guard.py'),*command],env)
