"""Bounded environment witnesses, never writes into base models/source data."""
import argparse,hashlib,importlib,importlib.metadata,json,os,platform,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--mode',required=True);p.add_argument('--report',required=True);a=p.parse_args()
root=Path(os.environ['PROJECT_ROOT']);repo=Path(os.environ['TTRV_REPO']);run=Path(os.environ['RUN_DIR']);model=Path(os.environ['BACKBONE_PATH']);data=Path(os.environ['DATA_LOCAL_DIR']);task=os.environ['TASK']
r={'mode':a.mode,'repo':str(repo),'model':str(model),'data':str(data),'job':os.getenv('SLURM_JOB_ID',os.getenv('CLUSTER_VERIFIED_JOB_ID')),'step':os.getenv('SLURM_STEP_ID'),'cuda_visible_devices':os.getenv('CUDA_VISIBLE_DEVICES'),'python':platform.python_version(),'time':time.time(),'formal_training':False}
def save():Path(a.report).write_text(json.dumps(r,indent=2))
r['git_head']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
r['packages']={n:importlib.metadata.version(n) for n in ['torch','transformers','vllm','ray','flash-attn','tensordict','peft','datasets','pyarrow']}
for n in ['torch','torchvision','transformers','ray','peft','flash_attn','vllm','verl.trainer.main_ppo','verl.workers.fsdp_workers','verl.utils.checkpoint.fsdp_checkpoint_manager']:
 importlib.import_module(n)
import torch,pyarrow.parquet as pq
from PIL import Image
from verl.utils import hf_tokenizer,hf_processor
from transformers import AutoConfig
cfg=AutoConfig.from_pretrained(model,trust_remote_code=True,local_files_only=True);tok=hf_tokenizer(str(model),trust_remote_code=True,local_files_only=True);processor=hf_processor(str(model));assert processor is not None
r['model_type']=cfg.model_type;r['tokenizer_class']=type(tok).__name__;r['processor_class']=type(processor).__name__
def image_paths(v):
 if isinstance(v,str) and v.startswith('/'):yield v
 elif isinstance(v,dict):
  for k,z in v.items():
   if k in ('image','path'):yield from image_paths(z)
 elif isinstance(v,(list,tuple)):
  for z in v:yield from image_paths(z)
paths=set();rows=0;tables=[]
tasks=['ai2d_20','aokvqa_da_val_20_hq_major_correct','capture_20','crpe_20_hq_major_correct20_seed42','mathverse_20_choice_norm_promptfix','mathvista_20_hq_train13_repeat20','mme_20','ocrbench_v1_20_hq_major_correct','realworldqa_20_aligned_1000_hq_major_correct20_20260614_114652','seed_20','visualsimpleqa_da_20_hq_major_correct']
assert task in tasks,task
selected=[data/t/(split+'.parquet') for t in (tasks if a.mode=='cpu' else [task]) for split in ['train','test']]
assert all(f.is_file() for f in selected), 'Required train/test Parquet is missing'
for f in selected:
 t=pq.read_table(f);rows+=len(t);tables.append(str(f))
 for row in t.to_pylist():
  for key in ['image','images','image_path']:
   for value in image_paths(row.get(key)):
    q=Path(value);assert q.is_relative_to(root/'benchmark/ttrv'),value;assert q.is_file(),value;paths.add(value)
assert paths,'No dataset images found'
for name in sorted(paths):
 with Image.open(name) as im:im.load()
r['data_check']={'tables':len(tables),'rows':rows,'images_decoded':len(paths)}
if a.mode=='train-smoke':
 for split,limit in [('train',max(2,int(os.environ['NO_GPU']))),('test',1)]:
  t=pq.read_table(data/task/(split+'.parquet'));out=run/'smoke_data'/task;out.mkdir(parents=True,exist_ok=True);pq.write_table(t.slice(0,limit),out/(split+'.parquet'))
if a.mode=='gpu-smoke':
 assert (os.environ.get('SLURM_JOB_ID') or os.environ.get('CLUSTER_VERIFIED_JOB_ID')) and torch.cuda.is_available();torch.manual_seed(1729);torch.cuda.manual_seed_all(1729)
 r['gpu']={'name':torch.cuda.get_device_name(0),'count':torch.cuda.device_count(),'memory_bytes':torch.cuda.get_device_properties(0).total_memory,'cuda_runtime':torch.version.cuda}
 from flash_attn import flash_attn_func
 q=torch.randn(1,32,4,32,device='cuda',dtype=torch.float16,requires_grad=True);k=torch.randn_like(q,requires_grad=True);v=torch.randn_like(q,requires_grad=True)
 actual=flash_attn_func(q,k,v,causal=True);expected=torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),is_causal=True).transpose(1,2)
 torch.testing.assert_close(actual,expected,atol=0.004,rtol=0.004);actual.float().square().mean().backward();assert torch.isfinite(q.grad).all()
 r['seeded_flash_attention_forward_backward']='passed';save()
 from transformers import AutoModel,Qwen2_5_VLForConditionalGeneration
 cls=Qwen2_5_VLForConditionalGeneration if 'qwen' in cfg.model_type else AutoModel
 kwargs={'trust_remote_code':True,'local_files_only':True,'torch_dtype':torch.bfloat16,'low_cpu_mem_usage':True,'device_map':{'':0}}
 if cls is AutoModel:kwargs['use_flash_attn']=True
 else:kwargs['attn_implementation']='flash_attention_2'
 r['stage']='loading_full_checkpoint';save();print('STAGE loading_full_checkpoint',flush=True)
 net=cls.from_pretrained(str(model),**kwargs);r['stage']='full_checkpoint_loaded';save();print('STAGE full_checkpoint_loaded',flush=True);net.train();net.gradient_checkpointing_enable()
 # Bounded language-head forward/backward on the full loaded checkpoint. Image
 # decoding/processor loading is tested above; this is not end-to-end RL proof.
 language=getattr(net,'language_model',net);ids=tok('This is a minimal training environment verification.',return_tensors='pt')['input_ids'].to('cuda')
 r['stage']='full_model_forward';save();print('STAGE full_model_forward',flush=True)
 output=language(input_ids=ids,attention_mask=torch.ones_like(ids),labels=ids,use_cache=False)
 assert torch.isfinite(output.loss);r['stage']='full_model_backward';save();print('STAGE full_model_backward',flush=True);output.loss.backward();grads=[x.grad for x in language.parameters() if x.grad is not None];assert grads and all(torch.isfinite(x).all().item() for x in grads)
 r['full_model_language_loss']=output.loss.item();r['full_model_backward']='passed';r['peak_cuda_bytes']=torch.cuda.max_memory_allocated();del net,language;torch.cuda.empty_cache()
r['status']='passed';save();print(json.dumps(r))
