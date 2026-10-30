"""Pretrained SmolVLA runtime benchmark only. Does not import or command robot drivers."""
import os,time,json,pathlib,traceback,resource,argparse,gc,contextlib,sys
base=pathlib.Path('/root/smolvla-test-20261020')
os.environ.setdefault('HF_HOME',str(base/'cache/huggingface'))
os.environ.setdefault('HF_HUB_DISABLE_XET','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
args=argparse.ArgumentParser();args.add_argument('--device',default='cpu');args.add_argument('--runs',type=int,default=2);args.add_argument('--cameras',type=int,default=3);args.add_argument('--num-steps',type=int,default=10);args.add_argument('--cpu-preprocess',action='store_true');args.add_argument('--compat-ops',action='store_true');args=args.parse_args()
result={'checkpoint':'lerobot/smolvla_base','revision':'5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6','device':args.device,'input':'deterministic synthetic images and zero NORMALIZED joint state; no physical action','output_units':'normalized action space; not calibrated joint commands','camera_count':args.cameras,'image_preprocess_hw':[512,512],'denoising_steps':args.num_steps,'chunk_size':50,'action_dim':6,'timings_seconds':[]}
def save():
 result['peak_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
 (base/f'result_{args.device}_{args.cameras}cam.json').write_text(json.dumps(result,indent=2))
def status(stage):
 print(json.dumps({'stage':stage,'elapsed':time.perf_counter()-started,'rss_peak_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}),flush=True)
started=time.perf_counter()
try:
 import torch
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 from accelerate import init_empty_weights
 from safetensors.torch import load_file
 from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
 from lerobot.configs.types import NormalizationMode
 from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy,standardise_state_dict
 if args.compat_ops and args.device=='npu':
  import torch.nn.functional as functional
  import math
  original_gelu=functional.gelu
  def decomposed_gelu(x,approximate='none'):
   if x.device.type!='npu':return original_gelu(x,approximate=approximate)
   a=x.float()
   if approximate=='tanh':y=0.5*a*(1+torch.tanh(math.sqrt(2/math.pi)*(a+0.044715*a*a*a)))
   else:y=0.5*a*(1+torch.erf(a/math.sqrt(2)))
   return y.to(x.dtype)
  functional.gelu=decomposed_gelu
  original_silu=functional.silu
  def decomposed_silu(x,inplace=False):
   if x.device.type!='npu':return original_silu(x,inplace=inplace)
   a=x.float()
   y=(a*torch.sigmoid(a)).to(x.dtype)
   if inplace:x.copy_(y);return x
   return y
  functional.silu=decomposed_silu
  original_linear=functional.linear
  def unfused_linear(x,weight,bias=None):
   if x.device.type!='npu':return original_linear(x,weight,bias)
   y=torch.matmul(x,weight.transpose(-1,-2))
   return y if bias is None else y+bias
  functional.linear=unfused_linear
  original_where=torch.where
  def compatible_where(condition,input=None,other=None,**kwargs):
   if condition.device.type!='npu':
    return original_where(condition) if input is None else original_where(condition,input,other,**kwargs)
   if isinstance(input,torch.Tensor) and input.dtype==torch.float32 and isinstance(other,(int,float)) and math.isfinite(other) and not kwargs:
    mask=condition.to(dtype=input.dtype)
    return mask*input+(1-mask)*other
   def host(v):return v.cpu() if isinstance(v,torch.Tensor) else v
   out=original_where(host(condition),host(input),host(other),**kwargs)
   return out.to(condition.device)
  torch.where=compatible_where
  original_bucketize=torch.bucketize
  def host_bucketize(input,boundaries,**kwargs):
   if input.device.type!='npu':return original_bucketize(input,boundaries,**kwargs)
   return original_bucketize(input.float().cpu(),boundaries.float().cpu(),**kwargs).to(input.device)
  torch.bucketize=host_bucketize
  import lerobot.policies.smolvla.modeling_smolvla as smol_module
  original_pos_embedding=smol_module.create_sinusoidal_pos_embedding
  def host_pos_embedding(time,dimension,min_period,max_period,device='cpu'):
   if str(device).startswith('npu'):
    return original_pos_embedding(time.cpu(),dimension,min_period,max_period,device=torch.device('cpu')).float().to(device)
   return original_pos_embedding(time,dimension,min_period,max_period,device=device)
  smol_module.create_sinusoidal_pos_embedding=host_pos_embedding
  result['compatibility_patches']=['GELU decomposed into equivalent scalar formula on NPU','SiLU decomposed into x times sigmoid(x) on NPU','Linear decomposed into matmul plus bias to avoid unavailable Addmm binary','finite FP32 scalar attention masking uses boolean arithmetic instead of SelectV2; other where calls use CPU','small bucketize index calculation on CPU','small sinusoidal time embedding computed on CPU then cast to float32 and transferred to NPU']
 status('imports_ok')
 import draccus
 raw_cfg=json.loads((base/'models/smolvla_base/config.json').read_text())
 raw_cfg.pop('type',None)
 cfg=draccus.decode(SmolVLAConfig,raw_cfg)
 cfg.device='cpu';cfg.load_vlm_weights=False;cfg.num_steps=args.num_steps
 cfg.vlm_model_name=str(base/'models/smolvlm_config')
 cfg.normalization_mapping={k:NormalizationMode.IDENTITY for k in ['VISUAL','STATE','ACTION']}
 with init_empty_weights():
  policy=SmolVLAPolicy(cfg)
 status('architecture_created_on_meta')
 weights=load_file(str(base/'models/smolvla_base/model.safetensors'),device='cpu')
 weights={k.replace('model._orig_mod.','model.'):v for k,v in weights.items()}
 weights,_=standardise_state_dict(weights,set(policy.state_dict()),verbose=False)
 missing,unexpected=policy.load_state_dict(weights,strict=False,assign=True)
 # Safetensors omit some tied weights. Restore the model's declared tying before auditing.
 policy.model.vlm_with_expert.vlm.tie_weights()
 unloaded=[k for k,v in policy.named_parameters() if v.is_meta]
 result.update(torch=torch.__version__,missing_keys=missing,unexpected_keys=unexpected,unloaded_parameters=unloaded,parameter_count=sum(v.numel() for v in policy.parameters()))
 if unloaded or unexpected:raise RuntimeError(f'Checkpoint load incomplete: {unloaded}; unexpected={unexpected}')
 del weights;gc.collect()
 if args.device=='cpu':policy=policy.float()
 policy.eval()
 if args.device=='npu':
  import torch_npu
  torch.npu.set_compile_mode(jit_compile=False)
  policy=policy.to(dtype=torch.float16,device='npu:0')
 if args.cpu_preprocess and args.device=='npu':
  original_prepare=policy.prepare_images
  def prepare_on_cpu(batch):
   host_batch={k:(v.detach().cpu() if isinstance(v,torch.Tensor) and '.images.' in k else v) for k,v in batch.items()}
   images,masks=original_prepare(host_batch)
   return [x.to('npu:0') for x in images],[x.to('npu:0') for x in masks]
  policy.prepare_images=prepare_on_cpu
  result['cpu_image_preprocessing']=True
 status('pretrained_weights_loaded')
 batch={'observation.state':torch.zeros(1,6,device=args.device),'task':['pick up the red cup']}
 for i,key in enumerate(cfg.image_features):
  if i>=args.cameras:break
  # Deterministic gradients give a non-uniform numeric input without using camera hardware.
  img=torch.linspace(0,1,256).view(1,1,1,256).expand(1,3,256,256).contiguous()
  batch[key]=img.to(args.device)
 result['image_keys']=[k for k in batch if '.images.' in k]
 for i in range(args.runs):
  torch.manual_seed(42+i)
  import numpy as np
  noise=torch.from_numpy(np.random.default_rng(42+i).standard_normal((1,cfg.chunk_size,cfg.max_action_dim)).astype('float32')).to(args.device)
  if args.device=='npu':torch.npu.synchronize()
  t=time.perf_counter()
  with torch.inference_mode(), (torch.autocast(device_type='npu',dtype=torch.float16) if args.device=='npu' else contextlib.nullcontext()):action=policy.predict_action_chunk(batch,noise=noise)
  if args.device=='npu':torch.npu.synchronize()
  dt=time.perf_counter()-t
  arr=action.detach().cpu()
  np.save(base/f'actions_{args.device}_{args.cameras}cam_run{i+1}.npy',arr.numpy())
  result['timings_seconds'].append(dt)
  result.update(output_shape=list(arr.shape),output_finite=bool(torch.isfinite(arr).all()),output_first_action=arr[0,0,:].tolist(),output_min=float(arr.min()),output_max=float(arr.max()))
  save();status(f'run_{i+1}_complete');print(json.dumps({'run':i+1,'seconds':dt,'shape':list(arr.shape),'finite':result['output_finite']}),flush=True)
 result['ok']=True
except Exception as e:
 result.update(ok=False,error=repr(e),traceback=traceback.format_exc());print(result['traceback'],flush=True)
finally:
 result['total_seconds']=time.perf_counter()-started
 save();print(json.dumps(result,indent=2),flush=True)
if not result.get('ok'):sys.exit(1)
