import json,pathlib
from safetensors import safe_open
import torch
out=pathlib.Path('部署测试/常驻集成验证_2026-10-21')
report={'shared':[],'different':[],'missing_in_full':[],'extra_text_layers_bytes_fp16':0}
with safe_open('tools/atlas/downloads/smolvla_base/model.safetensors',framework='pt') as v, safe_open('tools/atlas/downloads/smolvlm_full/model.safetensors',framework='pt') as f:
 full=set(f.keys()); prefix='model.vlm_with_expert.vlm.'
 for key in v.keys():
  if not key.startswith(prefix):continue
  short=key[len(prefix):]
  if short not in full:report['missing_in_full'].append(short);continue
  a=v.get_tensor(key);b=f.get_tensor(short)
  # VLA stores BF16: compare full weights after exactly the same conversion.
  match=torch.equal(a,b.to(a.dtype))
  row={'key':short,'shape':list(a.shape),'vla_dtype':str(a.dtype),'full_dtype':str(b.dtype),'equal_after_cast':match,'max_abs_after_cast':float((a.float()-b.to(a.dtype).float()).abs().max())}
  report['shared' if match else 'different'].append(row)
 for key in f.keys():
  if key.startswith('model.text_model.layers.') and int(key.split('.')[3])>=16:
   s=f.get_slice(key);n=1
   for d in s.get_shape():n*=d
   report['extra_text_layers_bytes_fp16']+=n*2
report['summary']={k:len(report[k]) for k in ('shared','different','missing_in_full')}
(out/'权重兼容性.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ['shared','different']},ensure_ascii=False))
