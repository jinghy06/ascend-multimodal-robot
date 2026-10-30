import torch,json,time,pathlib,traceback
from safetensors import safe_open
from transformers import AutoConfig,AutoModelForImageTextToText,AutoTokenizer
base=pathlib.Path('tools/atlas/downloads');out=pathlib.Path('部署测试/常驻集成验证_2026-10-21/完整语言骨干本地实测.json')
report={'device':'cuda','torch':torch.__version__,'note':'Local GPU correctness test; not Atlas latency','cases':[]}
try:
 config=AutoConfig.from_pretrained(base/'smolvlm_config',local_files_only=True)
 config._attn_implementation='eager';config.text_config._attn_implementation='eager'
 model=AutoModelForImageTextToText.from_config(config,dtype=torch.float16)
 state={}
 with safe_open(base/'smolvlm_full/model.safetensors',framework='pt') as f:
  for k in f.keys():state[k]=f.get_tensor(k).to(dtype=torch.float16)
 # All retained tensors come from the policy, not a second independent backbone.
 with safe_open(base/'smolvla_base/model.safetensors',framework='pt') as f:
  prefix='model.vlm_with_expert.vlm.'
  for k in f.keys():
   if k.startswith(prefix):state[k[len(prefix):]]=f.get_tensor(k).to(dtype=torch.float16)
 incompatible=model.load_state_dict(state,strict=True,assign=True);del state
 model=model.eval().to('cuda');tokenizer=AutoTokenizer.from_pretrained(base/'smolvlm_config',local_files_only=True);tokenizer.chat_template=json.loads((base/'smolvlm_config/chat_template.json').read_text())['chat_template'];from types import SimpleNamespace;processor=SimpleNamespace(tokenizer=tokenizer,apply_chat_template=tokenizer.apply_chat_template)
 report['layers']=len(model.model.text_model.layers);report['parameters']=sum(p.numel() for p in model.parameters());report['load_keys']=str(incompatible)
 prompts=['What is 1 + 1? Answer briefly.','请用中文回答：1加1等于几？','请用中文简单介绍一下你能做什么。','中国的首都是哪里？请用中文回答。','如果机器人前方有人，应该先停下还是继续前进？请用中文简短回答。','请把“帮我拿红色杯子”解释成机器人需要做的步骤，用中文回答。','请用中文解释为什么天空通常是蓝色的。','请用中文回答：3乘以7等于多少？','请用中文介绍西安交通大学。','请用中文写一句拿到物品后的礼貌回复。']
 def generate(messages):
  text=processor.apply_chat_template(messages,add_generation_prompt=True,tokenize=False)
  batch=processor.tokenizer(text,return_tensors='pt',add_special_tokens=False).to('cuda')
  first=[None]
  def capture(module,inputs,result):
   if first[0] is None:torch.cuda.synchronize();first[0]=time.perf_counter()
   return result
  hook=model.register_forward_hook(capture);torch.cuda.synchronize();start=time.perf_counter()
  try:
   with torch.inference_mode():ids=model.generate(**batch,max_new_tokens=96,do_sample=False,use_cache=True,pad_token_id=2,eos_token_id=49279)
   torch.cuda.synchronize();end=time.perf_counter()
  finally:hook.remove()
  ids=ids[0,batch['input_ids'].shape[1]:].cpu();return {'response':processor.tokenizer.decode(ids,skip_special_tokens=True),'new_tokens':len(ids),'seconds':end-start,'first_forward_seconds':first[0]-start,'raw_ids':ids.tolist()}
 for p in prompts:
  row={'prompt':p,**generate([{'role':'user','content':[{'type':'text','text':p}]}])};report['cases'].append(row)
  out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(row,ensure_ascii=False),flush=True)
 messages=[{'role':'user','content':[{'type':'text','text':'请记住我的名字叫小明，用中文回复。'}]}]
 row=generate(messages);messages.append({'role':'assistant','content':[{'type':'text','text':row['response']}]});messages.append({'role':'user','content':[{'type':'text','text':'我叫什么名字？请用中文回答。'}]})
 report['two_turn']={'first':row,'second':generate(messages)}
 report['gpu_peak_allocated_mib']=torch.cuda.max_memory_allocated()/2**20;report['execution_ok']=True
except Exception:report['execution_ok']=False;report['error']=traceback.format_exc();print(report['error'],flush=True)
out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
