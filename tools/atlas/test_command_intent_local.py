import torch,json,time,pathlib,traceback
from safetensors import safe_open
from transformers import AutoConfig,AutoModelForImageTextToText,AutoTokenizer
base=pathlib.Path('tools/atlas/downloads');out=pathlib.Path('部署测试/常驻集成验证_2026-10-21/中文控制意图本地实测.json')
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

 cases=[('跟着我走','A'),('跟随我','A'),('你跟在我后面','A'),('陪我往前走','A'),('去红点那里','B'),('走到激光笔指的位置','B'),('移动到地上的红色光点','B'),('到那个红点的位置去','B'),('捡起红色方块','C'),('把红色积木拿起来','C'),('帮我抓住红色方块','C'),('拿一下桌上的红色方块','C'),('停止','D'),('别动了','D'),('取消当前任务','D'),('现在停下来','D'),('你好','E'),('今天星期几','E'),('不要跟着我','D'),('不要捡东西，停下','D')]
 instruction='Classify a robot command. Reply with one letter only. A: follow the speaker. B: move to the red laser spot. C: pick up the red block. D: stop or cancel. E: other, unclear, or unrelated. Command: '
 saved=model.model.text_model.layers
 for layers in [32,16]:
  model.model.text_model.layers=torch.nn.ModuleList(list(saved)[:layers]);model.config.text_config.num_hidden_layers=layers
  rows=[]
  for phrase,expected in cases:
   messages=[{'role':'user','content':[{'type':'text','text':instruction+phrase}]}]
   text=processor.apply_chat_template(messages,add_generation_prompt=True,tokenize=False)
   ids=tokenizer(text,return_tensors='pt',add_special_tokens=False).input_ids.to('cuda');scores={};start=time.perf_counter()
   with torch.inference_mode():
    for letter in 'ABCDE':
     suffix=tokenizer(' '+letter,return_tensors='pt',add_special_tokens=False).input_ids.to('cuda')
     full=torch.cat([ids,suffix],dim=1)
     logits=model(input_ids=full,use_cache=False).logits[:,ids.shape[1]-1:-1].float()
     scores[letter]=float(logits.log_softmax(-1).gather(-1,suffix.unsqueeze(-1)).sum())
   torch.cuda.synchronize();pred=max(scores,key=scores.get)
   row={'command':phrase,'expected':expected,'predicted':pred,'correct':pred==expected,'scores':scores,'seconds_five_candidates':time.perf_counter()-start};rows.append(row)
   print(json.dumps({'layers':layers,**row},ensure_ascii=False),flush=True)
  report['cases'].append({'layers':layers,'correct':sum(r['correct'] for r in rows),'total':len(rows),'rows':rows})
  out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
 report['execution_ok']=True
except Exception:report['execution_ok']=False;report['error']=traceback.format_exc();print(report['error'],flush=True)
out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
