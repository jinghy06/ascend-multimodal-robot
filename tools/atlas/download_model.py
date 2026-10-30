import os,json,time,pathlib
from huggingface_hub import snapshot_download
base=pathlib.Path('/root/smolvla-test-20261020')
t=time.time()
folder=snapshot_download('lerobot/smolvla_base',revision='5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6',local_dir=str(base/'models/smolvla_base'),allow_patterns=['config.json','model.safetensors','*processor*.json','*processor*.safetensors'],max_workers=2)
print(json.dumps({'model_dir':folder,'download_seconds':time.time()-t}),flush=True)
