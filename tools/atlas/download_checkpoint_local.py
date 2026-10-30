import urllib.request,pathlib,concurrent.futures,time,json,struct
base=pathlib.Path(__file__).parent/'downloads'/'smolvla_base';base.mkdir(parents=True,exist_ok=True)
url='https://huggingface.co/lerobot/smolvla_base/resolve/5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6/model.safetensors'
size=906712520;block=32*1024*1024;t=time.time()
def getpart(start):
 end=min(start+block,size)-1;p=base/f'part-{start:010d}'
 if p.exists() and p.stat().st_size==end-start+1:return p
 for retry in range(4):
  try:
   req=urllib.request.Request(url+f'?download=true&segment={start}',headers={'Range':f'bytes={start}-{end}'})
   with urllib.request.urlopen(req,timeout=90) as r:
    if r.status!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{size}':raise RuntimeError(str(dict(r.headers)))
    with p.open('wb') as f:
     while data:=r.read(1024*1024):f.write(data)
   if p.stat().st_size!=end-start+1:raise RuntimeError('short download')
   print(json.dumps({'part_start':start,'part_bytes':p.stat().st_size,'elapsed':round(time.time()-t,1)}),flush=True);return p
  except Exception as ex:
   if retry==3:raise
   print(json.dumps({'retry':retry,'part':start,'error':str(ex)}),flush=True);time.sleep(2)
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
 list(pool.map(getpart,range(0,size,block)))
out=base/'model.safetensors'
with out.open('wb') as f:
 for start in range(0,size,block):
  p=base/f'part-{start:010d}'
  with p.open('rb') as src:
   while data:=src.read(4*1024*1024):f.write(data)
  p.unlink()
print(json.dumps({'completed':str(out),'bytes':out.stat().st_size,'seconds':time.time()-t}),flush=True)
