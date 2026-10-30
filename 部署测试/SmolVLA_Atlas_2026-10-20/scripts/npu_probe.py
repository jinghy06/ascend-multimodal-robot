import time,json,traceback,pathlib,sys
out={'test':'torch_npu_fp16_matmul','hardware':'Atlas 200I DK A2 / Ascend310B4'}
try:
 import torch
 out['torch']=torch.__version__
 import torch_npu
 out['torch_npu']=torch_npu.__version__
 if '--no-jit' in sys.argv:
  torch.npu.set_compile_mode(jit_compile=False)
  out['jit_compile']=False
 out['available']=torch.npu.is_available()
 out['device']=torch.npu.get_device_name(0)
 t=time.perf_counter()
 x=torch.ones((64,64),dtype=torch.float16,device='npu:0')
 y=x@x
 torch.npu.synchronize()
 out.update(ok=True,seconds=time.perf_counter()-t,value=y[0,0].item())
except Exception as e:
 out.update(ok=False,error=repr(e),traceback=traceback.format_exc())
pathlib.Path('/root/smolvla-test-20261020/npu_probe.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2),flush=True)
