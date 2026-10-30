from pathlib import Path
s=Path('tools/atlas/probe_shared_language.py').read_text(encoding='utf-8')
s=s.replace("output = folder / ('language_' + selection.device + '.json')", "output = folder / ('restored_language_' + selection.device + '.json')")
s=s.replace("'--compat-ops']", "'--compat-ops', '--cameras', '1', '--num-steps', '3']")
restore='''import numpy as np, gc
from optimize_runtime import install_static_loop
import lerobot.policies.smolvla.modeling_smolvla as smol_module
install_static_loop(policy,smol_module)
cfg.resize_imgs_with_padding=(256,256)
def action_probe():
    noise=torch.from_numpy(np.random.default_rng(42).standard_normal((1,cfg.chunk_size,cfg.max_action_dim)).astype('float32')).to(selection.device)
    start=time.perf_counter()
    with torch.inference_mode(),torch.autocast(device_type='npu',dtype=torch.float16):
        value=policy.predict_action_chunk(batch,noise=noise)
    torch.npu.synchronize()
    return value.detach().cpu().numpy(),time.perf_counter()-start
reference,reference_cold_seconds=action_probe()
reference,reference_warm_seconds=action_probe()
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from safetensors import safe_open
with safe_open(str(folder/'extra_layers_fp16.safetensors'),framework='pt') as f:
    for index in range(16,32):
        with torch.device('meta'):layer=LlamaDecoderLayer(vlm.config.text_config,layer_idx=index)
        prefix=f'model.text_model.layers.{index}.'
        weights={k[len(prefix):]:f.get_tensor(k) for k in f.keys() if k.startswith(prefix)}
        layer.load_state_dict(weights,strict=True,assign=True)
        vlm.model.text_model.layers.append(layer.to(selection.device))
        del weights
        gc.collect()
head = vlm.get_output_embeddings()'''
s=s.replace('head = vlm.get_output_embeddings()',restore)
s=s.replace("'real_language_layers': len(vlm.model.text_model.layers),", "'real_language_layers': len(vlm.model.text_model.layers), 'action_wrapper_layers':policy.model.vlm_with_expert.num_vlm_layers,'reference_warm_seconds':reference_warm_seconds,'added_layer_weight_mib':300.05859375,")
s=s.replace("    cases = ['What is 1 + 1? Answer briefly.',", "    probe,probe_seconds=action_probe()\n    report['action_after_restore']={'seconds':probe_seconds,'max_abs_diff':float(np.max(np.abs(probe-reference))),'exact':bool(np.array_equal(probe,reference))}\n    cases = ['What is 1 + 1? Answer briefly.',")
s=s.replace('max_new_tokens=32,','max_new_tokens=32,eos_token_id=49279,').replace('pad_token_id=tokenizer.eos_token_id','pad_token_id=2')
s=s.replace("    report['execution_ok'] = True", "    probe,probe_seconds=action_probe()\n    report['action_after_text']={'seconds':probe_seconds,'max_abs_diff':float(np.max(np.abs(probe-reference))),'exact':bool(np.array_equal(probe,reference))}\n    report['execution_ok'] = True\n    save()\n    (folder/'resident.ready').write_text(str(__import__('os').getpid()))\n    print('RESIDENT_READY',flush=True)\n    hold_start=time.monotonic()\n    while time.monotonic()-hold_start<1800 and not (folder/'resident.stop').exists():time.sleep(1)")
Path('tools/atlas/probe_restored_language.py').write_text(s,encoding='utf-8')
