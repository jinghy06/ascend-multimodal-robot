"""Use the actual language backbone and lm_head in the SmolVLA checkpoint."""
import argparse
import json
import pathlib
import resource
import sys
import time
import traceback

parser = argparse.ArgumentParser()
parser.add_argument('--device', choices=['npu', 'cpu'], default='npu')
selection = parser.parse_args()
folder = pathlib.Path(__file__).resolve().parent
output = folder / ('language_' + selection.device + '.json')
source = (folder / 'run_smolvla.py').read_text().split(' for i in range(args.runs):')[0]
source += '\nexcept Exception:\n raise\n'
sys.argv = ['bootstrap', '--device', selection.device, '--cpu-preprocess', '--compat-ops']
exec(compile(source, 'bootstrap', 'exec'), globals())

vlm = policy.model.vlm_with_expert.vlm
tokenizer = policy.language_tokenizer
head = vlm.get_output_embeddings()
report = {'device': selection.device, 'separate_qwen_loaded': False,
          'extra_model_weights_downloaded': False,
          'checkpoint': result['checkpoint'], 'revision': result['revision'],
          'real_language_layers': len(vlm.model.text_model.layers),
          'backbone_config_original_layers': vlm.config.text_config.num_hidden_layers,
          'lm_head_shape': list(head.weight.shape), 'prompts': []}

def save():
    report['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2))

try:
    if selection.device == 'npu':
        # Generation needs integer position ranges that the action path did not.
        original_arange = torch.arange
        def host_arange(*a, **kw):
            device = kw.get('device')
            if device is not None and str(device).startswith('npu'):
                kw['device'] = 'cpu'
                return original_arange(*a, **kw).to(device)
            return original_arange(*a, **kw)
        torch.arange = host_arange
        from transformers.models.llama.modeling_llama import LlamaModel
        original_mask = LlamaModel._prepare_4d_causal_attention_mask_with_cache_position
        def host_causal_mask(*a, **kw):
            device = kw.get('device')
            if device is not None and str(device).startswith('npu'):
                kw = {k:(v.cpu() if isinstance(v,torch.Tensor) else v) for k,v in kw.items()}
                kw['device'] = torch.device('cpu')
                return original_mask(*a, **kw).to(device)
            return original_mask(*a, **kw)
        LlamaModel._prepare_4d_causal_attention_mask_with_cache_position = staticmethod(host_causal_mask)
        rotary = vlm.model.text_model.rotary_emb
        rotary.to('cpu')
        rotary.original_inv_freq = rotary.inv_freq
        original_rotary = rotary.forward
        def host_rotary(x, position_ids):
            # RoPE only reads x.device and x.dtype, not its values or shape.
            dummy = torch.empty(1, dtype=x.dtype, device='cpu')
            cos, sin = original_rotary(dummy, position_ids.cpu())
            return cos.to(x.device), sin.to(x.device)
        rotary.forward = host_rotary
        report['generation_compatibility'] = ['small generation position ranges, causal masks, and FP32 RoPE computed on CPU']
    # HF cache metadata must describe the actual 16 retained layers, not the
    # original 32-layer backbone. This does not add or change checkpoint weights.
    vlm.model.text_model.config.num_hidden_layers = report['real_language_layers']
    vlm.model.text_model.config._attn_implementation = 'eager'
    for layer in vlm.model.text_model.layers:
        layer.self_attn.config._attn_implementation = 'eager'
    cases = ['What is 1 + 1? Answer briefly.',
             '请用中文回答：1加1等于几？',
             '请用中文简单介绍一下你能做什么。']
    for prompt in cases:
        messages = [{'role': 'user', 'content': [{'type': 'text', 'text': prompt}]}]
        text = policy.model.vlm_with_expert.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)
        batch_text = tokenizer(text, return_tensors='pt', add_special_tokens=False)
        batch_text = {k:v.to(selection.device) for k,v in batch_text.items()}
        if selection.device == 'npu':
            torch.npu.synchronize()
        started_text = time.perf_counter()
        with torch.inference_mode():
            generated = vlm.generate(**batch_text, max_new_tokens=32,
                                     do_sample=False, use_cache=True,
                                     pad_token_id=tokenizer.eos_token_id)
        if selection.device == 'npu':
            torch.npu.synchronize()
        elapsed = time.perf_counter() - started_text
        new_ids = generated[0, batch_text['input_ids'].shape[1]:].detach().cpu()
        response = tokenizer.decode(new_ids, skip_special_tokens=True)
        case = {'prompt': prompt, 'response': response, 'new_tokens': len(new_ids),
                'seconds': elapsed, 'raw_ids': new_ids.tolist()}
        report['prompts'].append(case)
        save()
        print(json.dumps(case, ensure_ascii=False), flush=True)
    report['execution_ok'] = True
except Exception:
    report['execution_ok'] = False
    report['error'] = traceback.format_exc()
    print(report['error'], flush=True)
finally:
    save()
    print(json.dumps(report, ensure_ascii=False), flush=True)
