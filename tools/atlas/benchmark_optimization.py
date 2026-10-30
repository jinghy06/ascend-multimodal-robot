"""Sequential on-board synthetic benchmarks; no motors or camera devices."""
import argparse
import collections
import json
import pathlib
import statistics
import sys
import time
import resource
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--phase', default='core', choices=['core', 'batch', 'tuning'])
options = parser.parse_args()
folder = pathlib.Path(__file__).resolve().parent
output = folder / ('results_' + options.phase)
output.mkdir(exist_ok=True)
bootstrap = (folder / 'run_smolvla.py').read_text()
bootstrap = bootstrap.split(' for i in range(args.runs):')[0] + '\nexcept Exception:\n raise\n'
sys.argv = ['bootstrap', '--device', 'npu', '--cpu-preprocess', '--compat-ops']
exec(compile(bootstrap, 'bootstrap', 'exec'), globals())
from optimize_runtime import install_static_loop, install_batched_vision
import lerobot.policies.smolvla.modeling_smolvla as module

report = {'phase': options.phase, 'cases': [], 'scope': 'synthetic 3 camera images; no physical action',
          'clock': 'synchronized complete predict_action_chunk including CPU preprocessing',
          'fixed_noise_seed': 42, 'observations_recomputed_every_call': True}
baseline = None

def save_report():
    report['peak_process_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    (output / 'benchmark.json').write_text(json.dumps(report, indent=2))

def predict():
    # Original model mutates injected noise in place: regenerate for EVERY call.
    noise = torch.from_numpy(np.random.default_rng(42).standard_normal(
        (1, cfg.chunk_size, cfg.max_action_dim)).astype('float32')).to('npu')
    torch.npu.synchronize()
    t = time.perf_counter()
    with torch.inference_mode(), torch.autocast(device_type='npu', dtype=torch.float16):
        action = policy.predict_action_chunk(batch, noise=noise)
    torch.npu.synchronize()
    elapsed = time.perf_counter() - t
    return elapsed, action.detach().cpu().numpy()

def run_case(name, steps=10, repeats=3):
    global baseline
    cfg.num_steps = steps
    case = {'name': name, 'steps': steps, 'seconds': [], 'runs_finite': [],
            'camera_count': sum('.images.' in k for k in batch),
            'resize_hw': list(cfg.resize_imgs_with_padding)}
    report['cases'].append(case)
    for i in range(repeats):
        elapsed, arr = predict()
        assert arr.shape == (1, 50, 6) and np.isfinite(arr).all()
        np.save(output / f'{name}_{i}.npy', arr)
        case['seconds'].append(elapsed)
        case['runs_finite'].append(True)
        if baseline is None:
            baseline = arr.astype(np.float64)
        diff = arr.astype(np.float64) - baseline
        case['relative_l2_vs_baseline'] = float(np.linalg.norm(diff) / np.linalg.norm(baseline))
        case['max_absolute_vs_baseline'] = float(np.max(np.abs(diff)))
        save_report()
        print(json.dumps({'case': name, 'run': i, 'seconds': elapsed,
                          'relative_l2': case['relative_l2_vs_baseline']}), flush=True)
    case['warm_median_seconds'] = statistics.median(case['seconds'][1:])
    save_report()

def profile():
    timings = collections.defaultdict(list)
    restorations = []
    def wrap(obj, name, label):
        old = getattr(obj, name)
        def measured(*a, **kw):
            torch.npu.synchronize()
            start = time.perf_counter()
            result = old(*a, **kw)
            torch.npu.synchronize()
            timings[label].append(time.perf_counter() - start)
            return result
        setattr(obj, name, measured)
        restorations.append((obj, name, old))
    wrap(policy, 'prepare_images', 'image_preprocess')
    wrap(policy.model.vlm_with_expert, 'embed_image', 'vision_encoder')
    wrap(policy.model, 'embed_prefix', 'prefix_embedding_including_vision')
    wrap(policy.model, 'denoise_step', 'denoise_step_including_expert')
    wrap(policy.model, 'embed_suffix', 'suffix_embedding')
    cfg.num_steps = 10
    try:
        total, _ = predict()
    finally:
        for obj, name, old in reversed(restorations):
            setattr(obj, name, old)
    report['profile'] = {'total_with_stage_synchronization': total,
        'note': 'Nested times overlap; synchronization instrumentation changes timing.',
        'stages': {k: {'count': len(v), 'sum_seconds': sum(v), 'seconds': v}
                   for k, v in timings.items()}}
    save_report()
    print(json.dumps({'profile': report['profile']}), flush=True)

try:
    run_case('baseline_10')
    if options.phase == 'core':
        profile()
        restore_loop = install_static_loop(policy, module)
        run_case('static_loop_10')
        for steps in [5, 3, 2, 1]:
            run_case(f'static_loop_{steps}', steps)
        restore_loop()
    elif options.phase == 'tuning':
        restore_loop = install_static_loop(policy, module)
        original_batch = dict(batch)
        image_keys = [k for k in batch if '.images.' in k]
        for cameras, resolution, steps in [(3,384,10),(3,256,10),(3,256,5),
                                            (2,256,5),(1,256,5),(1,256,3)]:
            batch = {k:v for k,v in original_batch.items()
                     if '.images.' not in k or k in image_keys[:cameras]}
            cfg.resize_imgs_with_padding = (resolution,resolution)
            run_case(f'{cameras}cam_{resolution}px_{steps}step', steps)
        batch = original_batch
        cfg.resize_imgs_with_padding = (512,512)
        restore_loop()
    else:
        restore_loop = install_static_loop(policy, module)
        restore_vision = install_batched_vision(policy)
        run_case('batched_vision_static_10')
        run_case('batched_vision_static_3', 3)
        restore_vision()
        restore_loop()
    report['ok'] = True
except Exception:
    import traceback
    report['ok'] = False
    report['error'] = traceback.format_exc()
    raise
finally:
    save_report()
    print(json.dumps(report), flush=True)
