"""Synthetic inference only. Defaults preserve the original operating point."""
import argparse
import datetime
import json
import pathlib
import statistics
import sys
import time
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--cameras', type=int, choices=[1, 2, 3], default=3)
parser.add_argument('--resolution', type=int, choices=[256, 384, 512], default=512)
parser.add_argument('--steps', type=int, choices=range(1, 11), default=10)
parser.add_argument('--runs', type=int, default=5)
selection = parser.parse_args()
if selection.runs < 1:
    parser.error('--runs must be positive')
folder = pathlib.Path(__file__).resolve().parent
output = folder / ('inference_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
output.mkdir()
bootstrap = (folder / 'run_smolvla.py').read_text().split(' for i in range(args.runs):')[0]
bootstrap += '\nexcept Exception:\n raise\n'
sys.argv = ['bootstrap', '--device', 'npu', '--cpu-preprocess', '--compat-ops',
            '--cameras', str(selection.cameras), '--num-steps', str(selection.steps)]
exec(compile(bootstrap, 'bootstrap', 'exec'), globals())
from optimize_runtime import install_static_loop
import lerobot.policies.smolvla.modeling_smolvla as module
restore = install_static_loop(policy, module)
cfg.resize_imgs_with_padding = (selection.resolution, selection.resolution)
record = {'scope': 'synthetic inputs; normalized state/actions; no hardware control',
          'cameras': selection.cameras, 'resolution': selection.resolution,
          'steps': selection.steps, 'chunk_shape': [1, 50, 6],
          'operating_point_changed': (selection.cameras, selection.resolution, selection.steps) != (3, 512, 10),
          'cross_observation_cache': False, 'seconds': [], 'seeds': []}
try:
    # First call is explicitly excluded from warm timing statistics.
    for i in range(selection.runs + 1):
        seed = 42 + i
        noise = torch.from_numpy(np.random.default_rng(seed).standard_normal(
            (1, cfg.chunk_size, cfg.max_action_dim)).astype('float32')).to('npu')
        torch.npu.synchronize()
        start = time.perf_counter()
        with torch.inference_mode(), torch.autocast(device_type='npu', dtype=torch.float16):
            actions = policy.predict_action_chunk(batch, noise=noise)
        torch.npu.synchronize()
        elapsed = time.perf_counter() - start
        arr = actions.detach().cpu().numpy()
        assert arr.shape == (1, 50, 6) and np.isfinite(arr).all()
        np.save(output / f'actions_seed{seed}.npy', arr)
        record['seconds'].append(elapsed)
        record['seeds'].append(seed)
        record['all_finite'] = True
        (output / 'result.json').write_text(json.dumps(record, indent=2))
        print(json.dumps({'warmup': i == 0, 'seed': seed, 'seconds': elapsed}), flush=True)
    record['warm_median_seconds'] = statistics.median(record['seconds'][1:])
    record['warm_min_seconds'] = min(record['seconds'][1:])
    record['warm_max_seconds'] = max(record['seconds'][1:])
    record['ok'] = True
finally:
    restore()
    (output / 'result.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)
