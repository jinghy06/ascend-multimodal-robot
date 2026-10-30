"""Read-only whole-board memory sampling, with no torch import."""
import json
import pathlib
import sys
import time
parent = pathlib.Path('/proc') / sys.argv[1]
target = pathlib.Path(__file__).parent / 'memory.jsonl'
start = time.monotonic()
with target.open('w') as out:
    while parent.exists() and time.monotonic() - start < 650:
        info = {k: int(v.split()[0]) for k, v in
                (line.split(':', 1) for line in pathlib.Path('/proc/meminfo').read_text().splitlines())}
        sample = {'elapsed': time.monotonic() - start,
                  **{k + '_kib': info[k] for k in
                     ['MemTotal', 'MemAvailable', 'MemFree', 'Cached', 'Shmem', 'SwapTotal', 'SwapFree']}}
        out.write(json.dumps(sample) + '\n')
        out.flush()
        time.sleep(2)
