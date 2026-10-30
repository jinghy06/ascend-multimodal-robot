"""Linux /proc measurements of isolated real ROS binaries. No robot drivers."""
import json, os, pathlib, statistics, subprocess, time, signal
root=pathlib.Path('/root/resident-integration-20261021')
prefix=root/'ros'; out=root/'ros-memory';out.mkdir(exist_ok=True)
os.environ.update(ROS_DOMAIN_ID='79',ROS_LOCALHOST_ONLY='1')
def kv(path):
    d={}
    for line in pathlib.Path(path).read_text().splitlines():
        p=line.replace(':',' ').split()
        if len(p)>=2:
            try:d[p[0]]=int(p[1])
            except ValueError:pass
    return d
def proc(pid):
    try:
        s=kv('/proc/%s/smaps_rollup'%pid);st=kv('/proc/%s/status'%pid)
        return {'pid':pid,'pss_kib':s.get('Pss',0),'rss_kib':s.get('Rss',0),'uss_kib':s.get('Private_Clean',0)+s.get('Private_Dirty',0),'swap_kib':st.get('VmSwap',0)}
    except (FileNotFoundError,ProcessLookupError):return None
def descendants(pid):
    found={pid};changed=True
    while changed:
        changed=False
        for p in pathlib.Path('/proc').glob('[0-9]*'):
            try:
                st=kv(p/'status')
                if st.get('PPid') in found and int(p.name) not in found:found.add(int(p.name));changed=True
            except (FileNotFoundError,ProcessLookupError):pass
    return found
def phase(name,commands,duration=20):
    jobs=[]; logs=[]
    for i,command in enumerate(commands):
        f=(out/f'{name}-{i}.log').open('w');logs.append(f)
        jobs.append(subprocess.Popen([str(x) for x in command],stdout=f,stderr=subprocess.STDOUT,start_new_session=True))
    time.sleep(5);rows=[]
    try:
        for i in range(duration):
            pids=set()
            for job in jobs:
                if job.poll() is not None:raise RuntimeError(f'{name} exited {job.returncode}; inspect log')
                pids.update(descendants(job.pid))
            processes=[v for pid in pids if (v:=proc(pid)) is not None]
            mem=kv('/proc/meminfo');vm=kv('/proc/vmstat')
            resident=None
            if (root/'resident.ready').exists():resident=proc(int((root/'resident.ready').read_text()))
            row={'phase':name,'time':time.time(),'processes':processes,'sum_pss_kib':sum(p['pss_kib'] for p in processes),'sum_rss_kib':sum(p['rss_kib'] for p in processes),'sum_uss_kib':sum(p['uss_kib'] for p in processes),'mem_available_kib':mem['MemAvailable'],'swap_used_kib':mem['SwapTotal']-mem['SwapFree'],'pswpin_pages':vm['pswpin'],'pswpout_pages':vm['pswpout'],'vla_process':resident}
            rows.append(row)
            with (out/'samples.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            time.sleep(1)
    finally:
        for job in jobs:
            if job.poll() is None:os.killpg(job.pid,signal.SIGINT)
        for job in jobs:
            try:job.wait(timeout=8)
            except subprocess.TimeoutExpired:os.killpg(job.pid,signal.SIGTERM);job.wait(timeout=5)
        for f in logs:f.close()
    summary={'phase':name,'samples':len(rows),'commands':[[str(x) for x in c] for c in commands]}
    for key in ['sum_pss_kib','sum_rss_kib','sum_uss_kib','mem_available_kib','swap_used_kib']:
        summary[key]={'median':statistics.median(r[key] for r in rows),'min':min(r[key] for r in rows),'max':max(r[key] for r in rows)}
    summary['swapin_pages_delta']=rows[-1]['pswpin_pages']-rows[0]['pswpin_pages'];summary['swapout_pages_delta']=rows[-1]['pswpout_pages']-rows[0]['pswpout_pages']
    print(json.dumps(summary),flush=True);return summary
if __name__=='__main__':
    report=[]
    for name,commands in [
        ('baseline',[]),
        ('cpp_one',[[prefix/'lib/demo_nodes_cpp/talker']]),
        ('cpp_pair',[[prefix/'lib/demo_nodes_cpp/talker'],[prefix/'lib/demo_nodes_cpp/listener']]),
        ('cpp_composed_pair',[[prefix/'lib/composition/dlopen_composition',prefix/'lib/libtalker_component.so',prefix/'lib/liblistener_component.so']])]:
        report.append(phase(name,commands));(out/'summary.json').write_text(json.dumps(report,indent=2))
