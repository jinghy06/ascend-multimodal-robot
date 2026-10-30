import urllib.request,json,pathlib,concurrent.futures,time
out=pathlib.Path(__file__).parent/'downloads/dataset_wheels';out.mkdir(parents=True,exist_ok=True)
versions={'datasets':'3.6.0','av':'14.2.0','pyarrow':'20.0.0','pandas':'2.2.3','dill':'0.3.8','multiprocess':'0.70.16','fsspec':'2025.3.0','aiohttp':'3.11.18','aiohappyeyeballs':'2.6.1','aiosignal':'1.3.2','frozenlist':'1.6.0','multidict':'6.4.3','propcache':'0.3.1','yarl':'1.20.0','python-dateutil':'2.9.0.post0','pytz':'2025.2','tzdata':'2025.2','xxhash':'3.5.0'}
def get(item):
 name,version=item
 data=json.load(urllib.request.urlopen(f'https://pypi.org/pypi/{name}/{version}/json',timeout=25))
 urls=[x for x in data['urls'] if x['filename'].endswith('.whl') and (('aarch64' in x['filename'] and 'cp310-cp310' in x['filename'] and 'manylinux' in x['filename']) or x['filename'].endswith('py3-none-any.whl') or x['filename'].endswith('py2.py3-none-any.whl'))]
 if not urls:raise RuntimeError((name,[x['filename'] for x in data['urls']]))
 x=urls[0];p=out/x['filename']
 if not p.exists() or p.stat().st_size!=x['size']:urllib.request.urlretrieve(x['url'],p)
 print(p.name,p.stat().st_size,flush=True)
with concurrent.futures.ThreadPoolExecutor(6) as pool:list(pool.map(get,versions.items()))
