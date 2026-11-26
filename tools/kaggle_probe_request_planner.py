# Короткая диагностика; запускать с уже загруженным brain.
import hashlib
from urllib.request import urlopen

assert 'brain' in globals(), 'Сначала загрузи brain обычной ячейкой загрузки модели.'
probe_url = 'https://raw.githubusercontent.com/IgorCim/geomagnetic-agent-dev/3138ef07abaabf21bac7b5ab3ec7ce10c2c0c107/tools/probe_request_planner.py'
with urlopen(probe_url, timeout=30) as response:
    probe_code = response.read()
assert hashlib.sha256(probe_code).hexdigest() == '2f006dcf0b08ae460fa41e79445e894ababadd60ad1bf5c0c51dc9c3b4b13a11', 'SHA256 диагностики не совпал.'

probe_ns = {'__name__': 'request_planner_probe', '__file__': probe_url}
exec(compile(probe_code, probe_url, 'exec'), probe_ns)
for name in ('demo', 'qa_download'):
    ui = globals().get(name)
    if ui is not None:
        ui.close()
probe_result = probe_ns['run'](brain)
print('ZIP диагностики:', probe_result['archive'], flush=True)
