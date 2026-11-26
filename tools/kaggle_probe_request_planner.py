# Короткая диагностика; запускать с уже загруженным brain.
import hashlib
from urllib.request import urlopen

assert 'brain' in globals(), 'Сначала загрузи brain обычной ячейкой загрузки модели.'
probe_url = 'https://raw.githubusercontent.com/IgorCim/geomagnetic-agent-dev/25723cb79cdc0279886b3889f51bdbd7b3d9b1c1/tools/probe_request_planner.py'
with urlopen(probe_url, timeout=30) as response:
    probe_code = response.read()
assert hashlib.sha256(probe_code).hexdigest() == '2c81ffbd24e8cdbdd9ea911f9a634178a9960082ceebe1bab601ee68d44ff7cb', 'SHA256 диагностики не совпал.'

probe_ns = {'__name__': 'request_planner_probe', '__file__': probe_url}
exec(compile(probe_code, probe_url, 'exec'), probe_ns)
for name in ('demo', 'qa_download'):
    ui = globals().get(name)
    if ui is not None:
        ui.close()
probe_result = probe_ns['run'](brain)
print('ZIP диагностики:', probe_result['archive'], flush=True)
