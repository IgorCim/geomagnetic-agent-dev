# Проверка нового компонента на существующем brain; это не установочная ячейка.
import hashlib
from types import SimpleNamespace
from urllib.request import urlopen

assert 'brain' in globals(), 'Сначала загрузи brain обычной ячейкой загрузки модели.'
preview_base = 'https://raw.githubusercontent.com/IgorCim/geomagnetic-agent-dev/25723cb79cdc0279886b3889f51bdbd7b3d9b1c1/'


def load_preview_component(filename, expected_sha256):
    url = preview_base + filename
    with urlopen(url, timeout=30) as response:
        code = response.read()
    assert hashlib.sha256(code).hexdigest() == expected_sha256, 'SHA256 не совпал: ' + filename
    namespace = {'__name__': 'isolated_planner_preview', '__file__': url,
                 '__source_sha256__': expected_sha256}
    exec(compile(code, url, 'exec'), namespace)
    return SimpleNamespace(**namespace)


new_planner = load_preview_component('request_planner.py', 'ede95a1c1919d3905164548a1612a64c0f913b8b134d42967f274ad90d3a7b09')
preview_probe = load_preview_component('tools/probe_request_planner.py', '2c81ffbd24e8cdbdd9ea911f9a634178a9960082ceebe1bab601ee68d44ff7cb')
for name in ('demo', 'qa_download'):
    ui = globals().get(name)
    if ui is not None:
        ui.close()
print('Проверяю новый компонент; установленный агент и brain не заменяются.', flush=True)
preview_result = preview_probe.preview(brain, new_planner)
checks = [row.get('routing_check', {}).get('passed', False) for row in preview_result['report']['cases']]
print('Принято маршрутов:', sum(checks), '/', len(checks), '(это ещё не научная приёмка)', flush=True)
print('ZIP проверки:', preview_result['archive'], flush=True)
