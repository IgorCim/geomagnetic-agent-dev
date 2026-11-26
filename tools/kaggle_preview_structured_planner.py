# Проверка нового компонента на существующем brain; это не установочная ячейка.
import hashlib
from types import SimpleNamespace
from urllib.request import urlopen

assert 'brain' in globals(), 'Сначала загрузи brain обычной ячейкой загрузки модели.'
preview_base = 'https://raw.githubusercontent.com/IgorCim/geomagnetic-agent-dev/ad624a0cc0f816039e7c192d3bec200166bcf0e0/'


def load_preview_component(filename, expected_sha256):
    url = preview_base + filename
    with urlopen(url, timeout=30) as response:
        code = response.read()
    assert hashlib.sha256(code).hexdigest() == expected_sha256, 'SHA256 не совпал: ' + filename
    namespace = {'__name__': 'isolated_planner_preview', '__file__': url,
                 '__source_sha256__': expected_sha256}
    exec(compile(code, url, 'exec'), namespace)
    return SimpleNamespace(**namespace)


new_planner = load_preview_component('request_planner.py', 'b347f37271291cb1d0145feaafe4860c1929a7b8389cbc24b145f00de59ed971')
preview_probe = load_preview_component('tools/probe_request_planner.py', 'b9666eb5d2dbfb9153708de3ad281ca446681214683befd53284179423fdf198')
for name in ('demo', 'qa_download'):
    ui = globals().get(name)
    if ui is not None:
        ui.close()
print('Проверяю новый компонент; установленный агент и brain не заменяются.', flush=True)
preview_result = preview_probe.preview(brain, new_planner)
checks = [row.get('routing_check', {}).get('passed', False) for row in preview_result['report']['cases']]
protocol = [row.get('production_decision', {}).get('status') == 'ok'
            and len(row.get('inspection', [])) == 1
            and row['inspection'][0].get('stage') == 'accepted'
            for row in preview_result['report']['cases']]
print('JSON принят:', sum(protocol), '/', len(protocol), flush=True)
print('Смысловая проверка маршрутов:', sum(checks), '/', len(checks), '(научные ответы ещё не проверялись)', flush=True)
print('ZIP проверки:', preview_result['archive'], flush=True)
