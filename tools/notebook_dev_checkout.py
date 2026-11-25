# Ячейка 1 для Kaggle / Colab: новая dev-ветка, без удаления старой работы.
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import tempfile

PROJECT_MODULES = {
    'agent_core', 'agent_planner', 'agent_state', 'project_store', 'gradio_app',
    'supermag_loader', 'iswa_loader', 'intermagnet_loader', 'space_weather_loader',
    'noaa_swpc_loader', 'geomag_analyzer', 'geomag_math', 'geomag_plotter',
    'geomag_coords', 'physics_engine', 'chart_naming', 'chat_session', 'request_planner',
}
loaded = sorted(PROJECT_MODULES.intersection(sys.modules))
assert not loaded and 'brain' not in globals(), (
    'Сначала перезапусти kernel. Файлы и веса удалять не нужно. '
    'Уже загруженные модули: ' + ', '.join(loaded)
)

REPO_URL = 'https://github.com/IgorCim/geomagnetic-agent-dev.git'
BRANCH = 'codex/semantic-request-planner'
MIN_REVISION = '8c67cc0982880c94be591bdfb17971949a2bedf0'
WORKING = next((p for p in (Path('/kaggle/working'), Path('/content')) if p.is_dir()), None)
assert WORKING is not None, 'Эта ячейка предназначена для Kaggle или Colab.'
DIR = Path(tempfile.mkdtemp(prefix='geomagnetic_dev_', dir=WORKING))
cloned = subprocess.run(
    ['git', 'clone', '--branch', BRANCH, '--single-branch', REPO_URL, str(DIR)],
    capture_output=True, text=True, timeout=300,
)
if cloned.returncode:
    raise RuntimeError('Не удалось скачать dev-ветку. Проверь доступ к GitHub.\n' + cloned.stderr)


def checkout_git(*args):
    return subprocess.check_output(['git', '-C', str(DIR), *args], text=True).strip()


HEAD = checkout_git('rev-parse', 'HEAD')
checkout_git('merge-base', '--is-ancestor', MIN_REVISION, HEAD)
checkout_git('checkout', '--detach', HEAD)
required = [
    'agent_core.py', 'request_planner.py', 'test_request_planner.py',
    'gradio_app.py', 'requirements-dev.txt', 'stress_300.py',
    'tools/run_request_planner_review.py', 'tools/run_chat_acceptance.py',
    'tools/run_full_evidence.py', 'tools/verify_offline.py', 'tools/verify_gradio.py',
    'docs/qa/SCIENTIFIC_ADVISOR_CASES.json',
    'docs/evidence/scientific_advisor_before.txt', 'docs/acceptance_protocol.json',
]
missing = [name for name in required if not (DIR / name).is_file()]
assert not missing, f'Неполная dev-поставка: {missing}; HEAD={HEAD}'
protocol = json.loads((DIR / 'docs/acceptance_protocol.json').read_text(encoding='utf-8'))
assert protocol.get('version') == 'github-20261102-v1', 'Другая версия установочного протокола.'
assert not checkout_git('status', '--porcelain'), 'Новый checkout неожиданно содержит изменения.'

tracked = subprocess.check_output(['git', '-C', str(DIR), 'ls-files', '-z']).decode('utf-8').split('\0')
hashes = {}
for name in filter(None, tracked):
    path = DIR / name
    assert path.is_file() and path.resolve().is_relative_to(DIR.resolve()), name
    hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
source = {
    'source': 'github', 'git_revision': HEAD, 'git_branch': BRANCH,
    'protocol_version': protocol['version'], 'dirty_tree': False, 'sha256': hashes,
}
(DIR / 'SOURCE_MANIFEST.json').write_text(json.dumps(source, ensure_ascii=False, indent=2), encoding='utf-8')

stem = 'qwen2.5-14b-instruct-q4_k_m'
model_candidates = []
if os.environ.get('GEOMAG_MODELS_DIR'):
    model_candidates.append(Path(os.environ['GEOMAG_MODELS_DIR']))
model_candidates.extend([
    WORKING / 'geomagnetic-agent/models', WORKING / 'geomagnetic_agent/models',
    WORKING / 'geomagnetic-models',
])
MODELS = next((p.resolve() for p in model_candidates if p.is_dir() and any(p.glob(stem + '*.gguf'))),
              WORKING / 'geomagnetic-models')
MODELS.mkdir(parents=True, exist_ok=True)
os.environ['GEOMAG_MODELS_DIR'] = str(MODELS)
os.environ['GEOMAG_REQUEST_PLANNER'] = '1'
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
os.environ['GEOMAG_OFFLINE'] = '0'
os.environ['GEOMAG_DEBUG_PARSER'] = '0'
os.environ['GEOMAG_DEBUG_SCRATCHPAD'] = '0'
os.chdir(DIR)
sys.path.insert(0, str(DIR))
print('DEV SOURCE VERIFIED:', len(hashes), 'tracked files', flush=True)
print('Репозиторий:', REPO_URL, flush=True)
print('Ветка:', BRANCH, flush=True)
print('HEAD:', HEAD, flush=True)
print('DIR:', DIR, flush=True)
print('MODELS:', MODELS, flush=True)
print('Проверка 12 сообщений: НАЙДЕНА', flush=True)
print('Теперь выполняй обычные ячейки 2–5, затем ячейку научной проверки. Старую ячейку 1 не запускай.', flush=True)
