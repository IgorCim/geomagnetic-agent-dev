"""Replay the original advisor prompts, preserving wording and recording real results.

No model download, automatic prompt repair, or automatic scientific PASS verdict.
Use the existing real brain from a fresh dev checkout in Colab/Kaggle.
"""
from pathlib import Path
import hashlib
import json
import runpy

ROOT = Path(__file__).resolve().parents[1]
CASE_FILE = 'docs/qa/SCIENTIFIC_ADVISOR_CASES.json'


def build_plan(root=ROOT):
    root = Path(root)
    catalog = json.loads((root / CASE_FILE).read_text(encoding='utf-8'))
    source = (root / catalog['source_file']).read_text(encoding='utf-8')
    assert hashlib.sha256(source.encode('utf-8')).hexdigest() == catalog['source_sha256_normalized_utf8'], 'Historical transcript changed'
    prompts = catalog['prompts']
    assert all(prompt in source for prompt in prompts.values()), 'Original wording changed'
    phases = [
        ('01_research_plan', ['plan']),
        ('02_days', ['days']),
        ('03_outliers', ['outliers']),
        ('04_stations', ['stations']),
        ('05_hypothesis_fresh', ['hypothesis']),
        ('06_baselines_fresh', ['baselines']),
        ('07_baseline_context', ['baselines', 'local_night']),
        ('08_critique', ['critique']),
        ('09_hypothesis_context', ['hypothesis', 'overclaim', 'strongest']),
    ]
    number = 0
    plan = []
    for phase, keys in phases:
        cards = []
        for key in keys:
            number += 1
            cards.append({'id':f'ADVISOR-{number:02d}', 'prompt':prompts[key], 'case':key})
        plan.append({'id':phase, 'cases':cards})
    assert number == 12
    return plan


def run(brain):
    import request_planner as rp
    assert rp.enabled(brain), 'Нужен новый LlamaBrain с включённым мини-планировщиком. Перезапусти сеанс с dev-веткой; загруженный код не подменяй.'
    helper = runpy.run_path(str(ROOT / 'tools/run_chat_acceptance.py'))
    plan = build_plan()
    print('Проверка понимания: 12 исходных сообщений, 9 независимых чатов.', flush=True)
    print('Два повтора сохранены; продолжения идут после исходной задачи. Научные ответы требуют оценки.', flush=True)
    return helper['run_acceptance'](brain, plan=plan)


if __name__ == '__main__':
    print(json.dumps([{'chat':p['id'], 'messages':len(p['cases'])} for p in build_plan()], ensure_ascii=False, indent=2))
