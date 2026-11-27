"""Record both criteria: new conversational controls + unchanged 12 advisor prompts.

Operator checks are not a scientific PASS. Final reasoning remains REVIEW_REQUIRED.
"""
from pathlib import Path
import json
import runpy

ROOT=Path(__file__).resolve().parents[1]


def build_plan(root=ROOT):
    original=runpy.run_path(str(Path(root)/'tools/run_request_planner_review.py'))['build_plan'](root)
    controls=[
      {'id':'language_reuse','cases':[
        {'id':'LANG-01','prompt':'Возьми минутные измерения Иркутска за 10 сентября 2024 года. Мне нужны минимум, максимум и медиана H, плюс один график H.'},
        {'id':'LANG-02','prompt':'Данные оставь те же. Теперь нужны минимум, максимум и медиана Y. Повторно ничего не скачивай; новый график не нужен.'}]},
      {'id':'language_two_stations','cases':[
        {'id':'LANG-03','prompt':'Хочу сопоставить H в IRT и BOU за 10 сентября 2024 года. Дай среднее H и число пропусков отдельно для каждой станции. Сохрани принадлежность чисел станции и дате.'}]},
      {'id':'science_units_holdout','cases':[
        {'id':'SCI-X01','prompt':'Разбери рассуждение коллеги: «Средняя H в одном городе больше, значит буря там сильнее. Скорость изменения H равна 30 нТ/мин, значит по проводам течёт ток 30 ампер». Какие выводы допустимы и какой информации не хватает? Это разбор текста; новые наблюдения не загружай.'}]}]
    return controls+original


def assess_operators(turn):
    """Checks explicit control outputs, not general language/science correctness."""
    engines=turn.get('engine_results',[])
    results=[e.get('result',{}) for e in engines]
    calls=[c for r in results for c in r.get('tool_calls',[]) if c.get('ok')]
    key=turn['id'];errors=[]
    if key in ('LANG-01','LANG-02','LANG-03'):
        component='Y' if key=='LANG-02' else 'H'
        expected={'irt','bou'} if key=='LANG-03' else {'irt'}
        observed=set()
        for call in calls:
            stats=(call.get('statistics') or {}).get(component)
            if not stats or not {'min','max','median','mean','missing'}<=set(stats):continue
            for source in call.get('input_provenance',[]):
                parts=source.get('handle','').split(':')
                if len(parts)>=3 and parts[2]=='2024-09-10' and parts[1] in expected:
                    observed.add(parts[1])
        if observed!=expected:errors.append('missing requested statistics/source identities')
        if key=='LANG-01' and not any(c.get('plot') for c in calls):errors.append('missing requested H plot')
        if key=='LANG-02':
            forbidden={'fetch_observatory_data','fetch_solar_wind','fetch_geomagnetic_indices','fetch_supermag_data'}
            executed=[c for r in results for c in r.get('tool_calls',[]) if c.get('executed') is not False]
            if any(c.get('tool') in forbidden for c in executed):errors.append('download attempted despite reuse constraint')
            if any(c.get('plot') for c in calls):errors.append('unrequested new plot')
    elif key=='SCI-X01':
        executed=[c for r in results for c in r.get('tool_calls',[]) if c.get('executed') is not False]
        if executed:errors.append('text review executed scientific tools')
        if not turn.get('answer','').strip():errors.append('missing review text')
    else:return {'status':'REVIEW_REQUIRED','scope':'Original scientific task; evaluate final answer and evidence manually.'}
    if not results:errors.append('no engine result recorded')
    return {'status':'FAIL' if errors else 'PASS','errors':errors,
            'scope':'Operator checks only; final claims and scientific reasoning require review.'}


def run(brain):
    import agent_core as ac
    import request_planner as rp
    import research_context
    if not isinstance(brain,ac.LlamaBrain) or not rp.enabled(brain):
        raise TypeError('Нужен настоящий brain из свежей dev-ветки.')
    if 'required' not in rp.ARGUMENT_SCHEMA['required']:
        raise RuntimeError('Старая версия планировщика. Нужен свежий kernel и dev-checkout; загруженные модули не заменяй.')
    helper=runpy.run_path(str(ROOT/'tools/run_chat_acceptance.py'))
    print('Два критерия: 4 новых контрольных сообщения + 12 исходных научных сообщений.',flush=True)
    print('Контроль действий и научная оценка учитываются отдельно; запись не означает PASS.',flush=True)
    result=helper['run_acceptance'](brain,plan=build_plan())
    audit=[]
    for phase in result['report']['phases']:
        for turn in phase['turns']:
            audit.append({'id':turn['id'],'operators':assess_operators(turn),'science':'REVIEW_REQUIRED'})
    root=Path(result['run_dir'])
    (root/'dual_review.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    import zipfile
    with zipfile.ZipFile(result['archive'],'a',zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(root/'dual_review.json','dual_review.json')
    controls=[row for row in audit if row['operators']['status']!='REVIEW_REQUIRED']
    print('Контроль действий:',sum(row['operators']['status']=='PASS' for row in controls),'/',len(controls),flush=True)
    print('Научные выводы: REVIEW_REQUIRED. Присылай полный лог и ZIP.',flush=True)
    return result
