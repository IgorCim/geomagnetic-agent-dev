"""QA operator checks cannot certify scientific reasoning or hide missing sources."""
import runpy
from pathlib import Path
import request_planner as rp

ROOT=Path(__file__).resolve().parent
DUAL=runpy.run_path(str(ROOT/'tools/run_dual_stage_review.py'))


def test_dual_plan_keeps_all_original_wording_and_adds_natural_language_controls():
    original=runpy.run_path(str(ROOT/'tools/run_request_planner_review.py'))['build_plan'](ROOT)
    combined=DUAL['build_plan'](ROOT)
    assert combined[3:]==original
    assert len(combined)==12 and sum(len(p['cases']) for p in combined)==16
    prompts=[c['prompt'] for p in combined[:3] for c in p['cases']]
    assert len(prompts)==4
    assert not any(tool in prompt for tool in rp.CAPABILITIES for prompt in prompts)


def turn(key,stations):
    calls=[{'ok':True,'tool':'get_statistics','statistics':{'H':{'min':1,'max':2,'median':1.5,'mean':1.5,'missing':0}},
            'input_provenance':[{'handle':f'raw:{s}:2024-09-10','rows':1440}]} for s in stations]
    return {'id':key,'answer':'TEST FIXTURE','engine_results':[{'result':{'tool_calls':calls}}]}


def test_comparison_operator_check_requires_both_source_identities():
    assert DUAL['assess_operators'](turn('LANG-03',['irt','bou']))['status']=='PASS'
    assert DUAL['assess_operators'](turn('LANG-03',['irt']))['status']=='FAIL'
    other=turn('LANG-03',['irt','bou'])
    other['engine_results'][0]['result']['tool_calls'][1]['input_provenance'][0]['handle']='raw:bou:2024-09-12'
    assert DUAL['assess_operators'](other)['status']=='FAIL'


def test_no_fetch_check_uses_actual_execution_not_a_recovered_rejected_attempt():
    item=turn('LANG-02',['irt'])
    item['engine_results'][0]['result']['tool_calls'][0]['statistics']={'Y':{'min':1,'max':2,'median':1.5,'mean':1.5,'missing':0}}
    denied={'ok':False,'tool':'fetch_observatory_data','executed':False}
    item['engine_results'][0]['result']['tool_calls'].append(denied)
    assert DUAL['assess_operators'](item)['status']=='PASS'
    denied['executed']=True
    assert DUAL['assess_operators'](item)['status']=='FAIL'


def test_operator_success_is_not_a_scientific_verdict():
    check=DUAL['assess_operators'](turn('SCI-X01',[]))
    assert check['status']=='PASS' and 'require review' in check['scope']
    assert DUAL['assess_operators'](turn('ADVISOR-12',[]))['status']=='REVIEW_REQUIRED'
