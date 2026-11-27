"""Diagnostic-boundary tests. Backend replies are fixtures, not live Qwen evidence."""
import copy
import json
from pathlib import Path
import runpy

import pytest
import agent_core as ac
import request_planner as rp

ROOT = Path(__file__).resolve().parent
P = runpy.run_path(str(ROOT/'tools/probe_request_planner.py'))


def plan(mode='plan', tools=None, stations=None, dates=None):
    return dict(mode=mode,tools=tools or [],metrics=[],stations=stations or [],dates=dates or [],date_scope='days')


def transport(message, finish='stop'):
    return {'choices':[{'message':message,'finish_reason':finish}],
            'usage':{'prompt_tokens':100,'completion_tokens':80,'total_tokens':180}}


def native(value):
    return {'content':'','tool_calls':[{'function':{'name':'plan_request','arguments':json.dumps(value)}}]}


def record(message, finish='stop'):
    return {'request':{'messages':[{'role':'system','content':rp.SYSTEM},
              {'role':'user','content':json.dumps({'request':'Предложи план.','session':''})}]},
            'response':transport(message,finish)}


@pytest.mark.parametrize('message',[
    native(plan()),
    {'content':'<tool_call>'+json.dumps({'name':'plan_request','arguments':plan()})+'</tool_call>'},
    {'content':json.dumps({'name':'plan_request','arguments':plan()})},
])
def test_existing_accepted_transports_are_reported_without_repair(message):
    row=record(message)
    before=copy.deepcopy(row)
    result=P['inspect_call'](row,ac,rp)
    assert result['stage']=='accepted'
    assert result['finish_reason']=='stop'
    assert result['usage']['completion_tokens']==80
    assert row==before


@pytest.mark.parametrize('message,finish',[
    ({'content':json.dumps(plan())},'stop'),
    ({'content':'Я предлагаю план исследования.'},'stop'),
    ({'content':'<tool_call>{"name":"plan_request","arguments":{'},'length'),
    ({'content':'','tool_calls':[]},'stop'),
    ({'content':'','tool_calls':[{'function':{'name':'fetch_observatory_data','arguments':'{}'}}]},'stop'),
])
def test_protocol_failures_retain_actual_text_and_finish_reason(message,finish):
    result=P['inspect_call'](record(message,finish),ac,rp)
    assert result['stage']=='parse'
    assert result['finish_reason']==finish
    assert result['message']


@pytest.mark.parametrize('mutation,expected',[
    ({'mode':'unknown'},'invalid request mode/scope'),
    ({'tools':['run_python']},'unregistered capability'),
    ({'tools':['get_statistics']},'nonexecuting mode cannot request tools'),
    ({'dates':['1900-01-01']},'invented date'),
    ({'tools':None},'invalid tools'),
])
def test_exact_validation_reason_is_visible(mutation,expected):
    value=plan();value.update(mutation)
    result=P['inspect_call'](record(native(value)),ac,rp)
    assert result['stage']=='validation'
    assert result['type']=='ValueError'
    assert result['message']==expected


def test_backend_failure_is_not_misreported_as_plan_validation():
    result=P['inspect_call']({'backend_exception':{'type':'RuntimeError','message':'backend failed'}},ac,rp)
    assert result=={'stage':'backend','type':'RuntimeError','message':'backend failed'}


def test_adapter_forwards_every_parameter_and_reuses_backend():
    class Backend:
        def create_chat_completion(self,**kwargs):
            self.request=kwargs
            return transport(native(plan()))
    backend=Backend();original=ac.LlamaBrain(backend,max_tokens=123,temperature=0.07,n_ctx=8192)
    adapter,recorder=P['_adapter'](original,ac)
    messages=[{'role':'user','content':'request'}]
    reply=adapter.chat(messages,[rp.PLAN_REQUEST_TOOL],max_tokens=360)
    assert original._llm is backend
    assert backend.request==dict(messages=messages,tools=[rp.PLAN_REQUEST_TOOL],temperature=0.07,max_tokens=360)
    assert original._max_tokens==123 and original._n_ctx==8192
    assert recorder.calls[0]['response']['usage']['completion_tokens']==80
    assert reply['tool_calls'][0]['function']['name']=='plan_request'


@pytest.mark.parametrize('preview',[False,True])
def test_probe_records_three_cases_without_scientific_execution(tmp_path,monkeypatch,preview):
    import shutil
    repo=tmp_path/'fixture_repo';(repo/'tools').mkdir(parents=True)
    for name in ['agent_core.py','request_planner.py','chat_session.py','gradio_app.py']:
        (repo/name).write_text('fixture source',encoding='utf-8')
    for name in ['tools/run_chat_acceptance.py','docs/qa/SCIENTIFIC_ADVISOR_CASES.json','docs/evidence/scientific_advisor_before.txt']:
        dest=repo/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    monkeypatch.setattr(ac,'__file__',str(repo/'agent_core.py'))
    monkeypatch.setenv('GEOMAG_REQUEST_PLANNER','1')
    monkeypatch.setenv('SUPERMAG_LOGON','fixture-private-logon')
    for name in ac.HANDLERS:
        monkeypatch.setitem(ac.HANDLERS,name,lambda *a,**k:pytest.fail('Scientific handler executed'))
    values=[plan(stations=['IRT','BOU'],dates=['2024-09-10','2024-09-12']),
            plan('analyze',['calculate_baseline'],['IRT'],['2024-09-12']),plan('explain')]
    class Backend:
        def __init__(self):self.calls=0
        def create_chat_completion(self,**kwargs):
            value=values[self.calls];self.calls+=1
            if 'response_format' in kwargs:
                wire={k:v for k,v in value.items() if k not in ('mode','tools','metrics')}
                wire['work']=[{'id':name,'required':value['mode'] in ('execute','analyze')} for name in value['tools']+value['metrics']]
                wire.update(action='compute' if value['mode'] in ('execute','analyze') else 'discuss',
                            answer='plan' if value['mode']=='plan' else 'interpretation')
                response=transport({'content':json.dumps(wire)})
            else:
                response=transport(native(value))
            response['model']='fixture-private-logon'
            return response
    backend=Backend();brain=ac.LlamaBrain(backend)
    if preview:
        from types import SimpleNamespace
        import sys
        installed = sys.modules['request_planner']
        component = SimpleNamespace(**runpy.run_path(str(ROOT/'request_planner.py')))
        monkeypatch.setattr(ac,'understand_request',lambda *a,**k:pytest.fail('Installed router used by preview'))
        result=P['_run'](brain,scope='TEST_FIXTURE: isolated new component; scripted backend',planner=component)
        assert sys.modules['request_planner'] is installed
    else:
        result=P['_run'](brain,scope='TEST_FIXTURE: scripted backend, not live Qwen')
    assert backend.calls==3 and brain._llm is backend
    report=json.loads((Path(result['run_dir'])/'report.json').read_text(encoding='utf-8'))
    assert report['status']=='RECORDED' and report['scope'].startswith('TEST_FIXTURE')
    assert [r['production_decision']['status'] for r in report['cases']]==['ok']*3
    assert all(r['inspection'][0]['stage']=='accepted' for r in report['cases'])
    assert all(r['routing_check']['passed'] for r in report['cases'])
    assert report['changed_source_files']==[] and report['original_backend_unchanged']
    assert report['scientific_acceptance']=='NOT_EVALUATED'
    assert 'fixture-private-logon' not in (Path(result['run_dir'])/'report.json').read_text(encoding='utf-8')
    assert Path(result['archive']).is_file()
    assert not list((repo/'verification').rglob('*.csv'))


@pytest.mark.parametrize('key,expected_pass', [('plan',True),('baselines',False),('critique',True)])
def test_recorded_v2_protocol_success_is_not_automatically_semantic_success(key,expected_pass):
    # Actual Qwen replies, not generated answers for a live-model PASS.
    recorded=json.loads((ROOT/'tests/fixtures/request_planner_v2_replies.json').read_text(encoding='utf-8'))[key]
    value=json.loads(recorded['response']['choices'][0]['message']['content'])
    decision={'status':'ok',**rp.validate(rp.decode_arguments(value),recorded['prompt'],'')}
    check=P['check_route'](key,decision,[{'stage':'accepted'}],1)
    assert check['passed'] is expected_pass
    if key=='baselines':
        assert decision['mode']=='analyze' and 'calculate_baseline' in decision['tools']
        assert decision['metrics']==['eej_index']  # Do not silently fix recorded evidence.
        assert any('eej_index' in error for error in check['errors'])


def test_baseline_route_check_allows_relevant_helpers_without_requiring_a_proxy():
    decision={'status':'ok',**plan('analyze',['calculate_baseline','calculate_derived_math','get_statistics'],
                                  ['irt'],['2024-09-12'])}
    assert P['check_route']('baselines',decision,[{'stage':'accepted'}],1)['passed']
    decision['tools'].append('plot_components')
    assert not P['check_route']('baselines',decision,[{'stage':'accepted'}],1)['passed']
