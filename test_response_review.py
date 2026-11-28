"""Mechanism regressions from the real 5e46603 log; no simulated Qwen PASS."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import agent_core as ac
import project_store as ps
import request_planner as rp
import response_review as rr
from chat_session import Request, component_identity
from test_request_planner import native, seed
from test_structured_request_planner import wire

ROOT=Path(__file__).resolve().parent
RECORDED=json.loads((ROOT/'tests/fixtures/dual_stage_5e46603.json').read_text(encoding='utf-8'))['cases']


def reviewed(style='interpretation',text='Вывод ограничен измеренным рядом; причина не установлена.',complete=True,not_needed=None,evidence=None,remaining=None):
    return dict(style=style,complete=complete,not_needed=not_needed or [],evidence=evidence or [],text=text,remaining=remaining or [])


class ReviewedBackend:
    def __init__(self,plan,*replies,review=None):
        self.plan=plan;self.replies=iter(replies);self.review=review or reviewed();self.calls=[]
    def create_chat_completion(self,**kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        props=(kwargs.get('response_format') or {}).get('schema',{}).get('properties',{})
        if 'work' in props:value=self.plan
        elif 'style' in props:value=self.review(kwargs) if callable(self.review) else self.review
        else:return next(self.replies,{'choices':[{'message':{'content':'Черновик: 999 nT, всё выполнено.'}}]})
        return {'choices':[{'finish_reason':'stop','message':{'content':json.dumps(value,ensure_ascii=False)}}]}


def context(tmp_path):
    return ac.FrameStore(workspace=ps.Workspace(root=tmp_path/'projects'))


def call(name,args):return {'choices':[{'message':native(name,args)}]}


def baseline_request(tmp_path):
    store=context(tmp_path);h=seed(store,day='2024-09-12')
    store.get(h)['X']=[3.1,5.9,8.7,10.4];store.get(h)['Y']=0.
    request=Request('Получить baseline и anomaly.',calculation_tools=('calculate_baseline',))
    args={'df_descriptor':h,'component':'H','mode':'full'}
    payload,_=ac.HANDLERS['calculate_baseline'](args,store)
    request.observe_result('calculate_baseline',args,payload,store)
    return store,h,request,payload['baseline']


def test_exact_recorded_reference_is_accepted_for_the_same_component_in_derived(tmp_path):
    store,h,request,baseline=baseline_request(tmp_path)
    derived,_=ac.HANDLERS['calculate_derived_components']({'df_descriptor':h,'components':['H']},store)
    args=dict(df_descriptor=derived['frame_handle'],component='H',metric='anomaly',baseline_value=baseline)
    assert component_identity(store,h,'H')==component_identity(store,derived['frame_handle'],'H')
    assert request.check_call('calculate_derived_math',args,store) is None
    actual,_=ac.HANDLERS['calculate_derived_math'](args,store)
    assert not ac.is_error(actual)
    np.testing.assert_allclose(store.get(actual['frame_handle'])['ANOMALY'],np.array([3.1,5.9,8.7,10.4])-baseline)


@pytest.mark.parametrize('change',['station','date','component','values','times','reused_source','invented_scalar'])
def test_equivalent_reference_check_does_not_allow_a_different_input(tmp_path,change):
    store,h,request,baseline=baseline_request(tmp_path)
    result,_=ac.HANDLERS['calculate_derived_components']({'df_descriptor':h,'components':['H']},store)
    target=result['frame_handle'];frame=store.get(target).copy()
    component='H';value=baseline
    if change=='station':target='derived:bou:2024-09-12'
    elif change=='date':target='derived:irt:2024-09-11'
    elif change=='component':component='Z'
    elif change=='values':frame['H']+=1
    elif change=='times':frame['timestamp']+=pd.Timedelta(minutes=1)
    elif change=='reused_source':target=h;frame['X']+=1;frame=frame.drop(columns=['H'],errors='ignore')
    elif change=='invented_scalar':value=22.5
    store.put(target,frame)
    assert request.check_call('calculate_derived_math',dict(df_descriptor=target,component=component,metric='anomaly',baseline_value=value),store)


def test_night_and_full_reference_values_remain_independent(tmp_path):
    store,h,request,first=baseline_request(tmp_path)
    args=dict(df_descriptor=h,component='H',mode='night',night_hours=[0,2])
    result,_=ac.HANDLERS['calculate_baseline'](args,store);request.observe_result('calculate_baseline',args,result,store)
    second=result['baseline'];assert first!=second
    for value in [first,second]:
        assert request.check_call('calculate_derived_math',dict(df_descriptor=h,component='H',metric='anomaly',baseline_value=value),store) is None


def test_actual_live_plans_all_passed_format_but_selected_the_wrong_answer_kind():
    assert len(RECORDED)==16 and all(r['request_plan']['status']=='ok' for r in RECORDED)
    row=next(r for r in RECORDED if r['id']=='ADVISOR-05')
    assert row['request_plan']['mode']=='execute' and 'Покажи, какие' in row['prompt']
    assert 'Подтверждённые результаты:' in row['answer']
    row=next(r for r in RECORDED if r['id']=='LANG-03')
    assert 'fetch_supermag_data' in row['request_plan']['required_operations']
    assert 'Не выполнено: выполнить fetch_supermag_data' in row['answer']


def test_wrong_facts_classification_cannot_suppress_the_final_explanation(tmp_path):
    store=context(tmp_path);h=seed(store,day='2024-09-12')
    plan=wire('compute','facts',['calculate_baseline'],stations=['IRT'],dates=['2024-09-12'])
    backend=ReviewedBackend(plan,call('calculate_baseline',dict(df_descriptor=h,component='H',mode='full')),
                            review=reviewed(text='Опорный уровень 5 nT относится к указанному ряду. Спокойствие интервала не проверено.',evidence=[0]))
    brain=ac.LlamaBrain(backend)
    parameters=(brain._llm,brain._temperature,brain._max_tokens)
    result=ac.run_agent('Получи опорный уровень H IRT за 12 сентября 2024 и объясни ограничения.',brain=brain,store=store,verbose=False)
    assert result['ok'],result
    assert result['text'].startswith('Опорный уровень 5') and 'Спокойствие' in result['text']
    assert '999' not in result['text'] and '999' in result['draft_text']
    review_calls=[c for c in backend.calls if 'style' in c.get('response_format',{}).get('schema',{}).get('properties',{})]
    assert len(review_calls)==1
    request=review_calls[0]
    assert 'tools' not in request and request['temperature']==0
    assert request['max_tokens']==rr.MAX_OUTPUT_TOKENS
    assert '999' not in request['messages'][1]['content']
    assert result['response_review']['input_chars_with_schema']<=rr.MAX_INPUT_CHARS
    assert (brain._llm,brain._temperature,brain._max_tokens)==parameters


def test_alternative_real_station_statistics_satisfy_task_without_unasked_supermag(tmp_path,monkeypatch):
    monkeypatch.delenv('SUPERMAG_LOGON',raising=False)
    store=context(tmp_path);a=seed(store);b=seed(store,station='bou')
    store.get(b)['X']=6.;store.get(b)['Y']=8.
    row=next(r for r in RECORDED if r['id']=='LANG-03')
    intent=copy.deepcopy(row['request_plan'])
    value=reviewed('facts',not_needed=['fetch_supermag_data','calculate_baseline'],evidence=[0,1])
    backend=ReviewedBackend(None,call('get_statistics',dict(df_descriptor=a,components=['H'])),call('get_statistics',dict(df_descriptor=b,components=['H'])),review=value)
    result=ac.run_agent(row['prompt'],brain=ac.LlamaBrain(backend),store=store,intent_plan=intent,verbose=False)
    assert result['ok'],result
    assert result['response_review']['not_needed']==['fetch_supermag_data','calculate_baseline']
    assert 'mean 5.0' in result['text'] or 'среднее: 5.0' in result['text']
    assert 'среднее: 10.0' in result['text'] and 'raw:irt:' in result['text'] and 'raw:bou:' in result['text']
    assert 'Запрос выполнен не полностью' not in result['text']
    assert [c['tool'] for c in result['tool_calls']]==['get_statistics','get_statistics']


@pytest.mark.parametrize('query,operation',[
    ('Вызови evaluate_custom_formula для H.','evaluate_custom_formula'),
    ('Загрузи данные именно SuperMAG.','fetch_supermag_data'),
    ('Получи INTERMAGNET IRT.','fetch_observatory_data'),
    ('Получи OMNI за день.','fetch_solar_wind'),
    ('Проверь Kyoto Dst.','fetch_geomagnetic_indices'),
])
def test_review_cannot_waive_an_explicit_tool_or_source(query,operation):
    contract=Request(query,required_operations=(operation,))
    assert rr.optional_methods(contract,[f'выполнить {operation}'])==[]


def test_required_physical_result_cannot_be_waived_as_an_optional_method():
    contract=Request('Вычисли GIC-прокси.',required_operations=('apply_physics_formula',),intent={'required_metrics':['gic']})
    assert rr.optional_methods(contract,['выполнить apply_physics_formula'])==[]


@pytest.mark.parametrize('bad',[
    {'not_needed':['get_statistics']}, {'not_needed':['run_python']},
    {'evidence':[99]}, {'evidence':[True]}, {'complete':'yes'},
    {'complete':False,'remaining':[]}, {'complete':True,'remaining':['нет результата']},
    {'style':'interpretation','text':''}, {'text':{}}, {'extra':'ignore'},
])
def test_invalid_review_cannot_become_a_successful_final_answer(tmp_path,bad):
    store=context(tmp_path);h=seed(store)
    value=reviewed();value.update(bad)
    plan=wire('compute','facts',['get_statistics'],stations=['IRT'],dates=['2024-09-10'])
    backend=ReviewedBackend(plan,call('get_statistics',dict(df_descriptor=h,components=['H'])),review=value)
    result=ac.run_agent('Получить статистику H IRT 2024-09-10.',brain=ac.LlamaBrain(backend),store=store,verbose=False)
    assert not result['ok'] and result['error']=='response_review_failed'
    assert '999' not in result['text']
    assert result['tool_calls'][0]['ok'] and 'statistics' in result['tool_calls'][0]


def test_reviewer_cannot_hide_a_missing_real_result(tmp_path):
    store=context(tmp_path);seed(store)
    plan=wire('compute','facts',['get_statistics'],stations=['IRT'],dates=['2024-09-10'])
    result=ac.run_agent('Получи статистику H IRT за 2024-09-10.',brain=ac.LlamaBrain(ReviewedBackend(plan,review=reviewed('facts'))),store=store,verbose=False)
    assert not result['ok'] and result['error']=='request_incomplete'
    assert 'get_statistics' in result['text']


def test_numeric_request_uses_canonical_values_even_if_review_text_is_wrong(tmp_path):
    store=context(tmp_path);h=seed(store)
    backend=ReviewedBackend(wire('compute','facts',['get_statistics'],stations=['IRT'],dates=['2024-09-10']),
            call('get_statistics',dict(df_descriptor=h,components=['H'])),review=reviewed('facts',text='H=999999 nT'))
    result=ac.run_agent('Получи статистику H IRT 2024-09-10.',brain=ac.LlamaBrain(backend),store=store,verbose=False)
    assert result['ok'] and '999999' not in result['text'] and 'медиана: 5.0' in result['text']


def test_scientific_text_review_is_fresh_and_has_no_execution_tools(tmp_path):
    store=context(tmp_path)
    query='Оцени вывод: отсутствие выбросов 3 сигма исключает бурю; абсолютное H доказывает силу бури; dH/dt даёт GIC в амперах; корреляция доказывает причину.'
    backend=ReviewedBackend(wire(),review=reviewed(text='Эти доказательства недостаточны: критерий выбросов не устанавливает наличие бури; сравнивать нужно возмущения с обоснованным фоном. Для токов нужна дополнительная модель; корреляция не доказывает причину.'))
    result=ac.run_agent(query,brain=ac.LlamaBrain(backend),store=store,verbose=False)
    assert result['ok'] and result['tool_calls']==[] and '999' not in result['text']
    payload=result['response_review']['input']
    assert payload['request']==query and not payload['records'] and not payload['available_tables']
    definitions=' '.join(payload['definitions_not_observations'])
    assert 'sigma outlier' in definitions and 'background field' in definitions
    assert 'amperes' in definitions and 'not unique causation' in definitions
    assert all(not c.get('tools') for c in backend.calls)


def test_inventory_keeps_kyoto_row_count_after_execution_evidence_is_trimmed(tmp_path):
    store=context(tmp_path);seed(store,day='2024-09-12',periods=1440,hourly=False)
    idx='idx:kyoto:2024-09-12';sw='sw:omni:2024-09-12'
    store.put(idx,pd.DataFrame({'timestamp':pd.date_range('2024-09-12',periods=24,freq='h',tz='UTC'),'Dst':np.arange(24)}))
    store.put(sw,pd.DataFrame({'timestamp':pd.date_range('2024-09-12',periods=1440,freq='min',tz='UTC'),'Bz':np.zeros(1440)}))
    store.conversation.evidence=[{'tool':'get_statistics','inputs':[{'handle':'raw:irt:2024-09-12','rows':1440}],
                                   'statistics':{'H':{'mean':i}},'padding':'x'*1000} for i in range(32)]
    backend=ReviewedBackend(None)
    result=rr.review(ac.LlamaBrain(backend),'Что позволяют уже полученные результаты?',store,[],{'mode':'explain','transport':'json_schema','tools':[]})
    assert result['status']=='ok',result
    payload=result['input'];tables={r['handle']:r for r in payload['available_tables']}
    assert tables[idx]['rows']==24 and tables[sw]['rows']==1440
    assert payload['omitted_records']>0 and result['input_chars_with_schema']<=rr.MAX_INPUT_CHARS


def test_review_input_and_raw_response_redact_logon(tmp_path,monkeypatch):
    monkeypatch.setenv('SUPERMAG_LOGON','private-test-logon')
    store=context(tmp_path);seed(store)
    value=reviewed(text='private-test-logon')
    result=rr.review(ac.LlamaBrain(ReviewedBackend(None,review=value)),'Объясни private-test-logon.',store,[],{'mode':'explain','transport':'json_schema','tools':[]})
    assert 'private-test-logon' not in json.dumps(result)


def test_optional_failed_plot_is_not_counted_as_a_requested_missing_artifact():
    log=[{'tool':'plot_comparison','ok':False,'required_by_plan':False,'arguments':{'filename':'optional.html'}}]
    assert ac._with_plot_tally('Два опорных уровня рассчитаны.',log)=='Два опорных уровня рассчитаны.'
    log[0]['required_by_plan']=True
    assert '0 из 1' in ac._with_plot_tally('Частично.',log)


def test_reference_does_not_move_to_new_content_after_same_slot_is_reused(tmp_path):
    store,h,request,old=baseline_request(tmp_path)
    store.get(h)['X']+=10
    args=dict(df_descriptor=h,component='H',mode='full')
    payload,_=ac.HANDLERS['calculate_baseline'](args,store)
    request.observe_result('calculate_baseline',args,payload,store)
    assert payload['baseline']!=old
    assert request.check_call('calculate_derived_math',dict(df_descriptor=h,component='H',metric='anomaly',baseline_value=old),store)
    assert request.check_call('calculate_derived_math',dict(df_descriptor=h,component='H',metric='anomaly',baseline_value=payload['baseline']),store) is None


def test_session_logon_availability_is_not_shared_or_sent_to_planner(tmp_path,monkeypatch):
    monkeypatch.delenv('SUPERMAG_LOGON',raising=False)
    a=context(tmp_path/'a');b=context(tmp_path/'b')
    a.supermag_logon='session-only-private-logon'
    for store,expected in [(a,True),(b,False)]:
        backend=ReviewedBackend(wire())
        result=ac.understand_request('Объясни ограничения SuperMAG.',ac.LlamaBrain(backend),store)
        assert result['status']=='ok'
        payload=json.loads(backend.calls[0]['messages'][1]['content'])
        assert payload['source_access']['supermag_logon_configured'] is expected
        assert 'session-only-private-logon' not in json.dumps(payload)
    value=reviewed(text='session-only-private-logon')
    result=rr.review(ac.LlamaBrain(ReviewedBackend(None,review=value)),'Объясни session-only-private-logon.',a,[],{'mode':'explain','transport':'json_schema','tools':[]})
    assert 'session-only-private-logon' not in json.dumps(result)


def test_alternative_method_requires_a_reference_to_recorded_evidence(tmp_path):
    store=context(tmp_path);h=seed(store)
    contract=Request('Получить среднее H.',required_operations=('calculate_baseline',))
    backend=ReviewedBackend(None,review=reviewed(not_needed=['calculate_baseline']))
    result=rr.review(ac.LlamaBrain(backend),contract.query,store,[],{'mode':'execute','transport':'json_schema','tools':[]},contract,['выполнить calculate_baseline'])
    assert result['status']=='error' and 'no cited evidence' in result['reason']


@pytest.mark.parametrize('optional',[[],['calculate_baseline']])
def test_review_schema_compiles_with_the_official_standalone_converter(optional):
    import importlib.metadata
    import importlib.util
    try:
        distribution=importlib.metadata.distribution('llama-cpp-python')
    except importlib.metadata.PackageNotFoundError:
        pytest.skip('Optional llama-cpp-python schema converter is not installed')
    spec=importlib.util.spec_from_file_location('_review_schema_converter',distribution.locate_file('llama_cpp/llama_grammar.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    grammar=module.LlamaGrammar.from_json_schema(json.dumps(rr.schema(optional)),verbose=False)
    assert 'root ::=' in grammar._grammar and 'not_needed' in grammar._grammar


def test_a_fetch_record_alone_cannot_waive_a_requested_calculation(tmp_path):
    store=context(tmp_path);h=seed(store)
    contract=Request('Получи уровень H.',required_operations=('calculate_baseline',))
    log=[{'tool':'fetch_observatory_data','ok':True,'frame_handle':h}]
    backend=ReviewedBackend(None,review=reviewed(not_needed=['calculate_baseline'],evidence=[0]))
    result=rr.review(ac.LlamaBrain(backend),contract.query,store,log,{'mode':'execute','transport':'json_schema','tools':[]},contract,['выполнить calculate_baseline'])
    assert result['status']=='error' and 'no computed result' in result['reason']


def test_missing_outcome_stays_incomplete_if_no_structured_review_adapter_exists(tmp_path):
    store=context(tmp_path);seed(store)
    intent={'status':'ok','mode':'execute','transport':'json_schema','stations':['irt'],'dates':['2024-09-10'],
            'date_scope':'days','tools':['get_statistics'],'metrics':[], 'required_operations':['get_statistics'],'required_metrics':[]}
    result=ac.run_agent('Получить статистику H IRT за 2024-09-10.',brain=ac.ScriptedBrain([{'content':'Готово.'}]*3),
                        store=store,intent_plan=intent,verbose=False)
    assert not result['ok'] and result['error']=='request_incomplete'


def test_review_receives_actual_archive_member_paths_not_the_project_tree(tmp_path):
    store=context(tmp_path)
    members={'1/X.html':'irt/2024-09-10/X.html','2/Y.html':'irt/2024-09-10/Y.html'}
    log=[{'tool':'export_project','ok':True,'export_result':{'filename':'custom.zip','source_files':members}}]
    result=rr.review(ac.LlamaBrain(ReviewedBackend(None,review=reviewed('facts'))),'Перечисли файлы ZIP.',store,log,
                     {'mode':'execute','transport':'json_schema','tools':[]})
    assert result['status']=='ok'
    assert result['input']['artifacts']==[{'tool':'export_project','file':'custom.zip','members':['1/X.html','2/Y.html'],'omitted_members':0}]


def test_production_gradio_handler_preserves_review_and_records_both_stages(tmp_path):
    import runpy
    import gradio_app as ga
    helper=runpy.run_path(str(ROOT/'tools/run_chat_acceptance.py'))
    store=context(tmp_path);h=seed(store)
    backend=ReviewedBackend(wire('compute','facts',['calculate_baseline'],stations=['IRT'],dates=['2024-09-10']),
             call('calculate_baseline',dict(df_descriptor=h,component='H',mode='full')),
             review=reviewed(text='Получен уровень 5 nT; спокойствие окна ещё нужно проверить.',evidence=[0]))
    records=[]
    with helper['_record_results'](ga,ac,records):
        result=ga.make_chat_handler(ac.LlamaBrain(backend))('Получи опорный уровень H IRT за 2024-09-10 и объясни ограничения.',[],[],store)
    messages,*_=ga.split_result(result)
    assert messages[-1]['content'].startswith('Получен уровень 5')
    assert '999' not in messages[-1]['content']
    final=records[-1]['result']
    assert final['response_review']['status']=='ok' and '999' in final['draft_text']
    assert store.conversation.evidence and '999' not in json.dumps(store.conversation.evidence)


def test_operator_audit_cannot_pass_a_failed_final_answer_with_successful_intermediates():
    import runpy
    assess=runpy.run_path(str(ROOT/'tools/run_dual_stage_review.py'))['assess_operators']
    stats={'H':dict(min=1,max=2,median=1.5,mean=1.5,missing=0)}
    calls=[{'ok':True,'statistics':stats,'input_provenance':[{'handle':'raw:irt:2024-09-10'}]},
           {'ok':True,'plot':'fixture.html'}]
    turn={'id':'LANG-01','answer':'Частично.','engine_results':[{'result':{'ok':False,'tool_calls':calls}}]}
    audit=assess(turn)
    assert audit['status']=='FAIL' and 'final engine result is incomplete or failed' in audit['errors']


@pytest.mark.parametrize('raw,finish',[('not json','stop'),('{}','stop'),('{"style":"interpretation","text":"unfinished','length')])
def test_malformed_or_truncated_review_preserves_results_without_false_success(tmp_path,raw,finish):
    class BrokenReview(ReviewedBackend):
        def create_chat_completion(self,**kwargs):
            if 'style' in kwargs.get('response_format',{}).get('schema',{}).get('properties',{}):
                self.calls.append(copy.deepcopy(kwargs))
                return {'choices':[{'message':{'content':raw},'finish_reason':finish}]}
            return super().create_chat_completion(**kwargs)
    store=context(tmp_path);h=seed(store)
    backend=BrokenReview(wire('compute','facts',['get_statistics'],stations=['IRT'],dates=['2024-09-10']),
                        call('get_statistics',dict(df_descriptor=h,components=['H'])))
    result=ac.run_agent('Получи статистику H IRT 2024-09-10.',brain=ac.LlamaBrain(backend),store=store,verbose=False)
    assert not result['ok'] and result['error']=='response_review_failed'
    assert result['tool_calls'][0]['ok'] and 'медиана: 5.0' in result['text']
    assert result['response_review']['raw_response']==raw
    assert len([c for c in backend.calls if 'style' in c.get('response_format',{}).get('schema',{}).get('properties',{})])==1
