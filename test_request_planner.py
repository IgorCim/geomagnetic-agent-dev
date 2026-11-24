"""Request-understanding regressions with real executor/adapter and controlled decisions.

No claim about a live Qwen model: the model's decisions are scripted, handlers real.
"""
import copy
import json
import pytest
import pandas as pd
import agent_core as ac
import gradio_app as ga
from chat_session import SessionMemory, station_in
import request_planner as rp


def decision(mode='execute',tools=None,stations=None,dates=None,scope='days',metrics=None):
    return dict(mode=mode,tools=tools or [],stations=stations or [],dates=dates or [],
                date_scope=scope,metrics=metrics or [])


def native(name,args):
    return {'content':'','tool_calls':[{'type':'function','function':{'name':name,'arguments':json.dumps(args)}}]}


class Brain:
    supports_request_planning=True
    def __init__(self,plan,*replies):self.replies=iter([native('plan_request',plan),*replies]);self.seen=[]
    def chat(self,messages,tools,max_tokens=None):
        self.seen.append((copy.deepcopy(messages),copy.deepcopy(tools),max_tokens))
        return next(self.replies,{'content':'Объяснение по рассчитанным данным.'})


def seed(store,day='2024-09-10',periods=4,hourly=True,station='irt'):
    handle=f'raw:{station}:{day}'
    store.put(handle,pd.DataFrame({'timestamp':pd.date_range(day,periods=periods,freq='h' if hourly else 'min',tz='UTC'),
                                 'X':[3.]*periods,'Y':[4.]*periods,'Z':[12.]*periods}))
    return handle


@pytest.mark.parametrize('term',['GIC','SSC','SML','SME','SMU','UTC','GSM','GSE','MLT'])
def test_scientific_entities_are_not_stations(term):
    assert station_in(f'Объясни {term}.') is None
    assert station_in(f'Для IRT объясни {term}.')=='irt'


def test_quoted_station_claim_is_not_operational_station():
    assert station_in('Оцени абзац: «Станция BOU измеряет GIC».') is None
    assert station_in('Загрузи API за 2024-09-10')=='api'


@pytest.mark.parametrize('query',[
 'Получи опорный уровень H для IRT за 2024-09-10.',
 'Нужна точка отсчёта горизонтального поля IRT за 2024-09-10.',
 'Оцени фон H IRT за 2024-09-10 и объясни выбор.'
])
def test_natural_baseline_gets_real_tool_and_dependencies(query):
    plan=decision('analyze',['calculate_baseline'],['IRT'],['2024-09-10'])
    store=ac.FrameStore();h=seed(store)
    brain=Brain(plan,native('calculate_baseline',dict(df_descriptor=h,component='H',mode='full')),
                {'content':'Опорный уровень рассчитан; пригодность требует проверки спокойствия окна.'})
    result=ac.run_agent(query,brain=brain,store=store,verbose=False,max_rounds=3)
    assert result['ok'],result
    assert [x['function']['name'] for x in brain.seen[0][1]]==['plan_request']
    assert {'calculate_baseline','calculate_derived_components','fetch_observatory_data'} <= {x['function']['name'] for x in brain.seen[1][1]}
    assert result['tool_calls'][0]['calculation']['baseline']==5
    assert result['tool_calls'][0]['input_provenance'][0]['rows']==4
    assert result['text'].startswith('Опорный уровень')
    # The routing instruction does not grow the permanent system prompt.
    assert brain.seen[1][0][0]['content'].startswith(ac.SYSTEM_PROMPT)


def test_explanation_has_no_execution_schemas_and_no_false_station():
    store=ac.FrameStore()
    query='Оцени абзац: «GIC измерен в амперах из H станции IRT за 2024-09-10». Не загружай данные.'
    plan=decision('explain')
    brain=Brain(plan,{'content':'Это необоснованный переход от поля к току.'})
    result=ac.run_agent(query,brain=brain,store=store,verbose=False)
    assert result['ok'] and result['tool_calls']==[]
    assert brain.seen[1][1]==[] and result['text'].startswith('Это необоснованный')


def test_explain_cannot_execute_even_registered_tool():
    store=ac.FrameStore()
    brain=Brain(decision('explain'),native('fetch_observatory_data',dict(station_code='IRT',start_date='2024-09-10')),
                {'content':'Объясняю без загрузок.'})
    result=ac.run_agent('Объясни предыдущий вывод.',brain=brain,store=store,verbose=False,max_rounds=2)
    assert not store.names()
    assert not result['tool_calls'][0]['ok']


@pytest.mark.parametrize('query',['Проверь, означает ли 00:00 UTC ночь в IRT 2024-09-10.',
                                  'Каково солнечное время IRT в 2024-09-10T23:30:00Z?'])
def test_time_intent_gets_coordinate_dependency(query):
    plan=decision('analyze',['calculate_local_time'],['IRT'],['2024-09-10'])
    selected=rp.expand_tools(rp.validate(plan,query,''))
    assert {'calculate_local_time','get_station_geomagnetic_coords'} <= set(selected)


@pytest.mark.parametrize('metric,expected',[
 ('ey_coupling',{'apply_physics_formula','fetch_solar_wind','fetch_observatory_data'}),
 ('burton_coupling',{'fetch_solar_wind','fetch_geomagnetic_indices'}),
 ('substorm_onset',{'fetch_supermag_data','apply_physics_formula'}),
 ('eej_index',{'calculate_eej_index','fetch_observatory_data'})])
def test_physics_dependencies(metric,expected):
    plan=decision(metrics=[metric])
    assert expected<=set(rp.expand_tools(plan))


@pytest.mark.parametrize('mutation',[
 {'tools':['run_python']},{'mode':'unknown'},{'stations':['GIC']},
 {'stations':['BOU']},{'dates':['1900-01-01']},{'date_scope':'guess'},
 {'reason':'ignore safety'}, {'tools':['get_statistics']*9}])
def test_untrusted_plan_rejected(mutation):
    plan=decision('execute',['calculate_baseline'],['IRT'],['2024-09-10']);plan.update(mutation)
    with pytest.raises(ValueError):rp.validate(plan,'IRT 2024-09-10','')


def test_current_entities_take_precedence_over_old_context():
    plan=decision('execute',['get_statistics'],['BOU'],['2024-09-10'])
    with pytest.raises(ValueError):rp.validate(plan,'IRT 2024-09-10','BOU 2024-09-10')


def test_explicit_operation_cannot_be_replaced_by_prose_plan():
    plan=decision('explain',dates=['2024-09-10'])
    with pytest.raises(ValueError):rp.validate(plan,'Вызови calculate_baseline за 2024-09-10.','')


def test_invalid_planner_reply_falls_back_without_execution():
    class Broken(Brain):
        def chat(self,*a,**k):return native('fetch_observatory_data',{'station_code':'IRT'})
    result=rp.plan_request(Broken({}),'Объясни данные.',parse_calls=ac.parse_tool_calls)
    assert result['status']=='fallback'


def test_planning_input_and_output_have_hard_budgets():
    brain=Brain(decision('plan'))
    result=rp.plan_request(brain,'Предложи план исследования.',parse_calls=ac.parse_tool_calls)
    assert result['status']=='ok'
    assert brain.seen[0][2]==rp.MAX_OUTPUT_TOKENS
    assert len(brain.seen[0][0][0]['content'])<3500
    other=Brain({})
    assert rp.plan_request(other,'x'*9000)['reason']=='planner_input_budget'
    assert other.seen==[]


def test_no_download_prohibition_survives_semantic_decision():
    store=ac.FrameStore();seed(store)
    q='Для IRT за 2024-09-10 получи опорный уровень. Не загружай данные.'
    intent=rp.validate(decision('execute',['calculate_baseline'],['IRT'],['2024-09-10']),q,'')
    request=SessionMemory().prepare(q,store);request.apply_intent(intent)
    assert request.check_call('fetch_observatory_data',{'station_code':'IRT','start_date':'2024-09-10'},store)
    assert request.check_call('fetch_many',{},store)


def test_separate_days_reject_three_day_fetch_and_three_day_statistics():
    store=ac.FrameStore();h=seed(store,periods=72)
    q='Сравни Kyoto за 2024-09-10 и 2024-09-12.'
    intent=rp.validate(decision('analyze',['fetch_geomagnetic_indices','get_statistics'],dates=['2024-09-10','2024-09-12']),q,'')
    request=SessionMemory().prepare(q,store);request.apply_intent(intent)
    assert request.check_call('fetch_geomagnetic_indices',{'start_date':'2024-09-10','end_date':'2024-09-12'},store)
    assert 'date_scope_mismatch' in request.check_call('get_statistics',{'df_descriptor':h,'components':['H']},store)


def test_explicit_range_still_allows_period_statistics():
    store=ac.FrameStore();h=seed(store,periods=72)
    q='Получи статистику H IRT за период 2024-09-10 — 2024-09-12.'
    intent=rp.validate(decision('execute',['get_statistics'],['IRT'],['2024-09-10','2024-09-12'],scope='range'),q,'')
    request=SessionMemory().prepare(q,store);request.apply_intent(intent)
    assert request.check_call('get_statistics',{'df_descriptor':h,'components':['H']},store) is None


def test_stats_keep_numeric_values_and_actual_coverage_in_model_message():
    store=ac.FrameStore();h=seed(store,periods=72)
    payload,_=ac._handle_stats({'df_descriptor':h,'components':['X','Y','Z','H']},store)
    payload['input_provenance']=[ac.frame_provenance(store,h)]
    compact=json.loads(ac._compact_tool_message(payload))
    assert compact['input_provenance'][0]['rows']==72
    assert compact['input_provenance'][0]['end'].startswith('2024-09-12')
    assert compact['statistics']['H']['mean']==5
    assert len(json.dumps(compact))<1600


def test_plan_stage_through_production_llama_adapter():
    class FakeLlama:
        def __init__(self):self.calls=[]
        def create_chat_completion(self,**kw):
            self.calls.append(kw)
            result=native('plan_request',decision('explain')) if len(self.calls)==1 else {'content':'Критика утверждения.'}
            return {'choices':[{'message':result}]}
    llm=FakeLlama();brain=ac.LlamaBrain(llm)
    result=ac.run_agent('Объясни различие наблюдения и гипотезы.',brain=brain,verbose=False)
    assert result['ok'] and len(llm.calls)==2
    assert llm.calls[0]['max_tokens']==rp.MAX_OUTPUT_TOKENS
    assert llm.calls[1]['tools']==[]


@pytest.mark.parametrize('phrase',[
 '10 и 12 сентября 2024 года', '10, 12 сентября 2024',
 '10.09.2024 и 12.09.2024', '2024-09-10 и 2024-09-12'])
def test_calendar_dates_normalized_without_guessing(phrase):
    query='Сравни IRT за '+phrase
    plan=decision('analyze',['get_statistics'],['IRT'],['2024-09-10','2024-09-12'])
    assert rp.validate(plan,query,'')['dates']==plan['dates']
    plan['dates']=['2024-09-10']
    with pytest.raises(ValueError):rp.validate(plan,query,'')


def test_date_without_year_is_not_guessed():
    with pytest.raises(ValueError):
        rp.validate(decision('execute',['get_statistics'],dates=['2024-09-10']),'10 сентября','')


@pytest.mark.parametrize('label',['Иркутск','Иркутске','Иркутска'])
def test_existing_verified_station_label_can_be_normalized(label):
    aliases={'IRT':'Иркутск','MOS':'Москва'}
    query=f'Нужен фон H для станции {label} 10 сентября 2024 года.'
    plan=decision('analyze',['calculate_baseline'],['IRT'],['2024-09-10'])
    assert rp.validate(plan,query,'',aliases)['stations']==['irt']
    plan['stations']=['MOS']
    with pytest.raises(ValueError):rp.validate(plan,query,'MOS 2024-09-10',aliases)


def test_production_entry_passes_relevant_station_labels_to_miniplanner():
    store=ac.FrameStore();seed(store)
    brain=Brain(decision('analyze',['calculate_baseline'],['IRT'],['2024-09-10']))
    plan=ac.understand_request('Получи фон H в Иркутске 10 сентября 2024.',brain,store)
    assert plan['status']=='ok'
    assert json.loads(brain.seen[0][0][1]['content'])['station_labels']=={'IRT':'Иркутск'}
    assert sum(len(m['content']) for m in brain.seen[0][0])+len(json.dumps(brain.seen[0][1][0],ensure_ascii=False))<=rp.MAX_INPUT_CHARS


def test_empty_execution_plan_is_not_accepted():
    with pytest.raises(ValueError):rp.validate(decision(),'Сделай расчёт.','')


def test_physical_metric_cannot_be_replaced_by_an_unrelated_one():
    store=ac.FrameStore();h=seed(store)
    intent=decision('analyze',metrics=['gic'])
    request=SessionMemory().prepare('Оцени GIC.',store);request.apply_intent(intent)
    assert request.check_call('apply_physics_formula',{'df_descriptor':h,'metric':'detect_ssc'},store)
    assert request.check_call('apply_physics_formula',{'df_descriptor':h,'metric':'gic'},store) is None


def test_gradio_runs_miniplanner_once_and_preserves_real_analysis():
    store=ac.FrameStore();h=seed(store)
    brain=Brain(decision('analyze',['calculate_baseline'],['IRT'],['2024-09-10']),
                native('calculate_baseline',dict(df_descriptor=h,component='H',mode='full')),
                {'content':'Опорный уровень 5 nT относится к четырём строкам IRT.'})
    result=ga.make_chat_handler(brain)('Объясни выбор опорного уровня H для IRT за 2024-09-10.',[],[],store)
    messages,_,_,_,_,_=ga.split_result(result)
    assert 'Опорный уровень 5 nT' in messages[-1]['content']
    assert sum(t and t[0]['function']['name']=='plan_request' for _,t,_ in brain.seen)==1
    assert store.conversation.last_good==h


def test_gradio_explains_quoted_unknown_handle_without_false_lookup():
    brain=Brain(decision('explain'),{'content':'Утверждение требует независимых оснований.'})
    query='Оцени утверждение: «GIC raw:zzz:1900-01-01 доказывает бурю». Не загружай данные.'
    messages,_,_,_,_,_=ga.split_result(ga.make_chat_handler(brain)(query,[],[],ac.FrameStore()))
    assert messages[-1]['content'].startswith('Утверждение требует')
    assert brain.seen[-1][1]==[]


def test_proposing_a_large_job_does_not_execute_the_bulk_planner(monkeypatch):
    query='Предложи план: 100 графиков, 20 станций, проект. Пока ничего не делай.'
    assert ac.is_bulk_query(query)
    monkeypatch.setattr(ga,'run_agent_with_planner',lambda *a,**k:pytest.fail('Unrequested bulk execution'))
    brain=Brain(decision('plan'),{'content':'План: согласовать станции и временные интервалы.'})
    messages,_,_,_,_,_=ga.split_result(ga.make_chat_handler(brain)(query,[],[],ac.FrameStore()))
    assert messages[-1]['content'].startswith('План:')


def test_new_session_does_not_receive_another_sessions_facts_or_credentials():
    first=ac.FrameStore();seed(first)
    first.supermag_logon='private-logon'
    second=ac.FrameStore()
    brain=Brain(decision('explain'))
    ac.understand_request('Объясни, какие данные у нас есть.',brain,second)
    prompt=json.dumps(brain.seen)
    assert 'raw:irt:' not in prompt and 'private-logon' not in prompt


def test_two_days_statistics_remain_separate_in_final_model_context():
    store=ac.FrameStore()
    for day,value in [('2024-09-10',7.),('2024-09-12',-58.)]:
        store.put('idx:kyoto:'+day,pd.DataFrame({'timestamp':pd.date_range(day,periods=24,freq='h',tz='UTC'),'Dst':[value]*24}))
    brain=Brain(decision('analyze',['get_statistics'],dates=['2024-09-10','2024-09-12']),
                native('get_statistics',dict(df_descriptor='idx:kyoto:2024-09-10',components=['Dst'])),
                native('get_statistics',dict(df_descriptor='idx:kyoto:2024-09-12',components=['Dst'])),
                {'content':'Дни различаются; вывод ограничен этими данными.'})
    result=ac.run_agent('Сравни Kyoto за 10 и 12 сентября 2024 года.',brain=brain,store=store,verbose=False)
    assert result['ok'],result
    messages=brain.seen[-1][0]
    data=[json.loads(m['content']) for m in messages if m['role']=='tool']
    assert [(d['input_provenance'][0]['start'][:10],d['input_provenance'][0]['rows'],d['statistics']['Dst']['mean']) for d in data]==[
        ('2024-09-10',24,7.),('2024-09-12',24,-58.)]
    assert result['plots']==[]
    assert result['request_plan']['mode']=='analyze'
    assert '2024-09-10' in messages[1]['content'] and '2024-09-12' in messages[1]['content']


def test_provenance_payload_is_bounded_and_discloses_omission():
    payload={'input_provenance':[{'handle':'raw:'+('a'*2000),'rows':24}], 'X':{'mean':1,'median':1,'count':24}}
    result=ac._compact_tool_message(payload)
    assert len(result)<=1600
    assert json.loads(result)['error']=='provenance_context_budget'


def test_capability_catalog_matches_registered_executor():
    assert set(rp.CAPABILITIES)==set(ac.TOOLS_BY_NAME)
    assert 'plan_request' not in ac.TOOLS_BY_NAME


def test_invalid_semantic_decision_retains_legacy_tool_path():
    store=ac.FrameStore();h=seed(store)
    brain=Brain({'mode':'unknown'},native('get_statistics',{'df_descriptor':h,'components':['H']}),{'content':'Готово.'})
    result=ac.run_agent('Получи статистику H IRT за 2024-09-10.',brain=brain,store=store,verbose=False)
    assert result['ok'],result
    assert result['tool_calls'][0]['tool']=='get_statistics'
    assert result['request_plan']['status']=='fallback'


@pytest.mark.parametrize('literal',['"IRT"','«IRT»'])
def test_quoted_station_literal_is_allowed_when_the_semantic_plan_requests_it(literal):
    query=f'Получи статистику H станции {literal} за 2024-09-10.'
    assert rp.validate(decision('execute',['get_statistics'],['IRT'],['2024-09-10']),query,'')['stations']==['irt']


def test_scientific_review_replays_original_wording_and_session_boundaries():
    from pathlib import Path
    import runpy
    root=Path(__file__).resolve().parent
    runner=runpy.run_path(str(root/'tools/run_request_planner_review.py'))
    plan=runner['build_plan'](root)
    assert len(plan)==9 and sum(len(p['cases']) for p in plan)==12
    assert [c['case'] for c in plan[6]['cases']]==['baselines','local_night']
    assert [c['case'] for c in plan[-1]['cases']]==['hypothesis','overclaim','strongest']


def test_provenance_context_does_not_collapse_two_stations_to_active_handle():
    store=ac.FrameStore();one=seed(store,station='irt');two=seed(store,station='bou')
    result=[ac.frame_provenance(store,h) for h in [one,two]]
    assert [r['handle'] for r in result]==[one,two]
    assert all(r['rows']==4 for r in result)


def test_switch_can_keep_old_routing_without_consuming_a_planner_reply(monkeypatch):
    monkeypatch.setenv('GEOMAG_REQUEST_PLANNER','0')
    brain=Brain(decision('explain'))
    assert ac.understand_request('Объясни GIC.',brain,ac.FrameStore()) is None
    assert brain.seen==[]


@pytest.mark.parametrize('source',['fetch_solar_wind','fetch_geomagnetic_indices','fetch_supermag_data'])
def test_index_or_wind_statistics_do_not_gain_unrelated_ground_dependencies(source):
    selected=rp.expand_tools(decision('analyze',[source,'get_statistics']),has_ground=True)
    assert 'get_statistics' in selected and source in selected
    assert 'fetch_observatory_data' not in selected
    assert 'calculate_derived_components' not in selected
