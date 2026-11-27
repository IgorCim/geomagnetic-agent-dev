"""Two axes: task execution and evidence delivery; no live-Qwen PASS claims."""
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import agent_core as ac
import project_store as ps
import request_planner as rp
import research_context as rc
from chat_session import SessionMemory, capture_calculation
from test_structured_request_planner import Backend, wire
from test_request_planner import native


def store_with_data(tmp_path):
    store=ac.FrameStore(workspace=ps.Workspace(root=tmp_path/'projects'))
    h='raw:irt:2024-09-12'
    store.put(h,pd.DataFrame({'timestamp':pd.date_range('2024-09-12',periods=8,freq='h',tz='UTC'),
                            'X':[3.,0.,3.,0.,10.,10.,10.,10.], 'Y':[0.,4.,0.,4.,0.,0.,0.,0.], 'Z':[0.]*8}))
    return store,h


def reply(name,args):
    return {'choices':[{'message':native(name,args)}]}


def test_optional_metrics_do_not_block_a_real_outlier_result(tmp_path):
    store,h=store_with_data(tmp_path)
    intent=wire('compute','interpretation',['detect_anomalies'],['sq_baseline','gic'],['IRT'],['2024-09-12'])
    intent['work']=[{**item,'required':item['id']=='detect_anomalies'} for item in intent['work']]
    backend=Backend(intent,reply('detect_anomalies',dict(df_descriptor=h,component='H',sigma_threshold=3)),
                    {'choices':[{'message':{'content':'Порог проверен. Отсутствие выбросов само по себе не исключает бурю.'}}]})
    result=ac.run_agent('Проверь выбросы H IRT за 12 сентября 2024 года при 3 сигма и объясни пределы вывода.',
                        brain=ac.LlamaBrain(backend),store=store,verbose=False)
    assert result['ok'],result
    assert [x['tool'] for x in result['tool_calls']]==['detect_anomalies']
    assert result['tool_calls'][0]['calculation']['n_flagged']==0
    assert 'apply_physics_formula' in {x['function']['name'] for x in backend.calls[1]['tools']}
    evidence=backend.calls[-1]['messages'][0]['content']
    assert 'n_flagged' in evidence and 'does not establish storm absence' in evidence
    assert result['request_plan']['required_metrics']==[]


def test_a_joint_request_still_requires_its_named_physical_metric(tmp_path):
    store,h=store_with_data(tmp_path)
    intent=wire('compute','interpretation',['detect_anomalies'],['gic'],['IRT'],['2024-09-12'])
    intent['work']=[{**item,'required':True} for item in intent['work']]
    decoded=rp.decode_arguments(intent,require_outcomes=True)
    request=store.conversation.prepare('Проверь выбросы H и вычисли GIC-прокси.',store)
    request.apply_intent(decoded)
    payload,_=ac.HANDLERS['detect_anomalies'](dict(df_descriptor=h,component='H',sigma_threshold=3),store)
    entry={'tool':'detect_anomalies','ok':True};capture_calculation(entry,entry['tool'],payload)
    assert any('gic' in x for x in request.missing([entry],store))
    fake={'tool':'apply_physics_formula','ok':True,'calculation':{'metric':'sq_baseline'}}
    assert any('gic' in x for x in request.missing([entry,fake],store))
    payload,_=ac.HANDLERS['apply_physics_formula']({'df_descriptor':h,'metric':'gic'},store)
    actual={'tool':'apply_physics_formula','ok':not ac.is_error(payload)}
    capture_calculation(actual,actual['tool'],payload)
    assert not request.missing([entry,actual],store)


@pytest.mark.parametrize('required', [['run_python'],['gic'],['plot_components']])
def test_required_outcome_must_be_in_selected_capabilities(required):
    value=wire('compute','facts',['get_statistics']);value['work']=[{'id':required[0]+'_invented','required':True}]
    with pytest.raises(ValueError,match='unregistered capability'):rp.decode_arguments(value,require_outcomes=True)


def test_structured_production_cannot_omit_outcomes_or_echo_a_clarification():
    value=wire();value.pop('work')
    with pytest.raises(ValueError,match='missing work'):rp.decode_arguments(value,require_outcomes=True)
    value=wire('clarify','interpretation',question='Сделай вывод увереннее.')
    with pytest.raises(ValueError,match='repeats'):
        rp.validate(rp.decode_arguments(value),'Сделай вывод увереннее.','')


def test_measured_baselines_and_their_effect_survive_same_slot_reuse(tmp_path):
    store,h=store_with_data(tmp_path)
    memory=store.conversation
    log=[];series=[];values=[]
    for mode in ['full','night']:
        args={'df_descriptor':h,'component':'H','mode':mode,'night_hours':[0,4]}
        payload,_=ac.HANDLERS['calculate_baseline'](args,store)
        values.append(payload['baseline'])
        entry={'tool':'calculate_baseline','ok':True,'arguments':args,
               'input_provenance':[ac.frame_provenance(store,h)]}
        capture_calculation(entry,entry['tool'],payload);log.append(entry)
        args={'df_descriptor':h,'component':'H','metric':'anomaly','baseline_value':payload['baseline']}
        payload,_=ac.HANDLERS['calculate_derived_math'](args,store)
        series.append(store.get(payload['frame_handle'])['ANOMALY'].copy())
        entry={'tool':'calculate_derived_math','ok':True,'arguments':args,
               'frame_handle':payload['frame_handle'],'input_provenance':[ac.frame_provenance(store,h)]}
        capture_calculation(entry,entry['tool'],payload);log.append(entry)
    assert values==[6.75,3.5]
    assert np.allclose(series[1]-series[0],3.25)
    assert np.ptp(series[1])==np.ptp(series[0])
    memory.record(log,store)
    assert len(memory.evidence)==4
    evidence=rc.render(memory.evidence,'Сравни влияние опорного уровня.',{'tools':['calculate_baseline']})
    assert '6.75' in evidence and '3.5' in evidence
    assert 'full' in evidence and 'night' in evidence and h in evidence and 'rows' in evidence
    assert 'max-min is unchanged' in evidence
    assert not SessionMemory().evidence
    assert len(evidence)<=rc.MAX_CONTEXT_CHARS


def test_evidence_is_successful_handler_data_never_model_prose_or_secret(tmp_path,monkeypatch):
    store,h=store_with_data(tmp_path)
    monkeypatch.setenv('SUPERMAG_LOGON','secret-fixture')
    log=[{'ok':False,'tool':'calculate_baseline','calculation':{'baseline':999}},
         {'ok':True,'tool':'unknown','calculation':{'baseline':888}},
         {'ok':True,'tool':'get_statistics','arguments':{'password':'secret-fixture'},
          'resolved_handles':[h],'statistics':{'H':{'mean':4,'count':8,'unit':'nT'}},
          'text':'invented correlation=1'}]
    store.conversation.record(log,store)
    context=store.conversation.context()
    assert 'mean' in context and '999' not in context and '888' not in context
    assert 'invented correlation' not in context and 'secret-fixture' not in context
    assert rc.record({'ok':True,'tool':'apply_physics_formula','calculation':{'result':'secret-fixture'}})
    assert 'secret-fixture' not in rc.render([rc.record({'ok':True,'tool':'apply_physics_formula','calculation':{'result':'secret-fixture'}})])


def test_evidence_and_rendered_context_remain_bounded_and_disclose_omissions():
    entries=[{'ok':True,'tool':'get_statistics','resolved_handles':['raw:irt:2024-09-12'],
              'statistics':{'H':{'mean':i,'count':1440,'unit':'nT'}}} for i in range(100)]
    evidence=rc.merge([],entries)
    assert len(evidence)==rc.MAX_RECORDS
    context=rc.render(evidence,max_chars=1100)
    assert len(context)<=1100 and json.loads(context)['omitted_records']>0


def test_discussion_continuation_receives_recorded_values_without_new_calculation(tmp_path):
    store,h=store_with_data(tmp_path)
    args={'df_descriptor':h,'component':'H','mode':'full'}
    payload,_=ac.HANDLERS['calculate_baseline'](args,store)
    entry={'ok':True,'tool':'calculate_baseline','arguments':args,'input_provenance':[ac.frame_provenance(store,h)]}
    capture_calculation(entry,entry['tool'],payload)
    store.conversation.record([entry],store)
    backend=Backend(wire('discuss','interpretation'),
                    {'choices':[{'message':{'content':'Среднее H=6.75 nT относится к этой таблице; причинная связь не проверялась.'}}]})
    result=ac.run_agent('Сформулируй вывод только по уже полученным наблюдениям.',brain=ac.LlamaBrain(backend),store=store,verbose=False)
    assert result['ok'] and result['tool_calls']==[]
    prompt=backend.calls[-1]['messages'][0]['content']
    assert h in prompt and '6.75' in prompt and 'full' in prompt
    assert 'Claim a computed correlation only' in prompt
    assert '6.75' in backend.calls[0]['messages'][1]['content']


def test_planner_supplies_verified_aliases_for_codes_and_marks_unknown_names():
    backend=Backend(wire('discuss','plan',stations=['IRT','BOU'],dates=['2024-09-12']))
    rp.plan_request(ac.LlamaBrain(backend),'IRT BOU 2024-09-12: предложи метод.',station_aliases={'IRT':'Иркутск'})
    labels=json.loads(backend.calls[0]['messages'][1]['content'])['station_labels']
    assert labels=={'BOU':None,'IRT':'Иркутск'}


def test_rejected_structured_plan_retains_raw_evidence_with_secret_redaction(monkeypatch):
    monkeypatch.setenv('SUPERMAG_LOGON','secret-fixture')
    value=wire('discuss','plan',stations=['IRT'],dates=['1900-01-01'],question='secret-fixture')
    result=rp.plan_request(ac.LlamaBrain(Backend(value)),'IRT за 2024-09-12: предложи метод.')
    assert result['status']=='error' and result['detail']=='invented date'
    assert '1900-01-01' in result['raw_response']
    assert 'secret-fixture' not in result['raw_response'] and '[redacted]' in result['raw_response']


def test_references_cover_the_four_distinct_claims_without_becoming_observations():
    query='Оцени: отсутствие выбросов 3 сигма исключает бурю; абсолютное H показывает силу бури; dH/dt даёт GIC в амперах; корреляция доказывает причину.'
    result=json.loads(rc.render([],query,{'mode':'explain','tools':[]}))
    notes=' '.join(result['reference_notes'])
    assert result['records']==[]
    assert 'sigma outlier' in notes and 'background field' in notes
    assert 'amperes' in notes and 'not unique causation' in notes


def test_local_time_evidence_preserves_station_and_exact_utc_timestamp(tmp_path):
    store,h=store_with_data(tmp_path)
    args={'station_code':'IRT','timestamp':'2024-09-12T00:00:00Z'}
    payload,_=ac.HANDLERS['calculate_local_time'](args,store)
    assert not ac.is_error(payload),payload
    entry={'ok':True,'tool':'calculate_local_time','arguments':args,
           'local_time':{key:payload[key] for key in ('lt_hm','local_mean_solar_datetime','longitude','model')}}
    store.conversation.record([entry],store)
    saved=store.conversation.evidence[0]
    assert saved['parameters']['station_code']=='IRT'
    assert saved['parameters']['timestamp']=='2024-09-12T00:00:00Z'
    assert saved['local_time']['model']==payload['model']
    assert not SessionMemory().evidence
