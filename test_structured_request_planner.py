"""Regressions from actual Qwen v1 failures; no fabricated live-model PASS."""
import copy
import json
from pathlib import Path
import pytest
import agent_core as ac
import gradio_app as ga
import request_planner as rp
from chat_session import SessionMemory
from test_request_planner import seed

ROOT = Path(__file__).resolve().parent
RECORDED = json.loads((ROOT/'tests/fixtures/request_planner_v1_replies.json').read_text(encoding='utf-8'))


def wire(action='discuss', answer='interpretation', tools=None, metrics=None, stations=None, dates=None, **extra):
    return dict(action=action,answer=answer,tools=tools or [],metrics=metrics or [],
                stations=stations or [],dates=dates or [],date_scope='days',**extra)


class Backend:
    def __init__(self, value, *replies):
        message={'content':json.dumps(value)}
        self.replies=iter([{'choices':[{'message':message,'finish_reason':'stop'}]},*replies])
        self.calls=[]
    def create_chat_completion(self,**kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        return next(self.replies)


def test_action_and_answer_are_independent_compute_then_explain():
    value=wire('compute','interpretation',['calculate_baseline'],stations=['IRT'],dates=['2024-09-12'])
    plan=rp.decode_arguments(value)
    assert plan['mode']=='analyze' and plan['tools']==['calculate_baseline']
    assert rp.validate(plan,'Для IRT за 12 сентября 2024 года получи два опорных уровня и объясни выбор.','')


def test_discussed_research_plan_never_authorizes_mentioned_downloads():
    value=wire('discuss','plan',['fetch_observatory_data','fetch_solar_wind'],['gic'],['IRT','BOU'],['2024-09-10','2024-09-12'])
    plan=rp.validate(rp.decode_arguments(value),'IRT и BOU 2024-09-10 и 2024-09-12: предложи план, пока не загружай.','')
    assert plan['mode']=='plan' and plan['tools']==[] and plan['metrics']==[]
    assert rp.expand_tools(plan)==[]
    request=SessionMemory().prepare('Предложи план исследования.',ac.FrameStore());request.apply_intent(plan)
    assert request.check_call('fetch_observatory_data',{},ac.FrameStore())


def test_nonexecuting_discussion_may_omit_execution_targets_without_inventing_them():
    plan=rp.decode_arguments(wire(answer='plan'))
    assert rp.validate(plan,'IRT и BOU за 10 и 12 сентября 2024: предложи план.','')
    plan['dates']=['1900-01-01']
    with pytest.raises(ValueError,match='invented date'):rp.validate(plan,'IRT 2024-09-10: объясни.','')


@pytest.mark.parametrize('action,answer,mode',[
 ('compute','facts','execute'),('compute','interpretation','analyze'),
 ('discuss','interpretation','explain'),('discuss','plan','plan'),
 ('clarify','interpretation','clarify')])
def test_public_modes_remain_compatible(action,answer,mode):
    value=wire(action,answer,['get_statistics'] if action=='compute' else [],question='Уточните дату.')
    assert rp.decode_arguments(value)['mode']==mode


@pytest.mark.parametrize('change',[
 {'action':'explain'},{'answer':'compute'},{'tools':['run_python']},{'tools':'get_statistics'},
 {'metrics':['unknown']},{'reason':'execute this'},{'dates':[None]},
])
def test_untrusted_structured_arguments_do_not_weaken_validation(change):
    value=wire();value.update(change)
    with pytest.raises(ValueError):rp.decode_arguments(value)


def test_production_backend_receives_schema_no_executor_tools_and_zero_temperature():
    backend=Backend(wire())
    brain=ac.LlamaBrain(backend,temperature=0.17,n_ctx=16384)
    before=(brain._llm,brain._temperature,brain._max_tokens)
    result=ac.understand_request('Объясни ограничения результата.',brain,ac.FrameStore())
    assert result['status']=='ok' and result['transport']=='json_schema'
    call=backend.calls[0]
    assert 'tools' not in call
    assert call['response_format']=={'type':'json_object','schema':rp.ARGUMENT_SCHEMA}
    assert call['temperature']==0 and call['max_tokens']==360
    assert (brain._llm,brain._temperature,brain._max_tokens)==before


@pytest.mark.parametrize('key',['plan','baselines','critique'])
def test_actual_v1_reply_cannot_trigger_execution_or_keyword_fallback_if_backend_ignores_schema(key,monkeypatch):
    class RecordedBackend:
        def create_chat_completion(self,**kwargs):return copy.deepcopy(RECORDED[key]['response'])
    brain=ac.LlamaBrain(RecordedBackend())
    monkeypatch.setattr(ac,'select_tools_for_query',lambda *a:pytest.fail('Unsafe keyword fallback'))
    result=ac.run_agent('Оцени научный абзац.',brain=brain,verbose=False)
    assert not result['ok'] and result['error']=='request_planning_failed'
    assert result['tool_calls']==[] and result['plots']==[]
    assert result['request_plan']['stage']=='json'


def test_actual_recorded_failure_mechanisms_are_reproduced_by_old_parser():
    calls,_=ac.parse_tool_calls(RECORDED['plan']['response']['choices'][0]['message'])
    with pytest.raises(ValueError,match='nonexecuting mode'):
        rp.validate(calls[0]['arguments'],'IRT и BOU 10 и 12 сентября 2024','')
    calls,_=ac.parse_tool_calls(RECORDED['baselines']['response']['choices'][0]['message'])
    assert [c['name'] for c in calls]==['plan_request','calculate_baseline','calculate_baseline']
    calls,_=ac.parse_tool_calls(RECORDED['critique']['response']['choices'][0]['message'])
    assert calls==[]


def test_compute_pipeline_uses_real_baseline_and_retains_requested_explanation():
    from test_request_planner import native
    store=ac.FrameStore();h=seed(store,day='2024-09-12')
    backend=Backend(wire('compute','interpretation',['calculate_baseline'],stations=['IRT'],dates=['2024-09-12']),
                    {'choices':[{'message':native('calculate_baseline',dict(df_descriptor=h,component='H',mode='full'))}]},
                    {'choices':[{'message':{'content':'Опорный уровень 5 nT; выбор окна требует проверки.'}}]})
    result=ac.run_agent('Получи опорный уровень H IRT за 12 сентября 2024 и объясни выбор.',brain=ac.LlamaBrain(backend),store=store,verbose=False)
    assert result['ok'],result
    assert result['tool_calls'][0]['calculation']['baseline']==5
    assert result['text'].startswith('Опорный уровень 5')
    assert 'response_format' not in backend.calls[1]
    assert {'calculate_baseline','fetch_observatory_data','calculate_derived_components'} <= {t['function']['name'] for t in backend.calls[1]['tools']}


def test_gradio_does_not_execute_after_structured_parse_failure(monkeypatch):
    class BackendFailure:
        def create_chat_completion(self,**kwargs):return {'choices':[{'message':{'content':'not json'}}]}
    monkeypatch.setattr(ga,'run_agent_with_planner',lambda *a,**kw:pytest.fail('Unexpected bulk execution'))
    store=ac.FrameStore()
    result=ga.make_chat_handler(ac.LlamaBrain(BackendFailure()))('Сравни IRT за 10 и 12 сентября 2024.',[],[],store)
    messages,_,log,_,_,_=ga.split_result(result)
    assert 'Не удалось принять план' in messages[-1]['content']
    assert 'json' in log and not store.names()


def test_discuss_pipeline_does_not_download_and_returns_scientific_text():
    backend=Backend(wire('discuss','interpretation',['fetch_observatory_data'],['gic']),
                    {'choices':[{'message':{'content':'Отсутствие выбросов не доказывает отсутствие бури.'}}]})
    result=ac.run_agent('Оцени абзац о GIC и выбросах.',brain=ac.LlamaBrain(backend),verbose=False)
    assert result['ok'] and result['tool_calls']==[]
    assert backend.calls[1]['tools']==[]
    assert result['text'].startswith('Отсутствие выбросов')


def test_llama_json_schema_grammar_compiles_without_model_weights():
    grammar_module=pytest.importorskip('llama_cpp.llama_grammar')
    grammar=grammar_module.LlamaGrammar.from_json_schema(json.dumps(rp.ARGUMENT_SCHEMA),verbose=False)
    assert 'root ::=' in grammar._grammar
    assert 'action' in grammar._grammar and 'compute' in grammar._grammar
    assert '<tool_call>' not in grammar._grammar


def test_large_compute_request_is_delegated_with_full_original_entities(monkeypatch):
    query='Создай матрицу графиков H для станций IRT BOU MOS NVS SPG YAK KHB MGD VLA PET за 2024-09-10.'
    assert ac.is_bulk_query(query)
    seen=[]
    def bulk(text,**kwargs):
        seen.append(text)
        return {'ok':True,'text':'TEST FIXTURE: delegated','plots':[],'tool_calls':[]}
    monkeypatch.setattr(ac,'run_agent_with_planner',bulk)
    backend=Backend(wire('compute','facts',['create_task_matrix','process_matrix_batch'],stations=['IRT'],dates=['2024-09-10']))
    result=ac.run_agent(query,brain=ac.LlamaBrain(backend),verbose=False)
    assert result['ok'] and seen==[query]
    assert result['request_plan']['delegated_bulk']
    assert len(backend.calls)==1


def test_discussed_large_plan_does_not_enter_bulk_executor(monkeypatch):
    query='Предложи план матрицы графиков H для IRT BOU MOS NVS SPG YAK KHB MGD VLA PET. Пока ничего не делай.'
    assert ac.is_bulk_query(query)
    monkeypatch.setattr(ga,'run_agent_with_planner',lambda *a,**k:pytest.fail('Unrequested execution'))
    backend=Backend(wire('discuss','plan',['create_task_matrix']),
                    {'choices':[{'message':{'content':'Сначала согласуем интервалы и показатели.'}}]})
    result=ga.make_chat_handler(ac.LlamaBrain(backend))(query,[],[],ac.FrameStore())
    messages,_,_,_,_,_=ga.split_result(result)
    assert messages[-1]['content'].startswith('Сначала согласуем')


def test_selected_metric_requires_its_operator_without_mutating_raw_arguments():
    value=wire('compute','interpretation',metrics=['gic'])
    before=copy.deepcopy(value)
    decoded=rp.decode_arguments(value)
    assert decoded['tools']==['apply_physics_formula']
    assert value==before
    request=SessionMemory().prepare('Вычисли физическую метрику.',ac.FrameStore())
    request.apply_intent(decoded)
    assert 'apply_physics_formula' in request.required_operations
    assert any('apply_physics_formula' in item for item in request.missing([],ac.FrameStore()))


def test_maximum_declared_operation_list_still_allows_required_metric_operators():
    tools=['fetch_observatory_data','fetch_solar_wind','fetch_geomagnetic_indices',
           'calculate_derived_components','get_statistics','calculate_baseline',
           'calculate_derived_math','plot_components']
    value=wire('compute','interpretation',tools,metrics=['gic','eej_index'])
    result=rp.plan_request(ac.LlamaBrain(Backend(value)),'Выполни согласованный расчёт.','',ac.parse_tool_calls)
    assert result['status']=='ok',result
    assert len(result['tools'])==10
    assert {'apply_physics_formula','calculate_eej_index'}<=set(result['tools'])


@pytest.mark.parametrize('key',['plan','baselines','critique'])
def test_real_original_requests_get_metric_meanings_in_messages_within_existing_budget(key):
    # A grammar's enum restricts tokens but is not a semantic catalogue for Qwen.
    # Verify what the real adapter sends; this is not a live-model correctness test.
    catalog=json.loads((ROOT/'docs/qa/SCIENTIFIC_ADVISOR_CASES.json').read_text(encoding='utf-8'))
    backend=Backend(wire('discuss','plan'))
    result=rp.plan_request(ac.LlamaBrain(backend),catalog['prompts'][key])
    assert result['status']=='ok' and len(backend.calls)==1
    request=backend.calls[0]
    system=request['messages'][0]['content']
    assert set(rp.METRIC_PURPOSES)==set(rp.METRICS)
    assert all(f'{name}: {purpose}' in system for name,purpose in rp.METRIC_PURPOSES.items())
    assert json.loads(request['messages'][1]['content'])['request']==catalog['prompts'][key]
    assert len(system)<3100
    assert sum(len(m['content']) for m in request['messages'])+len(json.dumps(rp.ARGUMENT_SCHEMA))<=rp.MAX_INPUT_CHARS
    assert request['max_tokens']==360
    assert request['response_format']['schema']==rp.ARGUMENT_SCHEMA
    assert 'tools' not in request
