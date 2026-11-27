"""Recorded wire regression: one operation selection, never a guessed synonym map."""
import copy
import json
from pathlib import Path
import pytest
import request_planner as rp
from test_structured_request_planner import wire, Backend
import agent_core as ac

ROOT=Path(__file__).resolve().parent
RECORDED=json.loads((ROOT/'tests/fixtures/planner_required_failure_1e716a1.json').read_text(encoding='utf-8'))['cases']


def test_live_contract_failures_remain_recorded_and_are_not_relabelled_pass():
    bad=[r for r in RECORDED if r['request_plan'].get('detail')=='required outcome was not selected']
    assert len(bad)==9
    for row in bad:
        with pytest.raises(ValueError,match='required outcome was not selected'):
            rp._decode_legacy_arguments(row['raw_arguments'],require_outcomes=True)
    assert next(r for r in RECORDED if r['id']=='LANG-01')['raw_arguments']['required']==['min','max','median','plot']


def test_nine_old_replies_cannot_silently_bypass_the_current_wire_schema():
    for row in RECORDED:
        if row['request_plan'].get('detail')!='required outcome was not selected':continue
        result=rp.plan_request(ac.LlamaBrain(Backend(row['raw_arguments'])),row['prompt'])
        assert result['status']=='error' and result['detail']=='missing work selection'
        assert result['raw_response']


@pytest.mark.parametrize('row',[r for r in RECORDED if r['request_plan'].get('detail')=='required outcome was not selected'],ids=lambda r:r['id'])
def test_the_current_grammar_excludes_invented_goal_identifiers(row):
    ids=rp.ARGUMENT_SCHEMA['properties']['work']['items']['properties']['id']['enum']
    assert set(ids)==set(rp.CAPABILITIES)|set(rp.METRICS)
    assert 'required' not in rp.ARGUMENT_SCHEMA['properties']
    assert 'tools' not in rp.ARGUMENT_SCHEMA['properties'] and 'metrics' not in rp.ARGUMENT_SCHEMA['properties']
    for old in row['raw_arguments']['required']:
        if old not in ids:
            invalid=wire('compute','facts',['get_statistics']);invalid['work'][0]['id']=old
            with pytest.raises(ValueError,match='unregistered capability'):rp.decode_arguments(invalid,require_outcomes=True)


@pytest.mark.parametrize('item',[{'id':'get_statistics','required':'min'},
                                 {'id':'get_statistics','required':1},
                                 {'id':'get_statistics','required':True,'goal':'mean'},
                                 {'required':True}])
def test_operation_role_is_a_boolean_not_a_second_identifier(item):
    value=wire('compute','facts',['get_statistics']);value['work']=[item]
    with pytest.raises(ValueError,match='invalid work item'):rp.decode_arguments(value,require_outcomes=True)


def test_every_registered_operation_can_be_chosen_without_a_second_matching_list():
    for name in list(rp.CAPABILITIES)+list(rp.METRICS):
        value=wire('compute','facts');value['work']=[{'id':name,'required':True}]
        decoded=rp.decode_arguments(value,require_outcomes=True)
        assert decoded['required_operations']
        rp.validate(decoded,'Выполни выбранную операцию.','',tool_limit=10)


def test_duplicate_operation_kinds_preserve_required_work_without_guessing_target_names():
    value=wire('compute','facts');value['work']=[{'id':'get_statistics','required':False},{'id':'get_statistics','required':True}]
    original=copy.deepcopy(value)
    decoded=rp.decode_arguments(value,require_outcomes=True)
    assert decoded['tools']==decoded['required_operations']==['get_statistics']
    assert value==original


def test_failed_planning_keeps_user_targets_but_creates_no_observations(tmp_path):
    import project_store as ps
    store=ac.FrameStore(workspace=ps.Workspace(root=tmp_path/'projects'))
    bad=next(row for row in RECORDED if row['id']=='ADVISOR-07')['raw_arguments']
    result=ac.understand_request('Для IRT за 12 сентября 2024 года получи два опорных уровня H.',ac.LlamaBrain(Backend(bad)),store)
    assert result['status']=='error' and store.names()==[]
    assert store.conversation.first_data is None and not store.conversation.evidence
    context=store.conversation.context()
    assert '2024-09-12' in context and 'user_requested_targets_not_observations' in context
    backend=Backend(wire('compute','interpretation',['calculate_local_time'],stations=['IRT'],dates=['2024-09-12']))
    followup=ac.understand_request('Проверь, означает ли это UTC-окно ночь в Иркутске.',ac.LlamaBrain(backend),store)
    assert followup['status']=='ok'
    assert '2024-09-12' in json.loads(backend.calls[0]['messages'][1]['content'])['session']


def test_requested_targets_are_bounded_and_separate_from_successful_anchors_and_secrets():
    from chat_session import SessionMemory
    memory=SessionMemory()
    for day in range(1,20):memory.remember_request(f'IRT 2024-09-{day:02d}. login=secret-fixture')
    assert len(memory.requested_targets)==2
    assert not memory.ground and memory.first_data is None and memory.last_good is None
    assert not memory.evidence and 'secret-fixture' not in memory.context()
    assert len(memory.request_context())<=601
    assert not SessionMemory().requested_targets


@pytest.mark.parametrize('quantity',['GIC','SML','EEJ','IMF'])
def test_quantities_cannot_become_requested_station_targets(quantity):
    from chat_session import SessionMemory,station_in
    memory=SessionMemory();memory.remember_request(f'Получи {quantity} за 2024-09-12.')
    assert station_in(f'Объясни {quantity}.') is None
    assert not memory.requested_targets[0].get('stations')
