"""Native regression for the real 3168e77 grammar failure; no model inference."""
import copy
import json
from pathlib import Path
import runpy

import pytest
import response_review as rr

ROOT=Path(__file__).resolve().parent
CHECK=runpy.run_path(str(ROOT/'tools/check_review_grammar.py'))


@pytest.mark.parametrize('optional',[[],['calculate_baseline'],[
    'fetch_observatory_data','fetch_solar_wind','fetch_geomagnetic_indices','fetch_supermag_data',
    'calculate_baseline','calculate_derived_math','evaluate_custom_formula','calculate_derived_components']])
def test_actual_native_parser_accepts_current_schema(optional):
    result=CHECK['probe'](rr.schema(optional))
    if result.get('runtime_unavailable'):
        pytest.skip('Native llama.cpp unavailable: '+result.get('error',''))
    assert result.get('accepted') and result.get('returncode')==0,result
    assert result['grammar_bytes']<3000


def test_recorded_schema_retains_the_large_repetition_that_failed_in_kaggle():
    old=json.loads((ROOT/'tests/fixtures/review_schema_3168e77.json').read_text(encoding='utf-8'))
    assert old['source_revision'].startswith('3168e77')
    assert old['schema']['properties']['text']['maxLength']==4000
    new=rr.schema([])
    expected=copy.deepcopy(old['schema'])
    expected['properties']['text'].pop('maxLength')
    expected['properties']['remaining']['items'].pop('maxLength')
    assert new==expected
    assert rr.MAX_OUTPUT_TOKENS==900 and rr.MAX_INPUT_CHARS==8000


@pytest.mark.parametrize('size,expected',[(4000,'ok'),(4001,'error')])
def test_text_length_limit_is_still_enforced_after_decoding(tmp_path,size,expected):
    from test_response_review import ReviewedBackend,reviewed,context
    import agent_core as ac
    store=context(tmp_path)
    result=rr.review(ac.LlamaBrain(ReviewedBackend(None,review=reviewed(text='x'*size))),
                     'Объясни ограничения.',store,[],{'mode':'explain','transport':'json_schema','tools':[]})
    assert result['status']==expected
    if expected=='error':assert 'invalid explanation' in result['reason']


@pytest.mark.parametrize('size,expected',[(180,'ok'),(181,'error')])
def test_remaining_goal_length_limit_is_still_enforced(tmp_path,size,expected):
    from test_response_review import ReviewedBackend,reviewed,context
    import agent_core as ac
    store=context(tmp_path)
    value=reviewed(complete=False,remaining=['x'*size])
    result=rr.review(ac.LlamaBrain(ReviewedBackend(None,review=value)),
                     'Что ещё неизвестно?',store,[],{'mode':'explain','transport':'json_schema','tools':[]})
    assert result['status']==expected
    if expected=='error':assert 'invalid remaining goals' in result['reason']


def test_live_runner_refuses_to_start_messages_when_native_grammar_fails(monkeypatch):
    import agent_core as ac
    runner=runpy.run_path(str(ROOT/'tools/run_dual_stage_review.py'))
    class Backend:
        def create_chat_completion(self,**kwargs):pytest.fail('Model called before native preflight passed')
    def load(path):
        assert Path(path).name=='check_review_grammar.py','Acceptance runner started despite bad grammar'
        return {'run':lambda:{'status':'FAIL'}}
    monkeypatch.setattr(runpy,'run_path',load)
    with pytest.raises(RuntimeError,match='16 сообщений не начат'):
        runner['run'](ac.LlamaBrain(Backend()))


def test_native_preflight_evidence_is_saved_in_acceptance_archive(tmp_path,monkeypatch):
    import agent_core as ac
    import zipfile
    runner=runpy.run_path(str(ROOT/'tools/run_dual_stage_review.py'))
    native={'status':'PASS','scope':'TEST FIXTURE: not a native run','current_schemas':[{'accepted':True}]*3}
    archive=tmp_path/'evidence.zip'
    with zipfile.ZipFile(archive,'w'):pass
    result={'run_dir':str(tmp_path),'archive':str(archive),'report':{'phases':[]}}
    def load(path):
        if Path(path).name=='check_review_grammar.py':return {'run':lambda:native}
        if Path(path).name=='run_chat_acceptance.py':return {'run_acceptance':lambda *args,**kw:result}
        pytest.fail('Unexpected runner')
    monkeypatch.setattr(runpy,'run_path',load)
    monkeypatch.setitem(runner['run'].__globals__,'build_plan',lambda:[])
    class Backend:
        def create_chat_completion(self,**kwargs):pytest.fail('Unexpected inference in archive fixture')
    output=runner['run'](ac.LlamaBrain(Backend()))
    assert output is result
    with zipfile.ZipFile(archive) as bundle:
        assert json.loads(bundle.read('native_review_grammar.json'))==native
        assert 'dual_review.json' in bundle.namelist()
