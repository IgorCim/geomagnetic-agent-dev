"""Inspect the real mini-planner boundary without editing a loaded agent.

Compatible with the existing 113a719 checkout. Three original prompts, one
production planning call per prompt, no scientific handlers or model loading.
"""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import copy
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import traceback
import zipfile


class BackendRecorder:
    """Delegate unchanged to the loaded backend, retaining transport evidence."""

    def __init__(self, backend):
        self.backend = backend
        self.calls = []

    def create_chat_completion(self, **kwargs):
        record = {'request': copy.deepcopy(kwargs)}
        self.calls.append(record)
        try:
            response = self.backend.create_chat_completion(**kwargs)
        except Exception as exc:
            record['backend_exception'] = {'type': type(exc).__name__, 'message': str(exc)}
            raise
        record['response'] = copy.deepcopy(response)
        return response


def inspect_call(record, ac, rp):
    """Report parse and validation errors separately, never repair the reply."""
    if record.get('backend_exception'):
        return {'stage': 'backend', **record['backend_exception']}
    response = record.get('response') or {}
    choices = response.get('choices') or []
    if not choices or not isinstance(choices[0].get('message'), dict):
        return {'stage': 'transport', 'message': 'Missing choices[0].message'}
    message = choices[0]['message']
    if record['request'].get('response_format', {}).get('type') == 'json_object':
        result = {'finish_reason': choices[0].get('finish_reason'), 'usage': response.get('usage')}
        try:
            value = json.loads(message.get('content') or '')
        except Exception as exc:
            return {**result, 'stage':'json', 'type':type(exc).__name__, 'message':str(exc)}
        try:
            decoded = rp.decode_arguments(value)
            envelope = json.loads(record['request']['messages'][1]['content'])
            aliases = dict(__import__('re').findall(r'([A-Z]{3})=([^,]+)', ac.STATION_HINT))
            validated = rp.validate(decoded, envelope['request'], envelope['session'], aliases)
        except Exception as exc:
            return {**result, 'stage':'validation', 'arguments':value,
                    'type':type(exc).__name__, 'message':str(exc)}
        return {**result, 'stage':'accepted', 'arguments':value, 'validated':validated}
    reply = {'content': message.get('content') or '', 'tool_calls': message.get('tool_calls')}
    calls, leftover = ac.parse_tool_calls(reply, debug=False)
    result = {'finish_reason': choices[0].get('finish_reason'), 'usage': response.get('usage'),
              'parsed_calls': calls, 'leftover': leftover}
    if len(calls) != 1:
        return {**result, 'stage': 'parse', 'message': f'Expected one call, got {len(calls)}'}
    call = calls[0]
    if call.get('name') != 'plan_request' or call.get('malformed'):
        return {**result, 'stage': 'parse', 'message': 'Expected a valid plan_request call'}
    try:
        envelope = json.loads(record['request']['messages'][1]['content'])
        aliases = dict(__import__('re').findall(r'([A-Z]{3})=([^,]+)', ac.STATION_HINT))
        validated = rp.validate(call['arguments'], envelope['request'], envelope['session'], aliases)
    except Exception as exc:
        return {**result, 'stage': 'validation', 'type': type(exc).__name__, 'message': str(exc)}
    return {**result, 'stage': 'accepted', 'validated': validated}



def check_route(key, decision, inspection, backend_calls):
    """Check these original requests, including unrequested scientific work.

    An accepted JSON object is a protocol result, not semantic correctness or
    scientific acceptance. Never repair the decision to make this check pass.
    """
    expected = {'plan':'plan', 'baselines':'analyze', 'critique':'explain'}[key]
    errors = []
    if decision.get('status') != 'ok' or decision.get('mode') != expected:
        errors.append('wrong status/mode')
    if backend_calls != 1 or len(inspection) != 1 or inspection[0].get('stage') != 'accepted':
        errors.append('expected one accepted backend response')
    if key in ('plan','critique'):
        if decision.get('tools') or decision.get('metrics'):
            errors.append('discussion must not authorize execution')
    if key == 'baselines':
        allowed = {'calculate_baseline','calculate_derived_math','get_statistics',
                   'fetch_observatory_data','calculate_derived_components'}
        if 'calculate_baseline' not in decision.get('tools', []):
            errors.append('missing calculate_baseline')
        extra_tools = set(decision.get('tools', [])) - allowed
        if extra_tools:
            errors.append('unrequested operations: ' + ', '.join(sorted(extra_tools)))
        if decision.get('metrics'):
            errors.append('unrequested physical metrics: ' + ', '.join(decision['metrics']))
        if decision.get('stations') != ['irt'] or decision.get('dates') != ['2024-09-12']:
            errors.append('wrong station/date for baseline comparison')
    return {'expected_mode':expected, 'passed':not errors, 'errors':errors}


def _adapter(brain, ac):
    recorder = BackendRecorder(brain._llm)
    diagnostic = ac.LlamaBrain(
        recorder, max_tokens=brain._max_tokens,
        temperature=brain._temperature, n_ctx=brain._n_ctx,
    )
    return diagnostic, recorder


def _run(brain, *, scope, planner=None):
    import agent_core as ac
    import request_planner as rp
    import project_store as ps
    if planner is not None:
        rp = planner

    if not isinstance(brain, ac.LlamaBrain) or isinstance(brain, ac.ScriptedBrain):
        raise TypeError('Нужен уже загруженный настоящий brain=LlamaBrain.')
    if not rp.enabled(brain):
        raise RuntimeError('Мини планировщик выключен. GEOMAG_REQUEST_PLANNER должен быть включён.')
    repo = Path(ac.__file__).resolve().parent
    helper = runpy.run_path(str(repo / 'tools/run_chat_acceptance.py'))
    helper['_verify_frozen_source'](repo)
    catalog = json.loads((repo / 'docs/qa/SCIENTIFIC_ADVISOR_CASES.json').read_text(encoding='utf-8'))
    transcript = (repo / catalog['source_file']).read_text(encoding='utf-8')
    if hashlib.sha256(transcript.encode('utf-8')).hexdigest() != catalog['source_sha256_normalized_utf8']:
        raise RuntimeError('Изменена историческая выгрузка запросов.')
    keys = ['plan', 'baselines', 'critique']
    prompts = [(key, catalog['prompts'][key]) for key in keys]
    if not all(prompt in transcript for _, prompt in prompts):
        raise RuntimeError('Изменены исходные формулировки.')

    parent = repo / 'verification'
    parent.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix='request_planner_probe_', dir=parent))
    redact = helper['Redactor']([os.environ.get('SUPERMAG_LOGON', '')])
    hashes = {name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
              for name in ['agent_core.py', 'request_planner.py', 'chat_session.py', 'gradio_app.py']}
    diagnostic, recorder = _adapter(brain, ac)
    original_backend = brain._llm
    report = {'scope': scope, 'backend_type': type(original_backend).__module__ + '.' + type(original_backend).__name__,
              'started_utc': datetime.now(timezone.utc).isoformat(), 'revision': ac.code_revision(),
              'n_ctx': brain._n_ctx, 'max_plan_tokens': rp.MAX_OUTPUT_TOKENS,
              'temperature': brain._temperature, 'source_sha256': hashes,
              'planner_component': {'origin':getattr(rp,'__file__',None),
                                    'sha256':getattr(rp,'__source_sha256__',None)},
              'status': 'RUNNING', 'cases': [], 'scientific_acceptance': 'NOT_EVALUATED'}

    def save():
        (run_dir / 'report.json').write_text(redact(json.dumps(report, ensure_ascii=False, indent=2, default=str)), encoding='utf-8')

    save()
    console = sys.stdout
    with (run_dir / 'console.log').open('w', encoding='utf-8') as logfile:
        tee = helper['Tee'](console, logfile, redact)
        try:
            with redirect_stdout(tee), redirect_stderr(tee):
                print('Диагностика: 3 исходных запроса; только мини планировщик.', flush=True)
                for key, prompt in prompts:
                    store = ac.FrameStore(workspace=ps.Workspace(root=run_dir/key/'projects'))
                    row = {'id': key, 'prompt': prompt}
                    report['cases'].append(row)
                    before = len(recorder.calls)
                    try:
                        print(f'\n=== {key} ===\n{prompt}', flush=True)
                        if planner is None:
                            row['production_decision'] = ac.understand_request(prompt, diagnostic, store, verbose=True)
                        else:
                            aliases = dict(__import__('re').findall(r'([A-Z]{3})=([^,]+)', ac.STATION_HINT))
                            row['production_decision'] = rp.plan_request(
                                diagnostic, prompt, store.conversation.context(), ac.parse_tool_calls, aliases)
                            print('[REQUEST PLANNER]', json.dumps(row['production_decision'], ensure_ascii=False), flush=True)
                        captured = recorder.calls[before:]
                        row['backend_calls'] = captured
                        row['inspection'] = [inspect_call(call, ac, rp) for call in captured]
                        print('RAW BACKEND RESPONSE:', json.dumps([call.get('response') or call.get('backend_exception') for call in captured], ensure_ascii=False, default=str), flush=True)
                        print('PARSE / VALIDATION:', json.dumps(row['inspection'], ensure_ascii=False, default=str), flush=True)
                        if hasattr(rp, 'ARGUMENT_SCHEMA'):
                            row['routing_check'] = check_route(
                                key, row['production_decision'] or {}, row['inspection'], len(captured))
                            print('ROUTING CHECK:', json.dumps(row['routing_check'], ensure_ascii=False), flush=True)
                        if store.names():
                            raise RuntimeError('Diagnostic unexpectedly produced data frames')
                    except Exception:
                        row['exception'] = traceback.format_exc()
                        print(row['exception'], flush=True)
                    finally:
                        save()
                report['status'] = 'RECORDED'
        except BaseException:
            report['status'] = 'INTERRUPTED'
            raise
        finally:
            tee.finish()
            report['changed_source_files'] = [name for name, digest in hashes.items()
                                              if hashlib.sha256((repo / name).read_bytes()).hexdigest() != digest]
            report['original_backend_unchanged'] = brain._llm is original_backend
            save()
            archive = run_dir.with_suffix('.zip')
            with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
                for path in sorted(run_dir.iterdir()):
                    if path.is_file():
                        bundle.write(path, path.name)
            print('Диагностика сохранена:', run_dir, flush=True)
            print('ZIP диагностики:', archive, flush=True)
    return {'run_dir': str(run_dir), 'archive': str(archive), 'report': report}


def run(brain):
    """Public entry: inspect the caller's real, already-loaded model."""
    return _run(brain, scope='Real loaded model; production mini-planner only. No scientific execution.')


def preview(brain, planner):
    """Exercise a separately loaded new component; never patch installed modules."""
    return _run(brain, scope='New structured planner component preview; installed executor unchanged.', planner=planner)
