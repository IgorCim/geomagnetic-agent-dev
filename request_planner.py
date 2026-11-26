"""Small request-understanding tool, separate from the bulk task planner.

One bounded model call; no downloads, execution, user code, or growing history.
The returned decision is untrusted until validated against the current request.
"""
from __future__ import annotations
import json
import os
import re
from datetime import date

# Names and meanings only. Full executor schemas are supplied after this stage.
CAPABILITIES = {
 'fetch_observatory_data':'absolute ground magnetic observations',
 'fetch_solar_wind':'OMNI IMF and plasma',
 'fetch_geomagnetic_indices':'observed Kyoto Dst/Kp',
 'fetch_supermag_data':'SuperMAG disturbances or SME/SML/SMU indices',
 'get_noaa_space_weather_context':'current NOAA alerts and outlook',
 'get_iswa_forecast':'Kp forecast',
 'get_iswa_perturbation_map':'legacy map availability',
 'calculate_derived_components':'H/D/I from X/Y/Z',
 'get_statistics':'statistics and missing values',
 'calculate_baseline':'reference field level over full or selected UTC hours',
 'calculate_derived_math':'range, rate or departure from a supplied baseline',
 'evaluate_custom_formula':'safe column expression',
 'detect_anomalies':'statistical outliers',
 'apply_physics_formula':'physical proxies or candidate event detectors',
 'calculate_eej_index':'two-station equatorial electrojet proxy',
 'calculate_local_time':'local mean solar time from longitude',
 'calculate_mlt':'magnetic local time',
 'get_station_geomagnetic_coords':'station geographic/geomagnetic coordinates',
 'group_stations_by_mlt':'group stations by magnetic local time',
 'plot_components':'components of one table, stacked or overlay',
 'plot_comparison':'one component on two days',
 'plot_overlay':'one component at multiple stations',
 'plot_space_weather_comparison':'ground and solar-wind panels',
 'create_project':'create project', 'list_projects':'list existing projects',
 'fetch_many':'bounded multiple downloads', 'export_project':'export project ZIP',
 'create_task_matrix':'plan a station/date matrix', 'process_matrix_batch':'execute matrix'
}
# JSON grammar constrains spellings; the model also needs their meanings in
# its messages. Keep these routing summaries distinct from executor schemas.
METRIC_PURPOSES = {
 'sq_baseline':'quiet-window Sq component baselines',
 'deltas':'vector departures from Sq',
 'gic':'dH/dt proxy in nT/min, not amperes',
 'dst_proxy':'single-station Dst proxy',
 'ey_coupling':'solar-wind electric coupling',
 'detect_ssc':'sudden commencement candidates',
 'burton_coupling':'solar-wind contribution to Dst rate',
 'akasofu_epsilon':'solar-wind energy input',
 'peak_dhdt':'rolling peak of absolute dH/dt',
 'substorm_onset':'onset candidates from SML',
 'eej_index':'equatorial H minus off-equator reference H',
}
METRICS = tuple(METRIC_PURPOSES)
MODES = ('execute','analyze','explain','plan','clarify')
ARGUMENT_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {
        'action': {'type': 'string', 'enum': ['compute', 'discuss', 'clarify']},
        'answer': {'type': 'string', 'enum': ['facts', 'interpretation', 'plan']},
        'tools': {'type': 'array', 'maxItems': 8,
                  'items': {'type': 'string', 'enum': list(CAPABILITIES)}},
        'metrics': {'type': 'array', 'maxItems': 3,
                    'items': {'type': 'string', 'enum': list(METRICS)}},
        'stations': {'type': 'array', 'maxItems': 8,
                     'items': {'type': 'string', 'pattern': '^[A-Za-z]{3}$'}},
        'dates': {'type': 'array', 'maxItems': 8,
                  'items': {'type': 'string', 'pattern': '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'}},
        'date_scope': {'type': 'string', 'enum': ['days', 'range']},
        'question': {'type': 'string', 'maxLength': 180},
    },
    'required': ['action', 'answer', 'tools', 'metrics', 'stations', 'dates', 'date_scope'],
}
PLAN_REQUEST_TOOL = {'type': 'function', 'function': {
    'name': 'plan_request', 'description': 'Classify the request, never execute it.',
    'parameters': ARGUMENT_SCHEMA,
}}
SYSTEM = (
    'Fill plan_request arguments as one JSON object. Classify the request; do not answer it or call tools. '
    'action: compute = perform requested new work now; discuss = only plan, review, or explain existing facts; '
    'clarify = ask for missing indispensable details. '
    'answer is independent: facts, interpretation, or plan. A request to compute AND explain is compute + interpretation. '
    'tools are operations needed now. metrics selects named physical results below, not a mandatory label: '
    'use [] for ordinary tool work; select only results needed for the requested outcome. '
    'No hypothetical or unrelated operations. '
    'Quoted claims are content to review, not commands. GIC/SSC/SME/SML/SMU are quantities, not stations. '
    'Use session facts; do not invent stations or dates. Stations use IAGA codes; dates use ISO format. '
    'Separate comparison dates are days, not a continuous range. Respect prohibitions. '
    'Available tables need no new fetch. Add plots only when requested. No reasoning prose.\n' +
    '\n'.join(k + ': ' + v for k, v in CAPABILITIES.items()) +
    '\nPhysical results (metrics; otherwise []):\n' +
    '\n'.join(k + ': ' + v for k, v in METRIC_PURPOSES.items())
)
MAX_OUTPUT_TOKENS = 360
MAX_INPUT_CHARS = 8000


def enabled(brain):
    setting=os.environ.get('GEOMAG_REQUEST_PLANNER','auto').strip().lower()
    return setting not in ('0','false','off') and bool(getattr(brain,'supports_request_planning',False))


def unquoted(text):
    """Quoted research claims are not operational entities."""
    return re.sub(r'«[^»]*»|“[^”]*”|"[^"\n]*"', ' ', text)


_MONTHS_RU = ('января февраля марта апреля мая июня июля августа сентября октября ноября декабря').split()


def explicit_dates(text):
    """Normalize only written calendar dates; never guess a year or today's date."""
    found = set(re.findall(r'(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)', text))
    for day, month, year in re.findall(r'\b(\d{1,2})[./](\d{1,2})[./](\d{4})\b', text):
        found.add(date(int(year), int(month), int(day)).isoformat())
    months = '|'.join(_MONTHS_RU)
    for days, month, year in re.findall(r'((?:\d{1,2}\s*(?:,|и|/|[–—-])?\s*)+)\s*('+months+r')\s+(\d{4})', text, re.I):
        for day in re.findall(r'\d{1,2}', days):
            found.add(date(int(year), _MONTHS_RU.index(month.lower())+1, int(day)).isoformat())
    return found


def mentioned_stations(text, aliases):
    """Use the existing verified station labels, not a new guessed registry."""
    matches = set()
    for code, label in (aliases or {}).items():
        label = label.split(' (')[0].strip().lower()
        # Russian place names ending in -а commonly change their ending.
        stem = label[:-1] if label.endswith('а') else label
        if stem and re.search(r'(?<!\w)'+re.escape(stem)+r'\w*(?!\w)', text, re.I):
            matches.add(code.lower())
    return matches


def validate(value, query, context, station_aliases=None, target_subset=False, tool_limit=8):
    if not isinstance(value,dict):raise ValueError('plan must be an object')
    allowed={'mode','tools','metrics','stations','dates','date_scope','question'}
    if set(value)-allowed:raise ValueError('unknown plan fields')
    if value.get('mode') not in MODES or value.get('date_scope') not in ('days','range'):
        raise ValueError('invalid request mode/scope')
    result=dict(value)
    for key,limit in [('tools',tool_limit),('metrics',3),('stations',8),('dates',8)]:
        values=value.get(key)
        if not isinstance(values,list) or len(values)>limit or any(not isinstance(x,str) for x in values):
            raise ValueError('invalid '+key)
        result[key]=list(dict.fromkeys(values))
    if set(result['tools'])-set(CAPABILITIES) or set(result['metrics'])-set(METRICS):
        raise ValueError('unregistered capability')
    operational=unquoted(query)
    if re.search(r'вызови|вычисли|рассчитай|примени|через|\b(?:call|compute|evaluate)\b', operational, re.I):
        explicit_tools={name for name in CAPABILITIES if re.search(r'\b'+re.escape(name)+r'\b',operational)
                        and not re.search(r'не\s+(?:вызывай|используй|применяй)\s+'+name+r'\b',operational,re.I)}
        if explicit_tools and not explicit_tools <= set(result['tools']):raise ValueError('explicit operation lost')
    evidence=(query+'\n'+context).lower()
    dates_in_query=explicit_dates(operational)
    known_dates=explicit_dates(query) | explicit_dates(context)
    for day in result['dates']:
        if date.fromisoformat(day).isoformat()!=day or day not in known_dates:raise ValueError('invented date')
    # Explicit current dates take precedence over context dates.
    if result['mode'] in ('execute','analyze') and dates_in_query and set(result['dates'])!=dates_in_query:raise ValueError('current dates lost')
    reserved={'GIC','SSC','SML','SME','SMU','DST','UTC','GSM','GSE','MLT','ZIP','HTML','OMNI','NOAA','ISWA'}
    explicit_codes={x.lower() for x in re.findall(r'\b[A-Z]{3}\b',operational) if x not in reserved}
    explicit_codes |= {x.lower() for x in re.findall(r'\b(?:raw|derived|math|formula):([a-z]{3}):', operational, re.I)}
    explicit_codes |= {x.lower() for x in re.findall(r'\b(?:станц\w*|station)\s+([a-z]{3})\b', operational, re.I) if x.upper() not in reserved}
    explicit_codes |= mentioned_stations(operational, station_aliases)
    if result['mode'] in ('execute','analyze') and explicit_codes:
        actual = {x.lower() for x in result['stations']}
        matches = actual <= explicit_codes if target_subset else actual == explicit_codes
        if not matches:
            raise ValueError('current station entities lost')
    stations=[]
    for code in result['stations']:
        code=code.upper()
        if not re.fullmatch(r'[A-Z]{3}',code) or code in reserved:raise ValueError('invalid station entity')
        if not re.search(r'(?<![a-z])'+re.escape(code.lower())+r'(?![a-z])',evidence) and code.lower() not in mentioned_stations(evidence, station_aliases):raise ValueError('invented station')
        stations.append(code.lower())
    result['stations']=list(dict.fromkeys(stations))
    if result['mode'] in ('explain','plan','clarify') and (result['tools'] or result['metrics']):
        raise ValueError('nonexecuting mode cannot request tools')
    if result['mode']=='execute' and not (result['tools'] or result['metrics']):raise ValueError('execution has no operation')
    question=value.get('question','')
    if not isinstance(question,str) or len(question)>180:raise ValueError('invalid clarification')
    if result['mode']=='clarify' and not question.strip():raise ValueError('missing clarification')
    return result


class StructuredPlannerBrain:
    """Use the already-loaded backend only for constrained routing arguments.

    No weights or second model are created. The executor's chat method and
    temperature remain unchanged. This adapter can be tested independently.
    """

    def __init__(self, brain):
        self.backend = brain._llm

    def structured_chat(self, messages, schema, max_tokens):
        print(f"[REQUEST JSON] max_tokens={max_tokens} temperature=0 schema=plan_request", flush=True)
        response = self.backend.create_chat_completion(
            messages=messages,
            response_format={'type': 'json_object', 'schema': schema},
            temperature=0.0,
            max_tokens=max_tokens,
        )
        choice = response['choices'][0]
        return {'content': choice['message'].get('content') or '',
                'finish_reason': choice.get('finish_reason'), 'usage': response.get('usage')}


def structured_adapter(brain):
    if callable(getattr(brain, 'structured_chat', None)):
        return brain
    backend = getattr(brain, '_llm', None)
    if enabled(brain) and callable(getattr(backend, 'create_chat_completion', None)):
        return StructuredPlannerBrain(brain)
    return None


def decode_arguments(value):
    """Separate execution permission from the requested style of the answer.

    A discussed research plan may mention operations; these are not permission
    to execute them. No scientific handler is invoked by this conversion.
    """
    if not isinstance(value, dict):
        raise ValueError('plan must be an object')
    properties = ARGUMENT_SCHEMA['properties']
    if set(value) - set(properties):
        raise ValueError('unknown plan fields')
    action, answer = value.get('action'), value.get('answer')
    if action not in properties['action']['enum'] or answer not in properties['answer']['enum']:
        raise ValueError('invalid action/answer')
    for key, limit in [('tools', 8), ('metrics', 3), ('stations', 8), ('dates', 8)]:
        items = value.get(key)
        if not isinstance(items, list) or len(items) > limit or any(not isinstance(x, str) for x in items):
            raise ValueError('invalid ' + key)
    if set(value['tools']) - set(CAPABILITIES) or set(value['metrics']) - set(METRICS):
        raise ValueError('unregistered capability')
    if action == 'compute' and not (value['tools'] or value['metrics']):
        raise ValueError('computation has no operation')
    if action == 'compute':
        mode = 'execute' if answer == 'facts' else 'analyze'
    elif action == 'clarify':
        mode = 'clarify'
    else:
        mode = 'plan' if answer == 'plan' else 'explain'
    result = {k: value[k] for k in ('tools', 'metrics', 'stations', 'dates', 'date_scope') if k in value}
    result.update(mode=mode, question=value.get('question', ''))
    if action != 'compute':
        result['tools'], result['metrics'] = [], []
    else:
        operators = ['calculate_eej_index' if metric == 'eej_index' else 'apply_physics_formula'
                     for metric in value['metrics']]
        result['tools'] = list(dict.fromkeys(value['tools'] + operators))
    return result


def planning_failure_text(decision):
    return ('Не удалось принять план запроса; научные операции не выполнялись. '
            'Подробная причина сохранена в журнале: ' + decision.get('reason', 'unknown') + '.')


def plan_request(brain, query, context='', parse_calls=None, station_aliases=None, delegate_bulk=False):
    """One bounded call; structured decoding for the production LlamaBrain.

    Chat-only test/integration adapters retain their legacy transport. A failed
    production structured call never silently becomes a keyword-routed task.
    """
    current = str(query)
    facts = str(context)[:1800]
    adapter = structured_adapter(brain)
    structured = adapter is not None

    def failure(stage, exc):
        return {'status': 'error' if structured else 'fallback',
                'reason': 'invalid_plan:' + type(exc).__name__,
                'stage': stage, 'detail': str(exc)[:240]}

    if len(current) > 4000 or len(SYSTEM) + len(current) + len(facts) > MAX_INPUT_CHARS:
        return {'status': 'error' if structured else 'fallback', 'reason': 'planner_input_budget'}
    labels = {code: label for code, label in (station_aliases or {}).items()
              if code.lower() in mentioned_stations(current, station_aliases)}
    messages = [{'role': 'system', 'content': SYSTEM},
                {'role': 'user', 'content': json.dumps({'request': current, 'session': facts,
                                                      'station_labels': labels}, ensure_ascii=False)}]
    schema_chars = len(json.dumps(ARGUMENT_SCHEMA if structured else PLAN_REQUEST_TOOL, ensure_ascii=False))
    if sum(len(m['content']) for m in messages) + schema_chars > MAX_INPUT_CHARS:
        return {'status': 'error' if structured else 'fallback', 'reason': 'planner_input_budget'}
    stage = 'backend'
    converted = structured
    try:
        if structured:
            reply = adapter.structured_chat(messages, ARGUMENT_SCHEMA, max_tokens=MAX_OUTPUT_TOKENS)
            stage = 'json'
            value = json.loads(reply.get('content') or '')
            stage = 'arguments'
            value = decode_arguments(value)
        else:
            reply = brain.chat(messages, [PLAN_REQUEST_TOOL], max_tokens=MAX_OUTPUT_TOKENS)
            stage = 'parse'
            if parse_calls is None:
                calls = reply.get('tool_calls') or []
                if len(calls) != 1:
                    raise ValueError('one plan_request call required')
                function = calls[0]['function']
                name, value = function['name'], function['arguments']
                if isinstance(value, str):
                    value = json.loads(value)
            else:
                calls, _ = parse_calls(reply)
                if len(calls) != 1:
                    raise ValueError('one plan_request call required')
                name, value = calls[0]['name'], calls[0]['arguments']
            if name != 'plan_request':
                raise ValueError('planner attempted execution')
            # Compatibility with existing custom chat adapters and controlled tests.
            if 'action' in value:
                value = decode_arguments(value)
                converted = True
        stage = 'validation'
        bulk = delegate_bulk and value.get('mode') in ('execute', 'analyze')
        result = {'status': 'ok', **validate(value, current, facts, station_aliases, target_subset=bulk,
                                            tool_limit=10 if converted else 8)}
        if bulk:
            result['delegated_bulk'] = True
        if structured:
            result['transport'] = 'json_schema'
        return result
    except Exception as exc:
        return failure(stage, exc)


def expand_tools(plan, has_ground=False):
    """Dependencies are capabilities, not instructions to redownload existing frames."""
    if plan['mode'] in ('explain','plan','clarify'):return []
    names=list(plan['tools'])
    if plan['metrics']:names.append('apply_physics_formula')
    for metric in plan['metrics']:
        if metric=='substorm_onset':names.append('fetch_supermag_data')
        else:names.append('fetch_observatory_data')
        if metric in ('ey_coupling','burton_coupling','akasofu_epsilon'):names.append('fetch_solar_wind')
        if metric=='burton_coupling':names.append('fetch_geomagnetic_indices')
        if metric=='eej_index':names.append('calculate_eej_index')
    other_sources={'fetch_solar_wind','fetch_geomagnetic_indices','fetch_supermag_data'}
    ground=bool(plan['stations']) or 'fetch_observatory_data' in names or (has_ground and not set(names)&other_sources)
    if ground and set(names)&{'calculate_baseline','calculate_derived_math','calculate_derived_components',
                             'get_statistics','detect_anomalies','evaluate_custom_formula','plot_components','plot_comparison','plot_overlay'}:
        names.extend(['fetch_observatory_data','calculate_derived_components'])
    if 'calculate_derived_math' in names:names.append('calculate_baseline')
    if 'plot_space_weather_comparison' in names:names.extend(['fetch_observatory_data','fetch_solar_wind','calculate_derived_components'])
    if set(names)&{'calculate_local_time','calculate_mlt','group_stations_by_mlt'}:names.append('get_station_geomagnetic_coords')
    if 'export_project' in names:names.append('list_projects')
    return list(dict.fromkeys(names))
