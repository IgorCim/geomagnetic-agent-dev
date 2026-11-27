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
 'calculate_baseline':'mean of one component, full period or UTC window',
 'calculate_derived_math':'max-min, rate, or component minus supplied scalar',
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
 'sq_baseline':'night medians X_sq/Y_sq/Z_sq; H_sq=hypot(X_sq,Y_sq)',
 'deltas':'dX=X-X_sq, dY=Y-Y_sq; dH=hypot(dX,dY)',
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
NON_STATION_CODES=frozenset({'OMNI','NOAA','ISWA','DST','IAGA','HTML','GSM','UTC','ZIP','APIX','GIC','SSC','SML','SME','SMU','MLT','GSE','EEJ','IMF'})
MODES = ('execute','analyze','explain','plan','clarify')
ARGUMENT_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {
        'action': {'type': 'string', 'enum': ['compute', 'discuss', 'clarify']},
        'answer': {'type': 'string', 'enum': ['facts', 'interpretation', 'plan']},
        'work': {'type':'array','maxItems':8,'items':{
            'type':'object','additionalProperties':False,
            'properties':{
                'id':{'type':'string','enum':list(CAPABILITIES)+list(METRICS)},
                'required':{'type':'boolean'},
            },
            'required':['id','required'],
        }},
        'stations': {'type': 'array', 'maxItems': 8,
                     'items': {'type': 'string', 'pattern': '^[A-Za-z]{3}$'}},
        'dates': {'type': 'array', 'maxItems': 8,
                  'items': {'type': 'string', 'pattern': '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'}},
        'date_scope': {'type': 'string', 'enum': ['days', 'range']},
        'question': {'type': 'string', 'maxLength': 180},
    },
    'required': ['action', 'answer', 'work', 'stations', 'dates', 'date_scope'],
}
PLAN_REQUEST_TOOL = {'type': 'function', 'function': {
    'name': 'plan_request', 'description': 'Classify the request, never execute it.',
    'parameters': ARGUMENT_SCHEMA,
}}
SYSTEM = (
    'JSON plan only. Decide: does the user want new evidence now, '
    'or only methods/review/explanation of existing evidence? compute obtains/calculates requested results; '
    'discuss only describes methods or interprets existing facts. Comparing observations is compute; proposing how is discuss. '
    'clarify only if indispensable user choices are missing, not when a tool can obtain data. '
    'answer: facts = numbers/files only; interpretation = explain, evaluate, qualify or choose; plan = methods. '
    'work lists operations from the catalogue: id plus required=true for user-requested results, false for optional helpers. '
    'Pick each operation once; code adds prerequisites. Omit unrelated metrics; discuss has work=[]. '
    'question only asks a missing choice; otherwise omit it. Never echo the request. '
    'Use existing tables and session facts; invent no targets. '
    'Quoted claims are text to review. GIC/SSC/SME/SML/SMU/EEJ are quantities. '
    'Station names come from station_labels; keep the code when its name is unverified. '
    'Separate comparison days from a range. Respect prohibitions; plot only if requested.\n' +
    '\n'.join(k + ': ' + v for k, v in CAPABILITIES.items()) +
    '\nPhysical results (metrics):\n' +
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
    for days, month, year in re.findall(r'(?<!\d)(\d{1,2}(?:\s*(?:или|либо|и|по|,|/|[–—-])\s*\d{1,2})*)\s*('+months+r')\s+(\d{4})', text, re.I):
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
    allowed={'mode','tools','metrics','stations','dates','date_scope','question','required_operations','required_metrics'}
    if set(value)-allowed:raise ValueError('unknown plan fields')
    if value.get('mode') not in MODES or value.get('date_scope') not in ('days','range'):
        raise ValueError('invalid request mode/scope')
    result=dict(value)
    for key,limit in [('tools',tool_limit),('metrics',8 if 'required_metrics' in value else 3),('stations',8),('dates',8)]:
        values=value.get(key)
        if not isinstance(values,list) or len(values)>limit or any(not isinstance(x,str) for x in values):
            raise ValueError('invalid '+key)
        result[key]=list(dict.fromkeys(values))
    if set(result['tools'])-set(CAPABILITIES) or set(result['metrics'])-set(METRICS):
        raise ValueError('unregistered capability')
    if 'required_operations' in value or 'required_metrics' in value:
        for key, available, limit in [('required_operations',result['tools'],10),
                                      ('required_metrics',result['metrics'],8)]:
            values=value.get(key)
            if (not isinstance(values,list) or len(values)>limit
                    or any(not isinstance(x,str) for x in values) or not set(values)<=set(available)):
                raise ValueError('invalid '+key)
            result[key]=list(dict.fromkeys(values))
    operational=unquoted(query)
    if re.search(r'вызови|вычисли|рассчитай|примени|через|\b(?:call|compute|evaluate)\b', operational, re.I):
        explicit_tools={name for name in CAPABILITIES if re.search(r'\b'+re.escape(name)+r'\b',operational)
                        and not re.search(r'не\s+(?:вызывай|используй|применяй)\s+'+name+r'\b',operational,re.I)}
        if explicit_tools and not explicit_tools <= set(result['tools']):raise ValueError('explicit operation lost')
        if 'required_operations' in result and not explicit_tools <= set(result['required_operations']):
            raise ValueError('explicit operation marked optional')
    evidence=(query+'\n'+context).lower()
    dates_in_query=explicit_dates(operational)
    known_dates=explicit_dates(query) | explicit_dates(context)
    for day in result['dates']:
        if date.fromisoformat(day).isoformat()!=day or day not in known_dates:raise ValueError('invented date')
    # Explicit current dates take precedence over context dates.
    if result['mode'] in ('execute','analyze') and dates_in_query and set(result['dates'])!=dates_in_query:raise ValueError('current dates lost')
    reserved=NON_STATION_CODES
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
    if result['mode']=='clarify':
        if not question.strip():raise ValueError('missing clarification')
        if re.sub(r'\W+','',question).casefold()==re.sub(r'\W+','',query).casefold():
            raise ValueError('clarification repeats the request')
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


def _decode_legacy_arguments(value, require_outcomes=False):
    """Separate execution permission from the requested style of the answer.

    A discussed research plan may mention operations; these are not permission
    to execute them. No scientific handler is invoked by this conversion.
    """
    if not isinstance(value, dict):
        raise ValueError('plan must be an object')
    properties = ARGUMENT_SCHEMA['properties']
    legacy_fields={'action','answer','tools','metrics','required','stations','dates','date_scope','question'}
    if set(value) - legacy_fields:
        raise ValueError('unknown plan fields')
    required=value.get('required')
    if require_outcomes and required is None:
        raise ValueError('missing required outcomes')
    if required is not None and (not isinstance(required,list) or len(required)>11
                                or any(not isinstance(x,str) for x in required)):
        raise ValueError('invalid required outcomes')
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
    if required is not None:
        if action != 'compute' and required:
            raise ValueError('discussion has required execution outcomes')
        if set(required)-set(value['tools'])-set(value['metrics']):
            raise ValueError('required outcome was not selected')
        required_metrics=[x for x in required if x in METRICS]
        operators=['calculate_eej_index' if x=='eej_index' else 'apply_physics_formula'
                   for x in required_metrics]
        result['required_operations']=list(dict.fromkeys([x for x in required if x in CAPABILITIES]+operators))
        result['required_metrics']=list(dict.fromkeys(required_metrics))
        if action=='compute' and not result['required_operations']:
            raise ValueError('computation has no required outcome')
    return result



def decode_arguments(value, require_outcomes=False):
    """Normalize a single typed operation list without guessing identifier aliases.

    Legacy replies are readable only through the explicit compatibility path;
    production structured planning must use the current work schema.
    """
    if not isinstance(value,dict):raise ValueError('plan must be an object')
    if 'work' not in value:
        if require_outcomes:raise ValueError('missing work selection')
        return _decode_legacy_arguments(value)
    if set(value)-set(ARGUMENT_SCHEMA['properties']):raise ValueError('unknown plan fields')
    action,answer=value.get('action'),value.get('answer')
    if action not in ARGUMENT_SCHEMA['properties']['action']['enum'] or answer not in ARGUMENT_SCHEMA['properties']['answer']['enum']:
        raise ValueError('invalid action/answer')
    for key in ('stations','dates'):
        items=value.get(key)
        if not isinstance(items,list) or len(items)>8 or any(not isinstance(x,str) for x in items):
            raise ValueError('invalid '+key)
    work=value['work']
    if not isinstance(work,list) or len(work)>8:raise ValueError('invalid work selection')
    roles={}
    for item in work:
        if (not isinstance(item,dict) or set(item)!={'id','required'}
                or type(item['required']) is not bool or not isinstance(item['id'],str)):
            raise ValueError('invalid work item')
        name=item['id']
        if name not in CAPABILITIES and name not in METRICS:raise ValueError('unregistered capability')
        roles[name]=roles.get(name,False) or item['required']
    mode=('execute' if answer=='facts' else 'analyze') if action=='compute' else ('clarify' if action=='clarify' else ('plan' if answer=='plan' else 'explain'))
    result={k:value[k] for k in ('stations','dates','date_scope') if k in value}
    result.update(mode=mode,question=value.get('question',''))
    selected=roles if action=='compute' else {}
    metrics=[name for name in selected if name in METRICS]
    required_metrics=[name for name in metrics if selected[name]]
    operator=lambda metric: 'calculate_eej_index' if metric=='eej_index' else 'apply_physics_formula'
    tools=[name for name in selected if name in CAPABILITIES]+[operator(metric) for metric in metrics]
    required=[name for name in selected if name in CAPABILITIES and selected[name]]+[operator(metric) for metric in required_metrics]
    result.update(tools=list(dict.fromkeys(tools)),metrics=metrics,
                  required_operations=list(dict.fromkeys(required)),required_metrics=required_metrics)
    if action=='compute' and not result['required_operations']:
        raise ValueError('computation has no required outcome')
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
    reply = None

    def failure(stage, exc):
        result={'status': 'error' if structured else 'fallback',
                'reason': 'invalid_plan:' + type(exc).__name__,
                'stage': stage, 'detail': str(exc)[:240]}
        if structured and isinstance(reply,dict):
            raw=str(reply.get('content') or '')[:2400]
            secret=os.environ.get('SUPERMAG_LOGON','')
            result['raw_response']=raw.replace(secret,'[redacted]') if secret else raw
            result['finish_reason']=reply.get('finish_reason')
        return result

    if len(current) > 4000 or len(SYSTEM) + len(current) + len(facts) > MAX_INPUT_CHARS:
        return {'status': 'error' if structured else 'fallback', 'reason': 'planner_input_budget'}
    known=station_aliases or {}
    codes=mentioned_stations(current,known) | {x.lower() for x in re.findall(r'\b[A-Z]{3}\b',current)
          if x not in NON_STATION_CODES}
    labels={code.upper():known.get(code.upper(),known.get(code)) for code in sorted(codes)}
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
            value = decode_arguments(value, require_outcomes=True)
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
    if plan['mode']=='analyze' and set(names)&{'fetch_observatory_data','fetch_solar_wind',
                                               'fetch_geomagnetic_indices','fetch_supermag_data'}:
        names.append('get_statistics')
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
