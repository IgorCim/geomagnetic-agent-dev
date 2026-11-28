"""Bounded, data-only research evidence and capability limitations.

This does not choose the user's task, execute science or certify an answer.
Topic matching retrieves definitions; it never selects or forbids tools.
No model prose, credentials, remote instructions or arbitrary frame values enter
session evidence. Records come from successful, registered handler results.
"""
from __future__ import annotations
import json
import math
import os
import re
from numbers import Real

MAX_RECORDS=32
MAX_CONTEXT_CHARS=2400
PARAMETERS=('metric','component','components','mode','night_hours','baseline_value','sigma_threshold',
            'station_code','equatorial_station','reference_station','stations','dates',
            'timestamp','start_date','end_date','time_system','window_seconds')
VALUES=('metric','component','computed','definitions','baseline','mode','window','night_hours','n_samples',
        'n_finite','unit','delta','max_delta','mean_delta','peak_rate','n_flagged','sigma_threshold',
        'baseline_value','result','limitations')


def clean(value, depth=0):
    """Allow only bounded JSON data, retaining units and source identity."""
    if depth>4:return None
    if value is None or isinstance(value,bool):return value
    if isinstance(value,Real):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value,str):return value[:240]
    if isinstance(value,(list,tuple)):return [clean(x,depth+1) for x in value[:12]]
    if isinstance(value,dict):return {str(k)[:60]:clean(v,depth+1) for k,v in list(value.items())[:12]}
    return None


def record(entry):
    if not entry.get('ok'):return None
    tool=entry.get('tool')
    from request_planner import CAPABILITIES
    if tool not in CAPABILITIES:return None
    data={k:clean(entry[k]) for k in ('calculation','statistics','local_time','coordinate_result') if entry.get(k)}
    if 'calculation' in data:
        data['calculation']={k:v for k,v in data['calculation'].items() if k in VALUES}
    inputs=entry.get('input_provenance') or [
        {'handle':h} for h in entry.get('resolved_handles',[])]
    if not inputs and entry.get('frame_handle'):inputs=[{'handle':entry['frame_handle']}]
    if not data and not inputs:return None
    item={'tool':tool,'inputs':clean(inputs)}
    args=entry.get('arguments') or {}
    params={k:clean(args[k]) for k in PARAMETERS if k in args}
    if params:item['parameters']=params
    item.update(data)
    if entry.get('frame_handle'):item['output']=str(entry['frame_handle'])[:120]
    return item


def merge(previous, log):
    result=list(previous)
    for entry in log:
        item=record(entry)
        if item and item not in result:result.append(item)
    return result[-MAX_RECORDS:]


def reference_notes(query, intent):
    """General definitions, explicitly separate from measured results.

    Kp: https://kp.gfz.de/en/about-kp
    Other numerical meanings match the existing handler/physics definitions.
    """
    q=query.casefold()
    tools=set((intent or {}).get('tools',[]))
    notes=['Separate recorded facts, interpretation and untested hypotheses. Do not turn an unperformed test into a result.']
    if 'detect_anomalies' in tools or re.search(r'сигм|выброс|sigma|outlier',q):
        notes.append('A sigma outlier test finds unusual samples; it does not establish storm absence/presence.')
    if 'fetch_geomagnetic_indices' in tools or re.search(r'\bkp\b|геомагнитн\w*\s+индекс',q):
        notes.append('Kp is a planetary geomagnetic activity index; K is a local station index.')
    if tools&{'calculate_baseline','calculate_derived_math'} or re.search(r'опорн|baseline|ноч|utc',q):
        notes.append('Changing constant H0 shifts H-H0; max-min is unchanged. mode=night only labels a UTC window, not verified local or quiet night. A full-day mean can include disturbance; neither reference is automatically quiet.')
    if re.search(r'gic|ампер|ток|dh.?dt|скорост[ьи].*измен',q):
        notes.append('dH/dt has nT/min units; currents in amperes need an additional geoelectric/conductivity/network model.')
    if re.search(r'гипотез|коррел|причин|связ|однознач|уверен|формулир|hypothes|caus|correl',q):
        notes.append('Association or visual coincidence is not unique causation. Claim a computed correlation only with a recorded statistical result.')
    if len((intent or {}).get('stations',[]))>1 or re.search(r'абсолютн|средн\w*\s+h\b',q):
        notes.append('H is horizontal magnitude hypot(X,Y); Z is vertical. Absolute H includes the background field; compare disturbances with justified references and coverage.')
    return notes[:5]


def render(records, query='', intent=None, labels=None, max_chars=MAX_CONTEXT_CHARS, with_notes=True):
    notes=reference_notes(query,intent) if with_notes else []
    selected=list(records[-8:])
    omitted=max(0,len(records)-len(selected))
    # Unknown names stay unknown. A three-letter code is not a place-name guess.
    payload={'source':'completed tool results; output slots may be reused; definitions are not observations',
             'station_labels':labels or {},'records':selected,'reference_notes':notes,'omitted_records':omitted}
    while True:
        text=json.dumps(payload,ensure_ascii=False,separators=(',',':'))
        secret=os.environ.get('SUPERMAG_LOGON','')
        if secret:text=text.replace(secret,'[redacted]')
        if len(text)<=max_chars:return text
        if payload['records']:
            payload['records']=payload['records'][1:];payload['omitted_records']+=1
        elif payload['reference_notes']:
            payload['reference_notes']=payload['reference_notes'][:-1]
        elif payload['station_labels']:
            payload['station_labels']={}
        else:
            return '{"source":"evidence_context_budget","records":[],"omitted_records":'+str(len(records))+'}'
