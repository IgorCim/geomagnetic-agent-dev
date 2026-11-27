"""Real validation and scientific handlers; inputs are explicit synthetic fixtures."""
import math
import pandas as pd
import pytest
import agent_core as ac
import project_store as ps
import request_planner as rp
from chat_session import Request

@pytest.mark.parametrize('days', ['10 или 12', '10 либо 12', '10 и 12', '10/12', '10–12', '10 по 12'])
def test_written_shared_month_dates_are_preserved(days):
    assert rp.explicit_dates(f'Сравни {days} сентября 2024 года.')=={'2024-09-10','2024-09-12'}

@pytest.mark.parametrize('component,other_handle', [('H',False),('X',False),('H',True)])
def test_multiple_baselines_remain_bound_to_their_table_component_and_request(tmp_path,component,other_handle):
    store=ac.FrameStore(workspace=ps.Workspace(root=tmp_path/'projects'))
    h='raw:irt:2024-09-12'
    store.put(h,pd.DataFrame({'timestamp':pd.date_range('2024-09-12',periods=8,freq='h',tz='UTC'),
                            'X':[3.,0.,3.,0.,10.,10.,10.,10.], 'Y':[0.,4.,0.,4.,0.,0.,0.,0.], 'Z':[0.]*8}))
    request=Request('TEST FIXTURE: multiple reference levels',calculation_tools=('calculate_baseline',))
    values=[]
    for mode in ['full','night']:
        args={'df_descriptor':h,'component':'H','mode':mode,'night_hours':[0,4]}
        payload,_=ac.HANDLERS['calculate_baseline'](args,store)
        assert not ac.is_error(payload)
        request.observe_result('calculate_baseline',args,payload)
        values.append(payload['baseline'])
    assert values==[6.75,3.5]
    for value in values:
        args={'df_descriptor':h,'component':'H','metric':'anomaly','baseline_value':value}
        assert request.check_call('calculate_derived_math',args,store) is None
        assert Request('new request',calculation_tools=('calculate_baseline',)).check_call('calculate_derived_math',args,store)
        args['baseline_value']=value+.1
        assert request.check_call('calculate_derived_math',args,store)
    args={'df_descriptor':'raw:bou:2024-09-12' if other_handle else h,'component':component,'metric':'anomaly','baseline_value':values[0]}
    if component!='H' or other_handle:
        assert request.check_call('calculate_derived_math',args,store)
