"""Check review_answer with the native llama.cpp parser, without model weights.

Each native call runs in a child process so a rejected legacy grammar cannot
crash the caller. This verifies grammar initialization, not Qwen answers.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def native_worker():
    try:
        import llama_cpp
        from llama_cpp import LlamaGrammar
        from llama_cpp import llama_cpp as lc
    except (ImportError, OSError, RuntimeError) as exc:
        print(json.dumps({'accepted':False,'runtime_unavailable':True,'error':type(exc).__name__+': '+str(exc)}),flush=True)
        return 3
    info={'llama_cpp_python':llama_cpp.__version__,'platform':platform.platform(),
          'scope':'Native grammar initialization and free only; no model loading or sampling.'}
    sampler=None
    try:
        value=json.load(sys.stdin)
        grammar=LlamaGrammar.from_json_schema(json.dumps(value),verbose=False)
        # These JSON grammars use character rules, not vocabulary-token rules.
        # The vocabulary is needed for sampling; this check never samples.
        sampler=lc.llama_sampler_init_grammar(None,grammar._grammar.encode('utf-8'),b'root')
        print(json.dumps({**info,'accepted':bool(sampler),'grammar_bytes':len(grammar._grammar.encode('utf-8'))}),flush=True)
        return 0 if sampler else 2
    except Exception as exc:
        print(json.dumps({**info,'accepted':False,'error':type(exc).__name__+': '+str(exc)}),flush=True)
        return 2
    finally:
        if sampler:lc.llama_sampler_free(sampler)


def probe(value, timeout=20):
    try:
        result=subprocess.run([sys.executable,'-B','-X','utf8',str(Path(__file__).resolve()),'--native-worker'],
                              input=json.dumps(value),text=True,encoding='utf-8',capture_output=True,timeout=timeout)
    except subprocess.TimeoutExpired:
        return {'accepted':False,'error':'native_parser_timeout','timeout_seconds':timeout}
    try:details=json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError,IndexError):details={'accepted':False,'error':'native_worker_did_not_return_json'}
    return {**details,'returncode':result.returncode,'stderr':result.stderr}


def run(output_dir=None, include_legacy=False):
    sys.path.insert(0,str(ROOT))
    import request_planner as rp
    import response_review as rr
    historical=json.loads((ROOT/'tests/fixtures/review_schema_3168e77.json').read_text(encoding='utf-8'))
    old=probe(historical['schema']) if include_legacy else None
    optional=[[],['calculate_baseline'],list(rp.CAPABILITIES)[:8]]
    current=[]
    for names in optional:
        result=probe(rr.schema(names))
        current.append({'optional_count':len(names),**result})
        print('NATIVE review_answer: '+('PASS' if result.get('accepted') and result.get('returncode')==0 else 'FAIL')+'; optional='+str(len(names)),flush=True)
    passed=all(r.get('accepted') and r.get('returncode')==0 for r in current)
    report={'status':'PASS' if passed else 'FAIL',
            'scope':'Native grammar parsing without weights, GPU, generation or scientific acceptance.',
            'historical_schema_revision':historical['source_revision'],
            'historical_schema':old,'current_schemas':current,
            'legacy_rejected_on_this_build':None if old is None else not old.get('accepted') and not old.get('runtime_unavailable')}
    if output_dir is None:
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        output_dir=ROOT/'verification'/('native_grammar_'+stamp)
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Evidence: '+str(output/'report.json'),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--include-legacy',action='store_true',help='Also reproduce the old grammar failure in an isolated child')
    args=parser.parse_args()
    if args.native_worker:return native_worker()
    return 0 if run(args.output_dir,include_legacy=args.include_legacy)['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
