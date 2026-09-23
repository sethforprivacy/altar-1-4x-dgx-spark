#!/usr/bin/env python3
"""Check model API behavior and tool calling on a running deployment."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True, help='API origin, without /v1')
    parser.add_argument('--key-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='altar-1')
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new receipt path")
    key = args.key_file.read_text().splitlines()[0]
    results = []
    def request(path, body=None, auth=key):
        headers = {'Content-Type': 'application/json'}
        if auth is not None:
            headers['Authorization'] = 'Bearer ' + auth
        req = urllib.request.Request(args.url.rstrip('/') + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers=headers)
        return urllib.request.urlopen(req, timeout=900)
    def chat(messages, **kwargs):
        body = {'model': args.model, 'messages': messages, 'temperature': 0,
                'max_tokens': 2048, 'reasoning_effort': 'max', **kwargs}
        with request('/v1/chat/completions', body) as response:
            return json.load(response)
    def gate(name, fn):
        started = time.monotonic()
        try:
            detail = fn()
            result = {'name': name, 'pass': True, 'detail': detail}
        except Exception as error:
            result = {'name': name, 'pass': False,
                      'error': f'{type(error).__name__}: {error}'}
        result['duration_s'] = round(time.monotonic() - started, 3)
        results.append(result)
        print(json.dumps(result), flush=True)
    def catalog():
        with request('/v1/models') as response:
            models = json.load(response)['data']
        assert [x['id'] for x in models] == [args.model], models
        return models
    gate('catalog', catalog)
    def unauthorized(auth):
        try:
            request('/v1/chat/completions', {'model':args.model,
                'messages':[{'role':'user','content':'hello'}]}, auth)
        except urllib.error.HTTPError as error:
            assert error.code == 401, error.code
            return {'status': error.code}
        raise AssertionError('unauthenticated generation accepted')
    gate('missing-key', lambda: unauthorized(None))
    gate('invalid-key', lambda: unauthorized('invalid-qualification-key'))
    def arithmetic(effort):
        result = chat([{'role':'user','content':'Compute 19 + 23. Reply with only the integer.'}],
                      reasoning_effort=effort)
        choice = result['choices'][0]
        assert choice['finish_reason'] == 'stop', choice
        assert choice['message']['content'].strip() == '42', choice
        assert result['usage']['completion_tokens'] > 0, result
        if effort in ('high', 'max'):
            assert choice['message'].get('reasoning_content') or choice['message'].get('reasoning'), choice
        return {'usage':result['usage'], 'reasoning_present':bool(
            choice['message'].get('reasoning_content') or choice['message'].get('reasoning'))}
    for effort in ('low','high','max'):
        gate('arithmetic-' + effort, lambda e=effort: arithmetic(e))
    def structured():
        result = chat([{'role':'user','content':'Return a JSON object with exactly one key, value, whose value is the integer 42.'}],
                      response_format={'type':'json_object'})
        assert json.loads(result['choices'][0]['message']['content']) == {'value':42}, result
        assert result['choices'][0]['finish_reason'] == 'stop', result
        return result['usage']
    gate('json-object', structured)
    def tool_turn():
        tools=[{'type':'function','function':{'name':'ticket_status',
            'description':'Look up a sample ticket status.',
            'parameters':{'type':'object','properties':{'ticket':{'type':'string'}},
                          'required':['ticket'],'additionalProperties':False}}}]
        messages=[{'role':'user','content':'Use ticket_status to look up OPS-314, then tell me its status. Do not guess.'}]
        first=chat(messages,tools=tools,tool_choice='auto')
        assistant=first['choices'][0]['message']
        calls=assistant.get('tool_calls',[])
        assert len(calls)==1, assistant
        assert first['choices'][0]['finish_reason']=='tool_calls', first
        call=calls[0]
        assert call['function']['name']=='ticket_status',call
        assert json.loads(call['function']['arguments'])=={'ticket':'OPS-314'},call
        messages.extend([assistant,{'role':'tool','tool_call_id':call['id'],
                                    'content':'{"status":"resolved","receipt":"LANTERN-7492"}'}])
        final=chat(messages,tools=tools,tool_choice='auto')
        text=final['choices'][0]['message'].get('content','').lower()
        assert 'resolved' in text, final
        assert final['choices'][0]['finish_reason']=='stop',final
        return {'tool':'ticket_status','continued':True}
    gate('tool-roundtrip', tool_turn)
    def reordered_tools():
        tools=[{'type':'function','function':{'name':'ticket_status',
            'description':'Return status for exactly one ticket. Call separately for every ticket.',
            'parameters':{'type':'object','properties':{'ticket':{'type':'string'}},
                          'required':['ticket'],'additionalProperties':False}}}]
        messages=[{'role':'user','content':'Look up tickets OPS-314 and OPS-271 using two ticket_status calls in the same turn. Then reply with only a JSON object mapping each ticket ID to its returned status.'}]
        first=chat(messages,tools=tools,tool_choice='auto',parallel_tool_calls=True)
        assistant=first['choices'][0]['message']; calls=assistant.get('tool_calls',[])
        assert len(calls)==2 and len({c['id'] for c in calls})==2, assistant
        tickets={json.loads(c['function']['arguments'])['ticket'] for c in calls}
        assert tickets=={'OPS-314','OPS-271'},calls
        statuses={'OPS-314':'resolved','OPS-271':'pending'}
        messages.append(assistant)
        for call in reversed(calls):
            assert call['function']['name']=='ticket_status',call
            ticket=json.loads(call['function']['arguments'])['ticket']
            messages.append({'role':'tool','tool_call_id':call['id'],
                             'content':json.dumps({'ticket':ticket,'status':statuses[ticket]})})
        final=chat(messages,tools=tools,tool_choice='auto',response_format={'type':'json_object'})
        content=final['choices'][0]['message'].get('content','')
        assert json.loads(content)==statuses,final
        assert final['choices'][0]['finish_reason']=='stop',final
        return {'parallel_calls':2,'reversed_results_accepted':True}
    gate('parallel-reordered-tools',reordered_tools)
    def stream():
        body={'model':args.model,'messages':[{'role':'user','content':'Reply with only STREAM_OK.'}],
              'temperature':0,'max_tokens':2048,'stream':True,
              'stream_options':{'include_usage':True},'reasoning_effort':'low'}
        text=''; usage=None; done=False; finish=None
        with request('/v1/chat/completions',body) as response:
            for line in response:
                line=line.decode().strip()
                if not line.startswith('data: '):continue
                data=line[6:]
                if data=='[DONE]':done=True;break
                data=json.loads(data)
                if data.get('usage'):usage=data['usage']
                for choice in data.get('choices',[]):
                    text+=choice.get('delta',{}).get('content') or ''
                    finish=choice.get('finish_reason') or finish
        assert done and finish=='stop' and 'STREAM_OK' in text and usage, (done,finish,text,usage)
        return {'usage':usage,'done':done}
    gate('streaming-usage',stream)
    def concurrency_gate():
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            return list(pool.map(arithmetic,['low']*4))
    gate('four-concurrent', concurrency_gate)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        json.dump({'model':args.model,'checks':results,
                   'passed':all(x['pass'] for x in results)},stream,indent=2)
    raise SystemExit(0 if all(x['pass'] for x in results) else 1)

if __name__=='__main__':
    main()
