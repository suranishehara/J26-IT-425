"""Check actual person-analysis HTTP inference on a synthetic test person."""
import base64
import json
from pathlib import Path
import sys
import threading
from urllib.request import Request,urlopen

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from app import Handler,ThreadingHTTPServer

if __name__=='__main__':
    source=ROOT.parent/'component-1-virtual-tryon/data/person_references/female/person-female-01.png'
    if len(sys.argv)>1:
        source=Path(sys.argv[1])
    payload={'image':'data:image/png;base64,'+base64.b64encode(source.read_bytes()).decode(),'consent':True}
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    try:
        request=Request('http://127.0.0.1:'+str(server.server_port)+'/api/analyse-person',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        with urlopen(request,timeout=300) as response:
            result=json.load(response)
        assert result['status']=='analysed',result
        assert result['photo_persisted'] is False
        assert result['overlay'].startswith('data:image/png;base64,')
        assert 'body' in result and 'skin' in result
        print(json.dumps({key:value for key,value in result.items() if key!='overlay'},indent=2))
        print('Real person-analysis HTTP smoke test passed. No accuracy or fairness claim.')
    finally:
        server.shutdown();server.server_close()
