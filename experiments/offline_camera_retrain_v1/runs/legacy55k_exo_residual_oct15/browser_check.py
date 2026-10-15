"""Exercise every replay case/frame boundary and actual play/pause in Chromium."""
import base64
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request
from websockets.sync.client import connect

ROOT=Path(__file__).resolve().parent/'reference_anchor'


def main():
    env=dict(os.environ,LD_LIBRARY_PATH='/tmp/scenemi_browser_deps/root/usr/lib/x86_64-linux-gnu',
        FONTCONFIG_FILE='/tmp/scenemi_browser_deps/fonts.conf')
    args=['/home/wenxin/.cache/ms-playwright/chromium-1223/chrome-linux64/chrome',
        '--headless','--no-sandbox','--disable-dev-shm-usage','--disable-background-networking',
        '--remote-debugging-port=9224','--remote-allow-origins=*','--window-size=1500,1000',
        '--user-data-dir=/tmp/scenemi_legacy_exo_cdp','about:blank']
    with (ROOT/'browser.log').open('w') as stream:browser=subprocess.Popen(args,env=env,stdout=stream,stderr=subprocess.STDOUT)
    socket=None
    try:
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for _ in range(100):
            try:
                pages=json.load(opener.open('http://127.0.0.1:9224/json',timeout=1))
                if pages:break
            except OSError:time.sleep(.1)
        socket=connect(pages[0]['webSocketDebuggerUrl'],max_size=30*1024*1024,proxy=None)
        serial=0;errors=[]
        def rpc(method,params=None):
            nonlocal serial
            serial+=1;socket.send(json.dumps(dict(id=serial,method=method,params=params or {})))
            while True:
                result=json.loads(socket.recv(timeout=30))
                if result.get('method')=='Runtime.exceptionThrown':errors.append(result['params'])
                if result.get('id')==serial:
                    if 'error' in result:raise RuntimeError(result)
                    return result.get('result',{})
        def evaluate(code):
            result=rpc('Runtime.evaluate',dict(expression=code,returnByValue=True))
            if 'exceptionDetails' in result:raise RuntimeError(result)
            return result['result'].get('value')
        rpc('Runtime.enable');rpc('Page.enable');rpc('Page.navigate',dict(url='http://127.0.0.1:8790/PROTECTED_REPLAY.html'))
        for _ in range(200):
            count=evaluate("document.getElementById('case')?.options.length")
            if count:break
            time.sleep(.1)
        assert count==32,count
        checked=[]
        for case in range(count):
            for frame in (0,48,95):
                value=evaluate(f"$('case').value={case};$('frame').value={frame};draw();$('time').textContent")
                assert 'NaN' not in value;checked.append([case,frame])
        evaluate("$('case').value=5;$('frame').value=20;draw();$('play').click()")
        time.sleep(.5);advanced=evaluate("+$('frame').value");assert advanced!=20
        evaluate("$('play').click()");time.sleep(.1);paused=evaluate("+$('frame').value")
        time.sleep(.3);assert evaluate("+$('frame').value")==paused
        assert evaluate("document.querySelectorAll('canvas').length")==3
        evaluate("$('frame').value=48;draw()")
        (ROOT/'REPLAY_BROWSER.png').write_bytes(base64.b64decode(rpc('Page.captureScreenshot',dict(format='png'))['data']))
        assert not errors,errors
        (ROOT/'BROWSER_CHECK.json').write_text(json.dumps(dict(status='passed',cases=count,case_frame_checks=checked,
            playback_advanced_to=advanced,pause_stable=True,canvas_count=3,runtime_exceptions=errors),indent=2)+'\n')
        print('BROWSER CHECK PASSED')
    finally:
        if socket is not None:socket.close()
        browser.terminate()
        try:browser.wait(timeout=5)
        except subprocess.TimeoutExpired:browser.kill()


if __name__=='__main__':main()
