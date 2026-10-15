"""Exercise local replay controls and capture both viewers in Chromium."""
import base64
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request
from websockets.sync.client import connect

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'visual_review_step8000'


def main():
    env = dict(os.environ, LD_LIBRARY_PATH='/tmp/scenemi_browser_deps/root/usr/lib/x86_64-linux-gnu',
        FONTCONFIG_FILE='/tmp/scenemi_browser_deps/fonts.conf')
    args = ['/home/wenxin/.cache/ms-playwright/chromium-1223/chrome-linux64/chrome',
        '--headless', '--no-sandbox', '--disable-dev-shm-usage', '--disable-background-networking',
        '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
        '--remote-debugging-port=9223', '--remote-allow-origins=*', '--window-size=1500,1100',
        '--user-data-dir=/tmp/scenemi_visual_cdp_profile', 'about:blank']
    with Path('/tmp/scenemi_visual_cdp.log').open('w') as log:
        browser = subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT)
    socket = None
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        pages = []
        last_error = None
        for _ in range(100):
            try:
                pages = json.load(opener.open('http://127.0.0.1:9223/json', timeout=1))
                if pages:break
            except OSError as exc:last_error = repr(exc)
            time.sleep(.1)
        if not pages:raise RuntimeError(f'Chromium debugging endpoint unavailable: {last_error}')
        socket = connect(pages[0]['webSocketDebuggerUrl'], max_size=30*1024*1024, proxy=None)
        serial = 0
        exceptions = []
        def rpc(method, params=None):
            nonlocal serial
            serial += 1
            ident = serial
            socket.send(json.dumps(dict(id=ident, method=method, params=params or {})))
            while True:
                msg = json.loads(socket.recv(timeout=30))
                if msg.get('method') == 'Runtime.exceptionThrown':exceptions.append(msg['params'])
                if msg.get('id') == ident:
                    if 'error' in msg:raise RuntimeError(msg['error'])
                    return msg.get('result', {})
        def evaluate(js):
            value = rpc('Runtime.evaluate', dict(expression=js, returnByValue=True))
            if 'exceptionDetails' in value:raise RuntimeError(value)
            return value['result'].get('value')
        rpc('Runtime.enable');rpc('Page.enable')
        rpc('Page.navigate', dict(url='http://127.0.0.1:8789/REPLAY.html'))
        for _ in range(100):
            if evaluate("document.querySelector('#case')?.options.length") == 6:break
            time.sleep(.1)
        checks = []
        for case in range(6):
            for noise in ('clean','noisy'):
                text = evaluate(f"$('case').value={case};$('noise').value='{noise}';$('frame').value=191;draw();$('info').textContent")
                assert 'NaN' not in text and 'MPJPE' in text
                checks.append(dict(case=case, regime=noise, frame=191, rendered=True))
        evaluate("$('case').value=2;$('noise').value='clean';$('frame').value=96;draw();$('play').click()")
        time.sleep(.5)
        frame = int(evaluate("+$('frame').value"))
        assert frame != 96
        evaluate("$('play').click();$('frame').value=96;draw()")
        png = rpc('Page.captureScreenshot', dict(format='png'))['data']
        (OUT / 'REPLAY_BROWSER.png').write_bytes(base64.b64decode(png))
        rpc('Page.navigate', dict(url='http://127.0.0.1:8788/'))
        for _ in range(200):
            if evaluate("/第\\d+帧/.test(document.body?.innerText)"):
                break
            time.sleep(.1)
        assert evaluate("document.querySelectorAll('canvas').length") >= 1
        assert evaluate("/第\\d+帧/.test(document.body.innerText)")
        controls = evaluate("Array.from(document.querySelectorAll('input[type=checkbox]')).map(x=>({id:x.id,checked:x.checked,html:x.parentElement.parentElement.outerHTML}))")
        (OUT / 'browser_native_controls.json').write_text(json.dumps(controls, ensure_ascii=False, indent=2))
        toggle_play = "Array.from(document.querySelectorAll('input[type=checkbox]')).find(x=>/^[0-9a-f]{8}-/.test(x.id)).click()"
        if evaluate("Array.from(document.querySelectorAll('input[type=checkbox]')).find(x=>/^[0-9a-f]{8}-/.test(x.id)).checked"):
            evaluate(toggle_play)
            time.sleep(.3)
        initial_frame = evaluate("document.body.innerText.match(/第(\\d+)帧/)[1]")
        evaluate(toggle_play)
        time.sleep(2.)
        playing_frame = evaluate("document.body.innerText.match(/第(\\d+)帧/)[1]")
        assert playing_frame != initial_frame
        evaluate(toggle_play)
        time.sleep(.3)
        paused = evaluate("document.body.innerText.match(/第(\\d+)帧/)[1]")
        time.sleep(.4)
        assert evaluate("document.body.innerText.match(/第(\\d+)帧/)[1]") == paused
        png = rpc('Page.captureScreenshot', dict(format='png'))['data']
        (OUT / 'NATIVE_VIEWER_BROWSER.png').write_bytes(base64.b64decode(png))
        assert not exceptions, exceptions
        report = dict(status='passed', portable_replay_cases=checks, portable_playback_advanced_to=frame,
            native_websocket_connected=True, native_mesh_canvas_present=True, native_playback_advanced=True, native_pause_stable=True,
            browser_runtime_exceptions=exceptions, sources='Actual archived step8000 VAL, not new model inference.')
        (OUT / 'BROWSER_CHECK.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        print(json.dumps(report), flush=True)
    finally:
        if socket is not None:socket.close()
        browser.terminate()
        try:browser.wait(timeout=5)
        except subprocess.TimeoutExpired:browser.kill()


if __name__ == '__main__':main()
