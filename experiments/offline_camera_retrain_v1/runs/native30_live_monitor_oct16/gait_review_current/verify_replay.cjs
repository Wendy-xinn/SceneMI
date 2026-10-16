// Exercise the actual generated HTML script against a minimal canvas/DOM host.
const fs = require('fs'), vm = require('vm'), path = require('path');
const html = fs.readFileSync(path.join(__dirname,'REPLAY.html'),'utf8');
const script = html.slice(html.lastIndexOf('<script>')+8,html.lastIndexOf('</script>'));
let drawingCalls=0, nextFrame;
const ctx = new Proxy({}, {get(target,key){if(key in target)return target[key];return (...args)=>{drawingCalls++;for(const value of args)if(typeof value==='number'&&!Number.isFinite(value))throw Error(`Non-finite canvas argument in ${String(key)}`)}}});
const elements = new Map();
function element(id){if(!elements.has(id)){elements.set(id,{id,value:id==='regime'?'clean':id==='speed'?'1':'0',checked:id==='trail',width:id==='charts'?1400:700,height:id==='charts'?400:520,appendChild(){},getContext(){return ctx},setPointerCapture(){}})}return elements.get(id)}
const views=['oblique','top','front','side'].map(v=>({dataset:{view:v}}));
const host={document:{getElementById:element,createElement:()=>({}),querySelectorAll:()=>views},requestAnimationFrame:cb=>{nextFrame=cb},console};
vm.createContext(host);vm.runInContext(script,host,{timeout:30000});
if(drawingCalls<1000)throw Error('Initial replay did not render real trajectories');
element('slider').value='96';element('slider').oninput();
if(!element('time').textContent.includes('帧 96'))throw Error('Frame drag did not change render');
element('play').onclick();nextFrame(100);nextFrame(200);
if(+element('slider').value<=96)throw Error('Play did not advance frame');
element('play').onclick();
for(let c=0;c<6;c++){element('case').value=String(c);element('case').onchange();for(const r of ['clean','noisy']){element('regime').value=r;element('regime').onchange();element('slider').value='191';element('slider').oninput();for(const view of views)view.onclick()}}
element('relative').checked=true;element('relative').onchange();element('relative').checked=false;element('relative').onchange();
console.log(JSON.stringify({status:'passed',actual_generated_script:true,step:10000,cases:6,regimes:2,canvas_arguments_all_finite:true,play_advances_frame:true,drag_frame_96:true,last_frame_191:true,views:4,absolute_and_relative:true,drawingCalls},null,2));
