// Real frontend stream consumer; fabricated transport events, not LLM evaluation.
const fs=require('node:fs'), vm=require('node:vm'), assert=require('node:assert/strict');
class Node {
  constructor(tag='div'){this.tagName=tag;this.children=[];this.value='';this.textContent='';this.classList={add(){},remove(){}};}
  append(...items){this.children.push(...items);}
  replaceChildren(...items){this.children=items;}
  querySelector(selector){if(selector.startsWith('option'))return option;return null;}
  setAttribute(name,value){this[name]=value;}
  get options(){return this.children;}
}
const map=new Map(),option=new Node('option');
const frames=[], observers=[];
function get(id){if(!map.has(id))map.set(id,new Node());return map.get(id);}
get('generation').value='baseline';
get('retrieval').value='bm25';
let release;
const gate=new Promise(resolve=>release=resolve);
const meta={session_id:'test',state:{occasion:'temple'},understanding:{intent:'knowledge_question',entities:[]},evidence:[],knowledge_sha256:'test',context_used:false,methods:{retrieval:'bm25',generation:'ollama_grounded_llm'}};
const encoder=new TextEncoder();
const events=[{type:'start',data:meta},{type:'delta',text:'Hello — '},{type:'delta',text:'world'}, {type:'done',data:{...meta,revision:1,answer:'Hello — world',model:'fixture:tiny',citation_check:{unknown_ids:[],missing_citation:false}}}];
const first=encoder.encode(events.slice(0,2).map(event=>JSON.stringify(event)).join('\n')+'\n');
const utf8Split=first.indexOf(0xe2)+1; // Split inside the em dash, not just ASCII JSON.
const rest=encoder.encode(events.slice(2).map(event=>JSON.stringify(event)).join('\n')+'\n');
const context=vm.createContext({document:{getElementById:get,createElement:tag=>new Node(tag),createTextNode:text=>({textContent:text})},TextDecoder,Uint8Array,AbortController,
  requestAnimationFrame:callback=>{frames.push(callback);return frames.length;},
  ResizeObserver:class{constructor(callback){this.callback=callback;this.targets=[];observers.push(this);}observe(node){this.targets.push(node);}},
  fetch:async(url,options)=>{
    if(url==='/api/chats')return {ok:true,json:async()=>({enabled:false,chats:[],deleted:[]})};
    if(url==='/api/session')return {ok:true,json:async()=>({session_id:'test',revision:0})};
    if(url==='/api/models')return {ok:true,json:async()=>({available:true,models:[{name:'fixture:tiny'}]})};
    if(url==='/api/retrieval')return {ok:true,json:async()=>({available:true,model:{name:'all-minilm:22m'}})};
    const request=JSON.parse(options.body);assert.equal(request.generation,'ollama');assert.equal(request.retrieval,'bm25');
    let index=0;
    return {ok:true,body:{getReader:()=>({read:async()=>{index++;if(index===1)return {value:first.slice(0,utf8Split),done:false};if(index===2)return {value:first.slice(utf8Split),done:false};if(index===3){await gate;return {value:rest,done:false};}return {done:true};},releaseLock(){}})}};
  }});
vm.runInContext(fs.readFileSync(require('node:path').join(__dirname,'../frontend/app.js'),'utf8'),context);
async function run(){
  await new Promise(resolve=>setImmediate(resolve));
  function paint(){const pending=frames.splice(0);for(const callback of pending)callback();}
  get('messages').scrollHeight=900;paint();
  assert.equal(get('messages').scrollTop,900,'pin after the initial layout');
  get('messages').scrollHeight=1100;observers[0].callback();paint();
  assert.equal(get('messages').scrollTop,1100,'composer/transcript resize should re-pin the latest message');
  assert.deepEqual(observers[0].targets,[get('messages'),get('chat-form'),get('status')]);
  get('message').value='Tell me about temple clothes';get('component-json').value='';
  const pending=get('chat-form').onsubmit({preventDefault(){}});
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(get('messages').children.at(-2).children.at(-1).textContent,'Tell me about temple clothes','user should appear before completion');
  assert.equal(get('messages').children.at(-1).children.at(-1).textContent,'Hello — ','partial response should already be rendered');
  assert.equal(vm.runInContext('session.revision',context),0);
  get('messages').scrollHeight=1300;paint();
  assert.equal(get('messages').scrollTop,1300,'after-layout scroll should use the new transcript geometry');
  release();await pending;
  get('messages').scrollHeight=1500;paint();
  assert.equal(get('messages').scrollTop,1500,'completion and Stop hiding must leave the latest answer in view');
  assert.equal(get('messages').children.at(-1).children.at(-1).textContent,'Hello — world');
  assert.equal(vm.runInContext('session.revision',context),1);
  assert.equal(get('stop').hidden,true);
  console.log('Immediate message, incremental UTF-8 stream, completion and revision UI checks passed.');
}
run().catch(error=>{release();console.error(error);process.exitCode=1;});
