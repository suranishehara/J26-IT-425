const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
class Node{
 constructor(){this.children=[];this.value='';this.textContent='';this.classList={add(){},remove(){}};}
 append(...items){this.children.push(...items);}
 replaceChildren(...items){this.children=items;}
 setAttribute(name,value){this[name]=value;}
 querySelector(selector){return selector.startsWith('option')?option:null;}
}
const map=new Map(),option=new Node(),get=id=>{if(!map.has(id))map.set(id,new Node());return map.get(id);};
get('generation').value='baseline';get('retrieval').value='bm25';
const id='a'.repeat(32),date='2026-10-19T00:00:00+00:00';
let data={enabled:true,chats:[{id,title:'Original chat',revision:1,updated:date}],deleted:[]},confirmed=true,prompt='Renamed chat',posts=0;
const result={revision:1,answer:'Saved response',state:{occasion:'wedding'},understanding:{intent:'knowledge_question',entities:[]},evidence:[],knowledge_sha256:'old-hash',context_used:false,methods:{retrieval:'bm25',generation:'extractive_template_no_llm'}};
const saved={id,title:'Original chat',revision:1,state:result.state,turns:[{user:'Saved question',result}],component_context:null};
const context=vm.createContext({document:{getElementById:get,createElement:()=>new Node(),createTextNode:text=>({textContent:text})},TextDecoder,Uint8Array,AbortController,
 window:{addEventListener(){},confirm:()=>confirmed,prompt:()=>prompt},
 fetch:async(url,options)=>{
  let value;
  if(url==='/api/chats')value=data;
  else if(url==='/api/chats/'+id)value=saved;
  else if(url==='/api/models')value={available:false,models:[],notice:'Fixture'};
  else if(url==='/api/retrieval')value={available:false,notice:'Fixture'};
  else {
   posts++;const payload=JSON.parse(options.body);
   if(url==='/api/chats/rename'){data.chats[0].title=payload.title;saved.title=payload.title;}
   else if(url==='/api/chats/delete'){data.deleted.push(...data.chats);data.chats=[];}
   else if(url==='/api/chats/restore'){data.chats.push(...data.deleted);data.deleted=[];}
   else throw Error('Unexpected fixture route '+url);
   value=data;
  }
  return {ok:true,json:async()=>JSON.parse(JSON.stringify(value))};
 }});
vm.runInContext(fs.readFileSync(require('node:path').join(__dirname,'../frontend/app.js'),'utf8'),context);
async function run(){
 await new Promise(resolve=>setImmediate(resolve));
 assert.equal(vm.runInContext('session.session_id',context),id,'startup reopens a persisted chat');
 assert.equal(get('messages').children[0].children.at(-1).textContent,'Saved question');
 assert.equal(get('messages').children[1].children.at(-1).textContent,'Saved response');
 assert.match(get('methods').textContent,/old-hash/);
 vm.runInContext('busy=true',context);await vm.runInContext(`renameSavedChat({id:'${id}',title:'Original chat'})`,context);assert.equal(posts,0);
 vm.runInContext('busy=false',context);
 prompt='<img src=x onerror=alert(1)>';
 await vm.runInContext(`renameSavedChat({id:'${id}',title:'Original chat'})`,context);
 assert.equal(get('history-list').children[0].children[0].textContent,prompt,'titles are literal text');
 confirmed=false;await vm.runInContext(`deleteSavedChat({id:'${id}',title:'test'})`,context);assert.equal(posts,1,'cancelled delete has no mutation');
 confirmed=true;await vm.runInContext(`deleteSavedChat({id:'${id}',title:'test'})`,context);
 assert.equal(vm.runInContext('session',context),null);
 assert.equal(get('messages').children.length,0);
 assert.equal(data.deleted.length,1);
 await vm.runInContext(`restoreSavedChat('${id}')`,context);assert.equal(data.chats.length,1);
 await vm.runInContext(`openSavedChat('${id}')`,context);assert.equal(vm.runInContext('session.revision',context),1);
 get('message').value='Unsent private draft';confirmed=false;
 await vm.runInContext(`openSavedChat('${id}')`,context);assert.equal(get('message').value,'Unsent private draft');
 console.log('History startup/load, literal rename, cancel/delete, restore, busy guards and draft protection passed.');
}
run().catch(error=>{console.error(error);process.exitCode=1;});
