const $ = id => document.getElementById(id);
let session = null, busy = false, controller = null, lastCompleted = null;
let historyEnabled = false;
const historyActions = new Set();
let latestScrollFrame = null;
function followLatest() {
  const scroll = () => { $('messages').scrollTop = $('messages').scrollHeight; };
  scroll();
  // Re-pin after layout: Stop/status/textarea changes can shrink the transcript
  // after a synchronous scroll has already used the previous viewport height.
  if (typeof requestAnimationFrame === 'function' && latestScrollFrame === null) {
    latestScrollFrame = requestAnimationFrame(() => { latestScrollFrame = null; scroll(); });
  }
}
function element(tag, text, cls) { const node = document.createElement(tag); node.textContent = text; if (cls) node.className = cls; return node; }
async function post(url, payload) {
  const response = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
  const result = await response.json(); if (!response.ok) throw Error(result.error || 'Request failed'); return result;
}
function lock(value) { busy = value; $('send').disabled = value || !session; $('history-new').disabled=value; for(const button of historyActions)button.disabled=value; for (const id of ['generation','model','refresh-models','retrieval']) $(id).disabled=value || (id === 'model' && $('generation').value !== 'ollama'); }
function clearChatView() {
  lastCompleted=null; $('messages').replaceChildren(); $('constraints').replaceChildren(); $('entities').replaceChildren();
  $('evidence').replaceChildren(); $('intent').textContent='Intent: —'; $('component-json').value=''; $('methods').textContent=''; $('message').value='';
}
async function refreshHistory() {
  const response=await fetch('/api/chats');
  if(!response.ok)throw Error('Could not load saved chat history.');
  const data=await response.json(); renderHistory(data); return data;
}
function renderHistory(data) {
  historyEnabled=Boolean(data.enabled); historyActions.clear();
  $('history-list').replaceChildren(); $('history-deleted').replaceChildren();
  $('history-notice').textContent=historyEnabled?'Completed chats saved on this computer. Not automatically used for training.':'History unavailable in this service; chats remain in memory only.';
  if(!data.chats.length)$('history-list').append(element('p','No saved chats yet.','hint'));
  const actionButton=(text,handler)=>{const button=element('button',text);button.type='button';button.disabled=busy;button.onclick=handler;historyActions.add(button);return button;};
  for(const chat of data.chats) {
    const entry=element('article','','history-entry'+(session?.session_id===chat.id?' active':''));
    const title=actionButton(chat.title,()=>openSavedChat(chat.id));title.className='history-title';title.title='Open '+chat.title;
    if(session?.session_id===chat.id)title.setAttribute('aria-current','true');
    entry.append(title,element('p',chat.revision+' turn(s) · '+new Date(chat.updated).toLocaleDateString(),'history-date'));
    const actions=element('div','','history-actions');
    actions.append(actionButton('Rename',()=>renameSavedChat(chat)),actionButton('Delete',()=>deleteSavedChat(chat)));
    entry.append(actions);$('history-list').append(entry);
  }
  if(!data.deleted.length)$('history-deleted').append(element('p','No deleted chats.','hint'));
  for(const chat of data.deleted) {
    const entry=element('article','','history-entry');entry.append(element('p',chat.title),actionButton('Restore',()=>restoreSavedChat(chat.id)));
    $('history-deleted').append(entry);
  }
}
async function openSavedChat(id) {
  if(busy)return;
  if($('message').value.trim()&&!window.confirm('Switch chats and discard this unsent draft?'))return;
  lock(true);
  try {
    const response=await fetch('/api/chats/'+encodeURIComponent(id));const saved=await response.json();
    if(!response.ok)throw Error(saved.error||'Could not open this chat.');
    session={session_id:saved.id,revision:saved.revision,state:saved.state};clearChatView();
    for(const turn of saved.turns){message('user',turn.user);message('assistant',turn.result.answer);lastCompleted=turn.result;}
    if(lastCompleted)renderMetadata(lastCompleted);
    else message('assistant','Ask a fashion question or supply a component result for a grounded explanation.');
    if(saved.component_context){$('component').value=String(saved.component_context.component);$('component-json').value=JSON.stringify(saved.component_context.result,null,2);}
    $('status').textContent='Saved chat reopened · '+saved.revision+' completed turn(s). Older source/model details are historical, not reverified.';
    await refreshHistory();followLatest();
  }catch(error){$('history-feedback').textContent=error.message;}
  finally{lock(false);}
}
async function renameSavedChat(chat) {
  if(busy)return;const title=window.prompt('Rename this chat (1–80 characters):',chat.title);if(title===null)return;
  lock(true);
  try{renderHistory(await post('/api/chats/rename',{session_id:chat.id,title}));$('history-feedback').textContent='Chat renamed.';}
  catch(error){$('history-feedback').textContent=error.message;}finally{lock(false);}
}
async function deleteSavedChat(chat) {
  if(busy||!window.confirm('Delete “'+chat.title+'” from history? It will move to Recently deleted and can be restored.'))return;
  lock(true);
  try{
    const active=session?.session_id===chat.id;
    const data=await post('/api/chats/delete',{session_id:chat.id});
    if(active){session=null;clearChatView();$('status').textContent='Chat moved to Recently deleted. Choose a saved chat or start a new one.';}
    renderHistory(data);$('history-feedback').textContent='Moved to Recently deleted. Data remains locally recoverable until restored; this is not permanent erasure.';
  }catch(error){$('history-feedback').textContent=error.message;}finally{lock(false);}
}
async function restoreSavedChat(id) {
  if(busy)return;lock(true);
  try{renderHistory(await post('/api/chats/restore',{session_id:id}));$('history-feedback').textContent='Chat restored. Select its title to reopen it.';}
  catch(error){$('history-feedback').textContent=error.message;}finally{lock(false);}
}
function message(role, text) {
  $('messages').querySelector('.empty')?.remove();
  const node = element('article', '', 'message ' + role); node.append(element('strong', role === 'user' ? 'YOU' : 'ADVISOR'), document.createTextNode(text));
  $('messages').append(node); followLatest(); return node;
}
function engineStatus() {
  $('model').disabled=busy || $('generation').value!=='ollama';
  $('engine-status').textContent=$('generation').value==='ollama' ? 'Local LLM + selected evidence retrieval + recent conversation · real streamed output. Intent/entity extraction remains rule-based.' : 'Baseline mode: templates + selected retrieval. A language model is not generating these answers.';
}
async function refreshRetrieval(initial=false) {
  try {
    const response=await fetch('/api/retrieval'); if(!response.ok) throw Error('Cannot check embeddings.');
    const data=await response.json();
    for(const name of ['dense','hybrid']) $('retrieval').querySelector(`option[value="${name}"]`).disabled=!data.available;
    if(!data.available) $('retrieval').value='bm25';
    $('retrieval-status').textContent=data.available ? 'Local all-minilm:22m embeddings available. Hybrid combines keyword and semantic ranks; scores are not confidence. Semantic search can retrieve unrelated passages.' : data.notice;
  } catch(error) { $('retrieval').value='bm25'; $('retrieval-status').textContent='Embedding check unavailable. BM25 selected.'; }
}
async function refreshModels(initial=false) {
  $('refresh-models').disabled=true;
  try {
    const response=await fetch('/api/models'); if (!response.ok) throw Error('Could not check local models.');
    const data=await response.json(); const previous=$('model').value;
    $('model').replaceChildren();
    for (const item of data.models) { const option=element('option',item.name); option.value=item.name; $('model').append(option); }
    const available=data.available && data.models.length>0;
    $('generation').querySelector('option[value="ollama"]').disabled=!available;
    if (available) {
      $('model').value=data.models.some(item=>item.name===previous)?previous:(data.models.some(item=>item.name==='qwen3:4b')?'qwen3:4b':data.models[0].name);
      if (initial) $('generation').value='ollama';
      engineStatus();
    } else {
      $('generation').value='baseline'; engineStatus();
      $('model').append(element('option','No local model connected'));
      $('engine-status').textContent=data.notice+' Baseline remains available; LLM mode is not active.';
    }
  } catch(error) { $('generation').value='baseline'; engineStatus(); $('engine-status').textContent=error.message; }
  finally { $('refresh-models').disabled=busy; }
}
$('generation').onchange=engineStatus;
$('refresh-models').onclick=()=>Promise.all([refreshModels(false),refreshRetrieval(false)]);
$('stop').onclick=()=>controller?.abort();
async function reset() {
  if(busy)return;
  if($('message').value.trim()&&typeof window!=='undefined'&&!window.confirm('Start a new chat and discard this unsent draft?'))return;
  lock(true);
  try {
    session = await post('/api/session', {}); clearChatView();
    $('status').textContent=historyEnabled?'Ready · completed turns will be saved locally.':'Ready · this service has no persistent history.'; message('assistant','Ask a fashion question or supply a component result for a grounded explanation.');
    await refreshHistory();
  } catch (error) { $('status').textContent=error.message; }
  finally { lock(false); }
}
$('history-new').onclick = reset;
$('chat-form').onsubmit = async event => {
  event.preventDefault(); if (busy || !session) return;
  const text = $('message').value.trim(); if (!text) return;
  lock(true); let pending=null, completed=false, received='', content=null;
  try {
    const raw = $('component-json').value.trim();
    const component_context = raw ? {component:Number($('component').value), result:JSON.parse(raw)} : null;
    const generation=$('generation').value, model=$('model').value;
    if (generation==='ollama' && !model) throw Error('Select an installed local model.');
    message('user',text); $('message').value=''; pending=message('assistant','Thinking…'); pending.classList.add('streaming');
    content=document.createTextNode('Thinking…'); pending.replaceChildren(element('strong','ADVISOR'),content);
    controller=new AbortController(); $('stop').hidden=false;
    $('status').textContent=generation==='ollama'?'Retrieving evidence and waiting for the local model…':'Preparing the baseline response…';
    followLatest();
    const response=await fetch('/api/chat/stream',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({session_id:session.session_id,expected_revision:session.revision,message:text,component_context,generation,model,retrieval:$('retrieval').value}),signal:controller.signal});
    if (!response.ok) { const error=await response.json(); throw Error(error.error||'Request failed'); }
    if (!response.body) throw Error('This browser does not expose streaming responses.');
    const reader=response.body.getReader(), decoder=new TextDecoder(); let buffer='';
    const handle=event=>{
      if(event.type==='error') throw Error(event.error);
      if(event.type==='start') { renderMetadata(event.data); $('status').textContent=event.data.methods.generation==='deterministic_clarification_no_llm'?'Clarifying the requested option…':(generation==='ollama'?'Local model is generating…':'Baseline response ready…'); }
      if(event.type==='delta') { received+=event.text; content.textContent=received; followLatest(); }
      if(event.type==='done') {
        const result=event.data; completed=true; session.revision=result.revision; lastCompleted=result; content.textContent=result.answer; renderMetadata(result);
        const engine=result.methods.generation==='deterministic_clarification_no_llm'?'clarification (no LLM)':(result.model||'baseline');
        $('status').textContent=`Turn ${result.revision} · ${engine} · ${result.context_used?'supplied component context':'knowledge only'} · no upstream changes made.`;
        if(result.citation_check?.unknown_ids.length || result.citation_check?.missing_citation) $('status').textContent+=' Citation IDs need review (this is not factual verification).';
        followLatest();
        refreshHistory().catch(error=>{$('history-feedback').textContent=error.message;});
      }
    };
    try {
      while(true) { const {value,done}=await reader.read(); buffer+=decoder.decode(value||new Uint8Array(),{stream:!done});
        let newline; while((newline=buffer.indexOf('\n'))>=0) { const line=buffer.slice(0,newline).trim(); buffer=buffer.slice(newline+1); if(line) handle(JSON.parse(line)); }
        if(done) break;
      }
      if(buffer.trim()) handle(JSON.parse(buffer));
      if(!completed) throw Error('Response stream ended before completion.');
    } finally { reader.releaseLock(); }
  } catch (error) {
    if (!completed && content && !received) content.textContent=error.name==='AbortError'?'Response stopped.':'Response unavailable.';
    if (!completed && !$('message').value) $('message').value=text;
    if (!completed) {
      if (lastCompleted) renderMetadata(lastCompleted);
      else { $('constraints').replaceChildren(); $('entities').replaceChildren(); $('evidence').replaceChildren(); $('intent').textContent='Intent: —'; $('methods').textContent='No completed turn yet.'; }
    }
    if(pending && !completed) pending.append(element('p',received?'Incomplete response — do not treat this as a completed answer.':'No completed answer. Check the selected engine/model and retry.','hint'));
    $('status').textContent=error.name==='AbortError'?'Stopped. Partial text is not a completed answer; start a new conversation if a revision error occurs.':error.message;
    if(controller) controller.abort();
  }
  finally { if(pending) pending.classList.remove('streaming'); controller=null; $('stop').hidden=true; lock(false); followLatest(); }
};
function renderMetadata(result) {
    $('intent').textContent='Intent: '+result.understanding.intent.replaceAll('_',' ');
    $('constraints').replaceChildren();
    for (const [key,value] of Object.entries(result.state)) { $('constraints').append(element('dt',key.replaceAll('_',' ')),element('dd',Array.isArray(value)?(value.join(', ')||'—'):String(value))); }
    $('entities').replaceChildren();
    for (const item of result.understanding.entities) $('entities').append(element('p',`${item.mention} → ${item.value}${item.negated?' (excluded)':''}`,'hint'));
    $('evidence').replaceChildren();
    if (!result.evidence.length) $('evidence').append(element('p',result.methods.retrieval==='not_run_clarification'?'Evidence lookup was not run: the requested topic or option needs clarification.':'No matching source retrieved.','hint'));
    for (const item of result.evidence) {
      const card = element('article','','source'); card.append(element('h3',`[${item.id}] ${item.title}`),element('p',item.text),element('p','Scope: '+item.scope));
      if (item.source.url && /^https:\/\//.test(item.source.url)) { const a=element('a',item.source.title); a.href=item.source.url; a.target='_blank'; a.rel='noopener noreferrer'; card.append(a); }
      else card.append(element('p',item.source.title+' · '+item.source.path));
      card.append(element('p',item.region+' · '+item.status.replaceAll('_',' '),'hint')); $('evidence').append(card);
    }
    $('methods').textContent=`${result.methods.retrieval} · ${result.methods.generation} · knowledge SHA-256 ${result.knowledge_sha256}`;
    if(result.conversation?.topic_reused) $('methods').textContent+=' · Follow-up: previous user topic retained.';
    if(result.conversation?.selected_item_indices?.length) $('methods').textContent+=' · Supplied option(s): '+result.conversation.selected_item_indices.map(index=>index+1).join(', ');
}
if (typeof ResizeObserver === 'function') {
  const chatLayoutObserver = new ResizeObserver(followLatest);
  for (const id of ['messages', 'chat-form', 'status']) chatLayoutObserver.observe($(id));
}
if (typeof window !== 'undefined') window.addEventListener('resize', followLatest);
async function initializeChats() {
  try{
    const history=await refreshHistory();
    if(history.chats.length)await openSavedChat(history.chats[0].id);
    else await reset();
  }catch(error){$('history-feedback').textContent=error.message;await reset();}
}
initializeChats();
refreshModels(true);
refreshRetrieval(true);
