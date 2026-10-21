/* Offline review: no network calls, localStorage, automatic approval or training. */
(function () {
  'use strict';
  const labels=['recommend','explain','compare','update_preferences','knowledge_question','greeting','clarify'];
  const slotNames=['occasion','weather','setting','style','colours','budget_lkr'];
  const values={occasion:['wedding','office','festival','casual','temple'],weather:['hot','cool','rainy'],setting:['outdoor','indoor'],style:['traditional','formal','casual','western']};
  const object=value=>value!==null && typeof value==='object' && !Array.isArray(value);
  const normalized=text=>text.normalize('NFKC').toLowerCase().replaceAll('’',"'");
  function validateReview(review, text, canonical) {
    if(!labels.includes(review.intent)) throw Error('Choose a primary intent.');
    if(!/^[A-Za-z0-9_-]{1,40}$/.test(review.reviewer_id||'')) throw Error('Enter a pseudonymous reviewer code, such as R01.');
    if(typeof review.family_id!=='string' || !review.family_id.trim()) throw Error('Assign a paraphrase family ID.');
    if(!Array.isArray(review.entities)||!object(review.slots)||!Array.isArray(review.remove_slots)) throw Error('Fill entities [], slots {} and removals [] explicitly after checking them.');
    if(Object.keys(review.slots).some(key=>!slotNames.includes(key)) || review.remove_slots.some(key=>!slotNames.includes(key))) throw Error('Unknown slot/removal name.');
    for(const [key,value] of Object.entries(review.slots)) {
      if(values[key] && !values[key].includes(value)) throw Error('Invalid '+key+' label.');
      if(key==='budget_lkr' && (typeof value!=='number'||!Number.isFinite(value)||value<=0||value>10000000)) throw Error('Check the LKR budget.');
      if(key==='colours' && (!Array.isArray(value)||value.some(item=>typeof item!=='string'||!item))) throw Error('Colours must be a string array.');
    }
    const seen=[]; const chars=Array.from(normalized(text));
    for(const item of review.entities) {
      if(!object(item)||typeof item.mention!=='string'||!item.mention||!canonical.includes(item.value)||typeof item.negated!=='boolean') throw Error('Check entity mention, canonical value and boolean negation.');
      const mention=Array.from(normalized(item.mention));
      const matches=[];
      for(let i=0;i<=chars.length-mention.length;i++) if(chars.slice(i,i+mention.length).join('')===mention.join('')) matches.push(i);
      const start=item.start===undefined?(matches.length===1?matches[0]:null):item.start;
      if(!Number.isInteger(start)||!matches.includes(start)) throw Error('Entity mention must match uniquely, or provide its normalized start offset.');
      const end=start+mention.length;
      if(seen.some(span=>start<span[1]&&end>span[0])) throw Error('Entity spans overlap.');
      seen.push([start,end]);
    }
    return true;
  }
  if(typeof module!=='undefined' && module.exports) { module.exports={validateReview}; return; }
  const $=id=>document.getElementById(id), pack=JSON.parse($('pack').textContent);
  let index=0, dirty=false, undownloaded=false;
  const current=()=>pack.items[index];
  function showStatus(text){$('status').textContent=text;}
  function jsonField(id){const raw=$(id).value.trim();return raw?JSON.parse(raw):null;}
  function formReview(status){
    return {...current().review,status,intent:$('intent').value||null,
      reviewer_id:$('reviewer').value.trim()||null,family_id:$('family').value.trim()||null,
      entities:jsonField('entities'),slots:jsonField('slots'),remove_slots:jsonField('removals'),
      notes:$('notes').value,reviewed_at:status==='reviewed'?new Date().toISOString():null};
  }
  function save(status){
    const review=formReview(status);
    if(status==='reviewed') {
      validateReview(review,current().text,pack.canonical_entities);
      if(!$('confirm').checked) throw Error('Confirm that you checked the annotations and source contexts.');
    }
    if(status==='needs_discussion'&&!review.notes.trim()) throw Error('Record what needs discussion in Notes.');
    current().review=review; dirty=false; undownloaded=true;
    $('saved-state').textContent='In-tab status: '+status.replaceAll('_',' ')+'. Download progress to keep it.';
    progress();
  }
  function progress(){
    const done=pack.items.filter(item=>item.review.status==='reviewed').length;
    $('progress').textContent=done+' / '+pack.items.length+' reviewed · not adjudicated training labels';
  }
  function context(){
    $('context').replaceChildren();
    const sourceId=$('context-choice').value;
    const rows=Object.values(pack.conversations).find(rows=>rows.some(row=>row.id===sourceId))||[];
    for(const row of rows){const li=document.createElement('li');li.textContent=row.id+': '+row.text;if(row.id===sourceId)li.className='current';$('context').append(li);}
  }
  function render(){
    const item=current(),review=item.review;
    $('position').textContent=(index+1)+' / '+pack.items.length;
    $('jump').value=index+1;$('jump').max=pack.items.length;
    $('message').textContent=item.text;
    $('identity').textContent=item.id+' · '+item.source_ids.length+' source occurrence(s). Check contexts before assigning a shared label.';
    $('context-choice').replaceChildren();
    for(const [conversationId,rows] of Object.entries(pack.conversations))for(const row of rows)if(item.source_ids.includes(row.id)){
      const option=document.createElement('option');option.value=row.id;option.textContent=row.id+' · '+conversationId;$('context-choice').append(option);
    }
    context();
    $('suggestions').hidden=true;
    $('suggestion-text').textContent=JSON.stringify({provided_intents:item.provided_intents,rule_suggestion:item.suggestion},null,2);
    $('intent').value=review.intent||'';$('family').value=review.family_id||'';$('notes').value=review.notes||'';
    for(const [field,key] of [['entities','entities'],['slots','slots'],['removals','remove_slots']])$(field).value=review[key]===null?'':JSON.stringify(review[key],null,2);
    $('confirm').checked=false;
    $('saved-state').textContent='In-tab status: '+review.status.replaceAll('_',' ')+(review.assisted?' · assisted review':'');
    $('previous').disabled=index===0;$('next').disabled=index===pack.items.length-1;
    dirty=false;progress();
  }
  function safe(action){return async()=>{try{await action();}catch(error){showStatus(error.message);}};}
  function move(target){if(!Number.isInteger(target)||target<0||target>=pack.items.length)throw Error('Choose a valid item number.');if(dirty)save('draft');index=target;render();showStatus('');}
  for(const label of pack.intents){const option=document.createElement('option');option.value=label;option.textContent=label.replaceAll('_',' ');$('intent').append(option);}
  for(const id of ['intent','family','entities','slots','removals','notes','confirm'])$(id).addEventListener('input',()=>{dirty=true;});
  $('previous').onclick=safe(()=>move(index-1));$('next').onclick=safe(()=>move(index+1));$('go').onclick=safe(()=>move(Number($('jump').value)-1));
  $('context-choice').onchange=context;
  $('show-suggestions').onclick=()=>{current().review.assisted=true;dirty=true;undownloaded=true;$('suggestions').hidden=false;showStatus('Assisted review recorded. Suggestions still require checking.');};
  $('copy').onclick=()=>{const suggestion=current().suggestion;$('intent').value=suggestion.intent;$('entities').value=JSON.stringify(suggestion.entities,null,2);$('slots').value=JSON.stringify(suggestion.slots,null,2);$('removals').value=JSON.stringify(suggestion.remove_slots,null,2);dirty=true;showStatus('Suggestions copied as a draft, not approved. Check and correct them.');};
  for(const [id,status] of [['draft','draft'],['reviewed','reviewed'],['defer','needs_discussion']])$(id).onclick=safe(()=>{save(status);showStatus('Saved in this tab. Download progress before leaving.');});
  $('export').onclick=safe(()=>{
    if(dirty)save('draft');
    const result={schema:'nlu_review_progress_v1',pack_sha256:pack.pack_sha256,source_sha256:pack.source_sha256,
      notice:'Human reviewer records, not automatically adjudicated or split-approved.',exported_at:new Date().toISOString(),
      records:pack.items.map(item=>({id:item.id,review:item.review}))};
    const blob=new Blob([JSON.stringify(result,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),anchor=document.createElement('a');
    anchor.href=url;anchor.download='nlu_review_'+($('reviewer').value.replace(/[^A-Za-z0-9_-]/g,'')||'draft')+'.json';anchor.click();URL.revokeObjectURL(url);
    undownloaded=false;showStatus('Progress download requested. Confirm the JSON is in your Downloads folder.');
  });
  $('resume').onchange=safe(async()=>{
    if(dirty||undownloaded)throw Error('Download current work before loading a progress file.');
    const file=$('resume').files[0];if(!file)return;if(file.size>4000000)throw Error('Progress file is too large.');
    const data=JSON.parse(await file.text());
    if(data.schema!=='nlu_review_progress_v1'||data.pack_sha256!==pack.pack_sha256||data.source_sha256!==pack.source_sha256||!Array.isArray(data.records)||data.records.length!==pack.items.length)throw Error('Progress file does not match this review pack.');
    const byId=new Map(data.records.map(row=>[row.id,row.review]));if(byId.size!==pack.items.length)throw Error('Duplicate progress IDs.');
    for(const item of pack.items){const review=byId.get(item.id);if(!object(review)||!['pending','draft','reviewed','needs_discussion'].includes(review.status)||typeof review.assisted!=='boolean')throw Error('Invalid progress record.');if(review.status==='reviewed')validateReview(review,item.text,pack.canonical_entities);}
    for(const item of pack.items)item.review=byId.get(item.id);
    render();showStatus('Progress loaded. Enter your reviewer code before further saves.');
  });
  window.addEventListener('beforeunload',event=>{if(dirty||undownloaded){event.preventDefault();event.returnValue='Download progress before closing.';}});
  render();
})();
