const form=document.querySelector('#preferences'),status=document.querySelector('#status'),results=document.querySelector('#results');
let rankedGarments=[],resultPage=0,selectedGarment=null;
const pageSize=2;
function showGarment(item){
 selectedGarment=item.id;
 const detail=document.querySelector('#detail-content');detail.replaceChildren();
 if(item.image_url){const stage=document.createElement('div');stage.className='detail-image';const image=new Image();image.src=item.image_url;image.alt=item.name;stage.append(image);detail.append(stage);}
 const title=document.createElement('h3'),info=document.createElement('p'),reasons=document.createElement('ul'),tag=document.createElement('p');
 title.textContent=item.name;info.textContent='LKR '+item.price_lkr.toLocaleString()+' · '+item.garment_type+' · '+item.pattern_type;
 for(const reason of item.reasons){const line=document.createElement('li');line.textContent=reason;reasons.append(line);}
 tag.className='record-tag';tag.textContent=item.data_status==='user_reviewed'?'Reviewed by '+item.reviewer_id:'Demonstration garment · synthetic reference image';
 const fit=document.createElement('p');fit.textContent=item.sizing?item.sizing.reason+' '+item.sizing.scope:'Size not assessed.';detail.append(title,info,fit,reasons,tag);
 for(const card of results.children)card.classList.toggle('selected',card.dataset.id===selectedGarment);
}
function renderPage(){
 results.replaceChildren();
 for(const item of rankedGarments.slice(resultPage*pageSize,(resultPage+1)*pageSize)){
 const card=document.createElement('article');card.className='garment-card';card.dataset.id=item.id;card.classList.toggle('selected',item.id===selectedGarment);
 if(item.image_url){const image=new Image();image.src=item.image_url;image.alt=item.name;card.append(image);}
 const body=document.createElement('div'),title=document.createElement('h3'),info=document.createElement('p'),button=document.createElement('button');
 title.textContent=item.name;info.textContent='LKR '+item.price_lkr.toLocaleString()+' · Score '+item.score;button.type='button';button.textContent='View garment →';button.onclick=()=>showGarment(item);body.append(title,info,button);card.append(body);results.append(card);
 }
 document.querySelector('#pagination').hidden=rankedGarments.length<=pageSize;document.querySelector('#previous-result').disabled=resultPage===0;document.querySelector('#next-result').disabled=(resultPage+1)*pageSize>=rankedGarments.length;document.querySelector('#page-label').textContent='Page '+(resultPage+1)+' of '+Math.ceil(rankedGarments.length/pageSize);
}
document.querySelector('#previous-result').onclick=()=>{if(resultPage>0){resultPage--;renderPage();}};
document.querySelector('#next-result').onclick=()=>{if((resultPage+1)*pageSize<rankedGarments.length){resultPage++;renderPage();}};
form.onsubmit=async event=>{
 event.preventDefault();const button=form.querySelector('button');button.disabled=true;
 const data=Object.fromEntries(new FormData(form));for(const key of ['colours','styles'])data[key]=data[key].split(',').map(x=>x.trim().toLowerCase()).filter(Boolean);data.budget=Number(data.budget);data.garment_types=data.garment_type?[data.garment_type]:[];data.patterns=data.pattern?[data.pattern]:[];data.only_size_compatible=form.elements.only_size_compatible.checked;if(activeUserProfile)data.user_profile=activeUserProfile;status.textContent='Finding options…';
 try{const response=await fetch('/api/recommendations',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)}),payload=await response.json();if(!response.ok)throw Error(payload.error);rankedGarments=payload.recommendations;resultPage=0;selectedGarment=null;renderPage();document.querySelector('#detail-content').replaceChildren();if(rankedGarments.length)showGarment(rankedGarments[0]);else document.querySelector('#detail-content').textContent=data.only_size_compatible?'No complete size-compatible garment records. Check measurements or turn off the size filter.':'No matching garments within budget.';status.textContent=rankedGarments.length?rankedGarments.length+' ranked options · '+payload.notice+(payload.profile_context?.measurements_provided?(payload.profile_context.fit_assessed?' Body-size charts assessed where available.':' Measurements supplied; no complete size comparison available.'):''):(data.only_size_compatible?'No size-compatible records. Add sizing charts and required measurements, or turn off the size filter.':'No options within this budget. Try a higher budget.');}catch(error){status.textContent=error.message;}finally{button.disabled=false;}
};
