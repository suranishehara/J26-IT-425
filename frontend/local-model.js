// Keep the existing baselines available; local mode needs an explicitly activated run.
const localModelOption=document.createElement('option');
localModelOption.value='local_attributes';localModelOption.disabled=true;
localModelOption.textContent='Local attributes · awaiting reviewed training data';
document.querySelector('#classifier-mode').append(localModelOption);
const dinoModelOption=document.createElement('option');
dinoModelOption.value='dinov2_pattern';dinoModelOption.disabled=true;
dinoModelOption.textContent='DINOv2 patterns + CLIP · restart backend if unavailable';
document.querySelector('#classifier-mode').append(dinoModelOption);
fetch('/api/health').then(r=>r.json()).then(health=>{
 if(health.local_attributes_configured){localModelOption.disabled=false;localModelOption.textContent=Object.keys(health.local_attribute_labels||{}).length===1&&health.local_attribute_labels?.drape_style?'Sri Lankan drape + CLIP · experimental':'Local attributes · trained heads + CLIP fallback';}
 if(health.dinov2_pattern_available){dinoModelOption.disabled=false;dinoModelOption.textContent='DINOv2 patterns + CLIP · experimental';}
}).catch(()=>{});
