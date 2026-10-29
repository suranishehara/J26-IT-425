const explainButton=document.querySelector('#explain-garment'),explainStatus=document.querySelector('#explanation-status'),explainImages=document.querySelector('#explanation-images');
let explanationVersion=0;
function clearExplanation(){explanationVersion++;explainImages.replaceChildren();explainStatus.textContent='';explainButton.disabled=!garmentData;}
upload.addEventListener('change',()=>{clearExplanation();explainButton.disabled=true;});
document.querySelector('#classifier-mode').addEventListener('change',clearExplanation);
document.querySelector('#explanation-attribute').addEventListener('change',clearExplanation);
// Explanations rerun the selected model and explain its prediction, not a manual correction.
explainButton.onclick=async()=>{
 if(!garmentData)return;
 const version=++explanationVersion;
 explainButton.disabled=true;explainImages.replaceChildren();explainStatus.textContent='Computing prediction attribution locally…';
 try{
  const response=await fetch('/api/analyse-garment',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image:garmentData,classifier_mode:document.querySelector('#classifier-mode').value,include_explanation:true,explanation_attribute:document.querySelector('#explanation-attribute').value})});
  const result=await response.json();if(!response.ok)throw Error(result.error);
  if(version!==explanationVersion)return;
  const explanation=result.explanation;
  if(!explanation)throw Error('Restart the Component 2 backend to enable visual explanations.');
  explainStatus.textContent=(explanation.model?explanation.model+' · ':'')+explanation.attribute.replaceAll('_',' ')+': '+explanation.label+' · '+explanation.notice+(explanation.status==='uninformative'?' No spatially informative positive heatmap was found.':'');
  for(const [key,caption] of [['model_crop','Actual model input crop'],['overlay','Grad-CAM overlay · predicted label, not your correction']]){
   if(!explanation[key])continue;
   const figure=document.createElement('figure'),image=new Image(),text=document.createElement('figcaption');image.src=explanation[key];image.alt=caption;text.textContent=caption;figure.append(image,text);explainImages.append(figure);
  }
 }catch(error){if(version===explanationVersion)explainStatus.textContent=error.message;}
 finally{if(version===explanationVersion)explainButton.disabled=!garmentData;}
};
// The main upload reader signals completion; no duplicate reading of the file.
upload.addEventListener('garment-ready',()=>{explainButton.disabled=!garmentData;});
