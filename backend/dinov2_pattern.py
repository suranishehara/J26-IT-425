"""Separate experimental DINOv2-Small pattern encoder/head; not an app replacement."""
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME',str(ROOT/'runtime/models'))
os.environ.setdefault('HF_HUB_DISABLE_XET','1')
MODEL_ID='facebook/dinov2-small'
MODEL_REVISION='ed25f3a31f01632728cabb09d1542f84ab7b0056'
RELEASE=ROOT/'config/dinov2_pattern_release.json'


def configured_run():
    if not RELEASE.is_file():raise ValueError('DINOv2 pattern model is not configured.')
    release=json.loads(RELEASE.read_text(encoding='utf-8'))
    directory=(ROOT/release['run_directory']).resolve()
    if not directory.is_relative_to((ROOT/'runtime/adaptation').resolve()):
        raise ValueError('DINO release path must be inside runtime/adaptation.')
    report=validate_run(directory)
    if report['checkpoint_sha256']!=release.get('checkpoint_sha256') or report['labels']!=release.get('labels'):
        raise ValueError('DINO release checkpoint/labels do not match the frozen selection.')
    return directory


def available():
    try:configured_run();return True
    except (OSError,ValueError,KeyError,TypeError):return False


def development_rows(rows):
    selected=[row for row in rows if row['split'] in ('train','validation') and row['pattern_type']!='unknown']
    if not selected:raise ValueError('No known train/validation pattern targets.')
    return selected


def aligned_validation(rows,baseline_predictions,labels):
    expected={r['sample_id']:r['pattern_type'] for r in rows if r['split']=='validation' and r['pattern_type']!='unknown'}
    predictions={}
    for row in baseline_predictions:
        if row['attribute']!='pattern_type':continue
        if row['sample_id'] in predictions:raise ValueError('Duplicate baseline prediction.')
        predictions[row['sample_id']]=row
    if set(expected)!=set(predictions):raise ValueError('Baseline and DINO validation sample IDs differ.')
    for identifier,truth in expected.items():
        row=predictions[identifier]
        if row['truth']!=truth or row['baseline'] not in labels or row['adapted'] not in labels:
            raise ValueError('Baseline target/candidate labels differ.')
    return predictions


class DinoEncoder:
    def __init__(self,device=None):
        import torch
        from transformers import AutoImageProcessor,AutoModel
        torch.set_num_threads(4)
        self.device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor=AutoImageProcessor.from_pretrained(MODEL_ID,revision=MODEL_REVISION)
        core=AutoModel.from_pretrained(MODEL_ID,revision=MODEL_REVISION,use_safetensors=True).eval().to(self.device)
        core.requires_grad_(False)
        class Features(torch.nn.Module):
            def __init__(self,encoder):
                super().__init__();self.encoder=encoder
            def get_image_features(self,pixel_values):
                return self.encoder(pixel_values=pixel_values).last_hidden_state[:,0,:]
        self.model=Features(core).eval()


def validate_run(directory):
    from local_attributes import taxonomy_hash
    from taxonomy import ATTRIBUTES
    directory=Path(directory)
    report=json.loads((directory/'report.json').read_text(encoding='utf-8'))
    if report.get('model')!=MODEL_ID or report.get('revision')!=MODEL_REVISION or report.get('taxonomy_sha256')!=taxonomy_hash():
        raise ValueError('DINO pattern run model/revision/taxonomy mismatch.')
    path=directory/'pattern_head.safetensors'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=report.get('checkpoint_sha256'):
        raise ValueError('DINO pattern checkpoint checksum mismatch.')
    labels=report['labels']
    if len(labels)<2 or len(set(labels))!=len(labels) or not set(labels)<=set(ATTRIBUTES['pattern_type']):
        raise ValueError('Invalid DINO pattern labels.')
    if report.get('embedding_dim')!=384:raise ValueError('Unexpected DINO embedding size.')
    return report


def load_pattern_head(directory,device):
    import torch
    from safetensors.torch import load_file
    directory=Path(directory)
    report=validate_run(directory)
    path=directory/'pattern_head.safetensors'
    head=torch.nn.Linear(report['embedding_dim'],len(report['labels'])).to(device)
    head.load_state_dict(load_file(str(path),device=device))
    return head.eval(),report


def explain_pattern(encoder,head,report,image,label):
    import numpy as np
    import torch
    from PIL import Image
    from explanations import patch_gradcam,png_url
    captured={}
    def capture(module,inputs,output):captured['activation']=output
    core=encoder.model.encoder
    hook=core.encoder.layer[-1].norm1.register_forward_hook(capture)
    try:
        with torch.enable_grad():
            pixels=encoder.processor(images=image.convert('RGB'),return_tensors='pt')['pixel_values'].to(encoder.device)
            pixels.requires_grad_(True)
            vector=encoder.model.get_image_features(pixel_values=pixels)
            vector=vector/vector.norm(dim=-1,keepdim=True)
            score=torch.nn.functional.linear(vector,head.weight.detach().clone(),head.bias.detach().clone())[0,report['labels'].index(label)]
            activation=captured['activation']
            gradient=torch.autograd.grad(score,activation)[0]
            grid=pixels.shape[-1]//core.config.patch_size
            cam,informative=patch_gradcam(activation.detach(),gradient.detach(),grid)
            heat=torch.nn.functional.interpolate(cam[None,None],size=pixels.shape[-2:],mode='bilinear',align_corners=False)[0,0].cpu().numpy()
            processor=encoder.processor
            mean=torch.tensor(processor.image_mean,device=pixels.device)[None,:,None,None]
            std=torch.tensor(processor.image_std,device=pixels.device)[None,:,None,None]
            crop=((pixels.detach()*std+mean).clamp(0,1)[0].permute(1,2,0).cpu().numpy()*255).round().astype('uint8')
            colour=np.zeros_like(crop,dtype=np.float32);colour[:,:,0]=255;colour[:,:,1]=80
            opacity=(heat*.6)[:,:,None]
            overlay=(crop*(1-opacity)+colour*opacity).clip(0,255).astype('uint8')
            return {'status':'ok' if informative else 'uninformative','method':'DINOv2 ViT Grad-CAM',
                'model':MODEL_ID,'revision':MODEL_REVISION,'classifier_mode':'dinov2_pattern',
                'attribute':'pattern_type','label':label,'target_score':'DINOv2 pattern head pre-softmax logit',
                'target_layer':'encoder.layer[-1].norm1','patch_grid':[grid,grid],
                'model_crop':png_url(Image.fromarray(crop)),'overlay':png_url(Image.fromarray(overlay)),
                'notice':'DINOv2 pattern prediction, not CLIP. Orange/red shows positive attribution on the actual model crop. Coarse attribution, not segmentation, confidence or proof of correctness. Centre-cropped areas only; this model has development validation, not final-test validation.'}
    finally:hook.remove()


def predict_pattern(encoder,head,report,image):
    import torch
    with torch.inference_mode():
        inputs=encoder.processor(images=image.convert('RGB'),return_tensors='pt')
        features=encoder.model.get_image_features(pixel_values=inputs['pixel_values'].to(encoder.device))
        features=features/features.norm(dim=-1,keepdim=True)
        scores=head(features).softmax(-1)[0].cpu().tolist()
    ranked=sorted(zip(report['labels'],scores),key=lambda pair:-pair[1])
    return {'model':MODEL_ID,'revision':MODEL_REVISION,'checkpoint_sha256':report['checkpoint_sha256'],
        'attribute':'pattern_type','suggested_label':ranked[0][0],'relative_score':ranked[0][1],
        'trained_labels':report['labels'],'alternatives':[{'label':label,'relative_score':score} for label,score in ranked],
        'notice':'Experimental pattern-only classifier. Uncalibrated relative scores, not measured confidence. Supports only the listed classes. No garment/drape/texture or Sri Lankan validation claim.'}
