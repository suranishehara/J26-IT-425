"""Class-targeted ViT Grad-CAM on CLIP's actual 224px model crop.

Uses the normalization before the final attention block, not its final output
patches (which do not feed the class-token prediction).
"""
import base64
import io


def patch_gradcam(activations, gradients, grid_size):
    import torch
    if activations.shape != gradients.shape or activations.ndim != 3:
        raise ValueError('Expected matching batch/token/channel activations and gradients.')
    if activations.shape[0] != 1 or activations.shape[1] != grid_size * grid_size + 1:
        raise ValueError('Unexpected ViT token grid.')
    patches=activations[:,1:,:]
    weights=gradients[:,1:,:].mean(dim=1,keepdim=True)
    raw=(patches*weights).sum(-1).reshape(grid_size,grid_size).relu()
    if not torch.isfinite(raw).all():
        raise ValueError('Non-finite explanation values.')
    span=raw.max()-raw.min()
    if span.item()<=1e-12:
        return torch.zeros_like(raw),False
    return (raw-raw.min())/span,True


def png_url(image):
    buffer=io.BytesIO()
    image.save(buffer,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()


def explain(classifier,image,prediction,attribute):
    import numpy as np
    import torch
    from PIL import Image
    from taxonomy import ATTRIBUTES
    if attribute not in ATTRIBUTES:
        raise ValueError('Unsupported explanation attribute.')
    if prediction['classifier_mode']=='dinov2_pattern' and attribute=='pattern_type':
        from dinov2_pattern import explain_pattern
        return explain_pattern(classifier.dino_encoder,classifier.dino_head,classifier.dino_report,image,
                               prediction['attributes']['pattern_type']['suggested_label'])
    label=prediction['attributes'][attribute]['suggested_label']
    if prediction['attributes'][attribute].get('rule_applied'):
        return {'status':'not_applicable','attribute':attribute,'label':label,
                'notice':'This drape label came from a category rule, not an image-model score. No heatmap is generated.'}
    captured={}
    def capture(module,inputs,output):
        captured['activation']=output
    layer=classifier.model.vision_model.encoder.layers[-1].layer_norm1
    hook=layer.register_forward_hook(capture)
    try:
        with torch.enable_grad():
            pixels=classifier.processor(images=image.convert('RGB'),return_tensors='pt')['pixel_values'].to(classifier.device)
            pixels.requires_grad_(True)
            vector=classifier.model.get_image_features(pixel_values=pixels)
            vector=vector/vector.norm(dim=-1,keepdim=True)
            if prediction['classifier_mode']=='local_attributes' and attribute in classifier.local_heads:
                details=classifier.local_heads[attribute]
                head=details['head']
                scores=torch.nn.functional.linear(vector,head.weight.detach().clone(),head.bias.detach().clone())
                index=details['labels'].index(label)
                target_method='Reviewed local attribute head pre-softmax logit'
            elif attribute=='garment_type' and prediction['classifier_mode']=='indofashion_probe':
                from garment_probe import PROBE_LABELS
                head=classifier.probe
                scores=torch.nn.functional.linear(vector,head.weight.detach().clone(),head.bias.detach().clone())
                index=PROBE_LABELS.index(label)
                target_method='IndoFashion probe pre-softmax logit'
            else:
                features=classifier.text_features[attribute].detach().clone()
                scores=classifier.model.logit_scale.exp()*vector@features.T
                index=list(ATTRIBUTES[attribute]).index(label)
                target_method='CLIP image/text pre-softmax similarity logit'
            activation=captured['activation']
            gradient=torch.autograd.grad(scores[0,index],activation)[0]
            grid=pixels.shape[-1]//classifier.model.config.vision_config.patch_size
            cam,informative=patch_gradcam(activation.detach(),gradient.detach(),grid)
            heat=torch.nn.functional.interpolate(cam[None,None],size=pixels.shape[-2:],mode='bilinear',align_corners=False)[0,0].cpu().numpy()
            processor=classifier.processor.image_processor
            mean=torch.tensor(processor.image_mean,device=pixels.device)[None,:,None,None]
            std=torch.tensor(processor.image_std,device=pixels.device)[None,:,None,None]
            crop=((pixels.detach()*std+mean).clamp(0,1)[0].permute(1,2,0).cpu().numpy()*255).round().astype('uint8')
            # Transparent orange-red: stronger positive relevance has higher opacity.
            colour=np.zeros_like(crop,dtype=np.float32)
            colour[:,:,0]=255
            colour[:,:,1]=80
            opacity=(heat*.6)[:,:,None]
            overlay=(crop*(1-opacity)+colour*opacity).clip(0,255).astype('uint8')
            target_notice=''
            if prediction['classifier_mode']=='local_attributes' and attribute=='drape_style' and attribute in classifier.local_heads:
                target_notice='Explains the trained Sri Lankan drape head, not zero-shot drape matching. Nivi/Osariya development pilot only. '
            return {'status':'ok' if informative else 'uninformative','method':'ViT Grad-CAM',
                'attribute':attribute,'label':label,'classifier_mode':prediction['classifier_mode'],
                'model':classifier.model.config._name_or_path,
                'target_score':target_method,'target_layer':'vision_model.encoder.layers[-1].layer_norm1',
                'patch_grid':[grid,grid],'model_crop':png_url(Image.fromarray(crop)),
                'overlay':png_url(Image.fromarray(overlay)),
                'notice':target_notice+'Orange/red highlights positive contributions to this predicted label on the model crop. Coarse attribution, not segmentation, confidence, proof of correctness or a causal explanation. CLIP resizes and centre-crops the image; excluded areas are not explained.'}
    finally:
        hook.remove()
