"""Opt-in local person analysis; no photographs or results are persisted."""
import base64
import io
from pathlib import Path
import threading
import numpy as np
from PIL import ImageDraw

from attributes import decode_image
from person_math import estimate_skin,pose_proportions,MST_HEX,MST_SOURCE

ROOT=Path(__file__).resolve().parents[1]
LOCK=threading.Lock()
_analyser=None

class PersonAnalyser:
    def __init__(self):
        import mediapipe as mp
        import torch
        from transformers import SegformerImageProcessor,SegformerForSemanticSegmentation
        self.mp=mp;self.torch=torch
        torch.set_num_threads(4)
        model_path=ROOT/'runtime/models/person/pose_landmarker_lite.task'
        if not model_path.is_file():
            raise ValueError('Person models are missing. Run scripts/setup_person_models.py.')
        self.pose=mp.tasks.vision.PoseLandmarker.create_from_options(mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_buffer=model_path.read_bytes()),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,num_poses=2))
        directory=ROOT/'runtime/models/person/segformer'
        self.processor=SegformerImageProcessor.from_pretrained(directory,local_files_only=True)
        self.segmenter=SegformerForSemanticSegmentation.from_pretrained(directory,local_files_only=True).eval()

    def predict(self,image):
        image=image.copy();image.thumbnail((768,1024))
        rgb=np.array(image)
        result=self.pose.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB,data=rgb))
        if len(result.pose_landmarks)!=1:
            return {'status':'needs_photo','notice':'Use a clear photo of exactly one person.',
                    'body':{'status':'unavailable'},'skin':{'status':'unavailable'},'photo_persisted':False}
        points=[{'x':p.x,'y':p.y,'visibility':p.visibility} for p in result.pose_landmarks[0]]
        body=pose_proportions(points,image.width,image.height)
        with self.torch.inference_mode():
            logits=self.segmenter(**self.processor(images=image,return_tensors='pt')).logits
            logits=self.torch.nn.functional.interpolate(logits,size=(image.height,image.width),mode='bilinear',align_corners=False)
            probabilities=logits.softmax(1)
            confidence,labels=probabilities.max(1)
        skin,cheeks=estimate_skin(rgb,labels[0].numpy(),confidence[0].numpy())
        overlay=image.copy();pixels=np.array(overlay)
        pixels[cheeks]=(pixels[cheeks]*.3+np.array([40,190,150])*.7).astype('uint8')
        from PIL import Image
        overlay=Image.fromarray(pixels);draw=ImageDraw.Draw(overlay)
        for a,b in ((11,12),(11,23),(12,24),(23,24)):
            if min(points[a]['visibility'],points[b]['visibility'])>=.6:
                draw.line([(points[i]['x']*image.width,points[i]['y']*image.height) for i in (a,b)],fill='#3bb8d3',width=3)
        for i in (11,12,23,24):
            if points[i]['visibility']>=.6:
                x,y=points[i]['x']*image.width,points[i]['y']*image.height
                draw.ellipse((x-4,y-4,x+4,y+4),fill='#f0bf4c')
        buffer=io.BytesIO();overlay.save(buffer,format='PNG')
        return {'status':'analysed','body':body,'skin':skin,
                'overlay':'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode(),
                'photo_persisted':False,'models':{'pose':'MediaPipe Pose Landmarker Lite float16 v1',
                    'segmentation':'mattmdjaga/segformer_b2_clothes','segmentation_revision':'584abc1e1d260e23c0fc627c5217a09b2b461046'},
                'mst_swatches':MST_HEX,'mst_source':MST_SOURCE,
                'notice':'Review the landmarks and green cheek samples. This is a research baseline, not locally calibrated or fairness validated. No ethnicity or Fitzpatrick type is inferred.'}

def analyse_person(payload):
    global _analyser
    if not isinstance(payload,dict) or payload.get('consent') is not True:
        raise ValueError('Confirm permission to analyse this person photo.')
    image=decode_image(payload)
    with LOCK:
        if _analyser is None:
            _analyser=PersonAnalyser()
        return _analyser.predict(image)
