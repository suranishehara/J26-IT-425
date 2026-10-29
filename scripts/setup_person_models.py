"""Download official model assets only; photographs are never uploaded."""
import hashlib
import json
from pathlib import Path
import urllib.request
from huggingface_hub import snapshot_download

ROOT=Path(__file__).resolve().parents[1]
POSE_URL='https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'
SEGMENTER='mattmdjaga/segformer_b2_clothes'
REVISION='584abc1e1d260e23c0fc627c5217a09b2b461046'
if __name__=='__main__':
    target=ROOT/'runtime/models/person'
    target.mkdir(parents=True,exist_ok=True)
    pose=target/'pose_landmarker_lite.task'
    if not pose.is_file():
        temporary=target/'pose_landmarker_lite.task.download'
        urllib.request.urlretrieve(POSE_URL,temporary)
        temporary.replace(pose)
    directory=target/'segformer'
    required=['config.json','preprocessor_config.json','model.safetensors']
    if not all((directory/name).is_file() for name in required):
        snapshot_download(SEGMENTER,revision=REVISION,allow_patterns=required,local_dir=directory)
    hashes={str(path.relative_to(target)):hashlib.sha256(path.read_bytes()).hexdigest() for path in [pose,*[directory/name for name in required]]}
    (target/'assets.json').write_text(json.dumps({'pose_url':POSE_URL,'segformer':SEGMENTER,'revision':REVISION,'sha256':hashes},indent=2))
    print('Person model assets ready:',target)
