"""Generated importer fixtures; no real dataset performance assertions."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from deepfashion_import import PATTERNS,TEXTURES,read_labels,source_group,group_split,region_crop
from local_attributes import audit_manifest

class DeepFashionImportTests(unittest.TestCase):
    def test_only_direct_mappings(self):
        self.assertEqual(PATTERNS,{0:'floral',2:'striped',3:'solid',4:'checked'})
        self.assertNotIn(1,PATTERNS)
        self.assertNotIn(7,PATTERNS)
        self.assertEqual(TEXTURES,{4:'knitted'})
    def test_garment_views_share_group_and_split(self):
        a=source_group('WOMEN-Blouses_Shirts-id_00000001-01_1_front.jpg')
        b=source_group('WOMEN-Blouses_Shirts-id_00000001-02_2_back.jpg')
        self.assertEqual(a,b)
        self.assertEqual(group_split(a),group_split(b))
        self.assertNotEqual(a,source_group('MEN-Tees_Tanks-id_00000001-01_1_front.jpg'))
        with self.assertRaises(ValueError):source_group('random.jpg')
    def test_crop_excludes_other_clothes_and_skin(self):
        pixels=np.zeros((100,80),dtype='uint8');pixels[10:90,10:70]=1;pixels[20:30,30:40]=15
        result=region_crop(Image.new('RGB',(80,100),'red'),Image.fromarray(pixels),'upper',100)
        crop,box,area=result
        self.assertEqual(box,(10,10,70,90))
        self.assertEqual(area,4700)
        self.assertEqual(crop.getpixel((25,15)),(245,245,245))
    def test_rgb_masks_misalignment_and_dresses_rejected(self):
        image=Image.new('RGB',(80,100))
        with self.assertRaises(ValueError):region_crop(image,Image.new('RGB',image.size),'upper')
        with self.assertRaises(ValueError):region_crop(image,Image.new('L',(1,1)),'upper')
        self.assertIsNone(region_crop(image,Image.new('L',image.size,4),'upper'))
    def test_strict_label_parser(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'labels.txt'
            for text in ('../bad.jpg 0 1 2','x.jpg 0 1','x.jpg 0 1 8','x.jpg 0 1 2\nx.jpg 0 1 2'):
                path.write_text(text)
                with self.assertRaises(ValueError):read_labels(path)
            path.write_text('WOMEN-Tees_Tanks-id_00000001-01_1_front.jpg 0 2 7')
            self.assertEqual(next(iter(read_labels(path).values())),[0,2,7])
    def test_import_cli_retains_provider_provenance_and_raw_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);images=root/'image';masks=root/'parsing';images.mkdir();masks.mkdir()
            name='WOMEN-Tees_Tanks-id_00000001-01_1_front.jpg'
            Image.new('RGB',(100,100),'blue').save(images/name)
            Image.new('L',(100,100),1).save(masks/(Path(name).stem+'_segm.png'))
            labels=root/'patterns.txt';labels.write_text(name+' 0 2 7')
            raw=(images/name).read_bytes();output=root/'derived'
            subprocess.run([sys.executable,str(ROOT/'scripts/prepare_deepfashion_multimodal.py'),
                '--images',str(images),'--parsing',str(masks),'--mask-suffix','_segm','--pattern-labels',str(labels),'--output',str(output),
                '--assign-group-splits','--confirm-permission','--permission-reference','Generated software fixture only'],check=True,capture_output=True)
            row=json.loads((output/'manifest.jsonl').read_text())
            self.assertEqual(row['review_status'],'provider_annotated')
            self.assertEqual(row['pattern_type'],'floral')
            self.assertEqual(row['fabric_texture'],'unknown')
            self.assertEqual(row['garment_type'],'unknown')
            self.assertEqual((images/name).read_bytes(),raw)
            audited,audit=audit_manifest(output/'manifest.jsonl',output)
            self.assertEqual(len(audited),1)
            self.assertEqual(audit['counts'][row['split']]['annotation_status']['provider_annotated'],1)
            row.pop('label_provenance')
            (output/'manifest.jsonl').write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError,'provenance'):audit_manifest(output/'manifest.jsonl',output)
