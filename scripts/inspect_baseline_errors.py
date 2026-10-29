"""Create an image contact sheet for inspecting baseline mistakes."""
import json
from pathlib import Path
from PIL import Image,ImageOps,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    rows={r['sample_id']:r for r in map(json.loads,(ROOT/'runtime/datasets/indofashion_clean_01/manifest.jsonl').read_text().splitlines())}
    predictions=[json.loads(line) for line in (ROOT/'runtime/evaluations/clip_indofashion_validation_01/predictions.jsonl').read_text().splitlines()]
    errors=[p for p in predictions if p.get('prediction') and p['truth']!=p['prediction']['attributes']['garment_type']['suggested_label']]
    selected=[]
    for label in ('blouse','dress','kurta'):
        selected.extend([p for p in errors if p['truth']==label][:4])
    sheet=Image.new('RGB',(1000,330*((len(selected)+3)//4)),'white')
    draw=ImageDraw.Draw(sheet)
    for index,p in enumerate(selected):
        x=(index%4)*250;y=(index//4)*330
        with Image.open(rows[p['sample_id']]['image_path']) as source:
            image=ImageOps.contain(source.convert('RGB'),(240,265))
        sheet.paste(image,(x+(250-image.width)//2,y))
        draw.text((x+5,y+270),'Label: '+p['truth'],fill='black')
        draw.text((x+5,y+288),'Predicted: '+p['prediction']['attributes']['garment_type']['suggested_label'],fill='black')
        draw.text((x+5,y+306),p['sample_id'].replace('indofashion-validation-','ID '),fill='black')
    target=ROOT/'runtime/evaluations/clip_indofashion_validation_01/error_contact_sheet.jpg'
    sheet.save(target)
    print(target)
