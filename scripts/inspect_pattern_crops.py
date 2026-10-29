"""Build a small labelled train/validation contact sheet, never inspect test crops."""
import argparse
import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageOps

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Choose a new output folder.')
    rows=[json.loads(line) for line in (args.dataset/'manifest.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    selected=[]
    for label in ('solid','striped','checked','floral'):
        for split in ('train','validation'):
            candidates=[r for r in rows if r['pattern_type']==label and r['split']==split]
            groups=set()
            for row in candidates:
                if row['source_group'] in groups:continue
                groups.add(row['source_group']);selected.append(row)
                if len(groups)==3:break
    if not selected:raise ValueError('No known training/validation pattern labels.')
    sheet=Image.new('RGB',(4*230,((len(selected)+3)//4)*265),'white')
    draw=ImageDraw.Draw(sheet)
    for index,row in enumerate(selected):
        x=(index%4)*230;y=(index//4)*265
        with Image.open(args.dataset/row['image_path']) as source:
            image=ImageOps.contain(source.convert('RGB'),(210,210))
        sheet.paste(image,(x+(230-image.width)//2,y+5))
        draw.text((x+8,y+220),str(index+1)+': '+row['pattern_type']+' / '+row['split'],fill='black')
        draw.text((x+8,y+238),'Provider label; not independent review',fill='black')
    args.output.mkdir(parents=True)
    sheet.save(args.output/'contact_sheet.jpg',quality=92)
    (args.output/'examples.json').write_text(json.dumps(selected,indent=2),encoding='utf-8')
    print('Training/validation crops only:',args.output/'contact_sheet.jpg')
