"""Disjoint collected attribute assessment/final evaluation; unknown labels masked."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True); parser.add_argument('--manifest', required=True)
    parser.add_argument('--image-root', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--split', choices=['assessment', 'final_test'], default='assessment')
    parser.add_argument('--acknowledge-final-test', action='store_true')
    args = parser.parse_args()
    if args.split == 'final_test' and not args.acknowledge_final_test: raise ValueError('Acknowledge frozen one-shot final evaluation explicitly.')
    output, run = Path(args.output), Path(args.run)
    if output.exists(): raise ValueError('Use a new output directory.')
    import torch
    from PIL import Image, ImageOps
    from safetensors.torch import load_file
    from backend.research_data import load_manifest, provider_hashes, sha, value
    from backend.research_inference import load
    from backend.attribute_heads import AttributeHeads, evaluate_head
    from scripts.evaluate_research_release import final_receipt
    torch.set_num_threads(4)
    report = json.loads((run / 'report.json').read_text())
    if report.get('status') != 'attribute_heads_development_not_activated' or report['head_checkpoint_sha256'] != sha(run / 'attribute_heads.safetensors'):
        raise ValueError('Invalid attribute checkpoint/report.')
    if report['manifest_sha256'] != sha(args.manifest): raise ValueError('Collection changed since head training; preserve frozen manifest.')
    all_rows = load_manifest(args.manifest, args.image_root, provider_hashes())
    rows = [row for row in all_rows if row['split'] == args.split and row['kind'] != 'non_fabric']
    if not rows: raise ValueError('No collected attribute evaluation images.')
    model, transform, encoder = load('convnext_v2')
    if encoder != report['encoder']: raise ValueError('Frozen encoder identity mismatch.')
    tasks = {name: head['task'] for name, head in report['heads'].items()}
    heads = AttributeHeads(tasks).eval(); heads.load_state_dict(load_file(str(run / 'attribute_heads.safetensors')), strict=True)
    identity = {'checkpoint_sha256': report['head_checkpoint_sha256'], 'encoder': encoder, 'head_report_sha256': sha(run / 'report.json')}
    receipt = final_receipt(ROOT / 'runtime/evaluations/final_test_ledger', identity, rows, 'attribute-final-test') if args.split == 'final_test' else None
    features = []; root = Path(args.image_root).resolve()
    with torch.inference_mode():
        for row in rows:
            path = (root / row['path']).resolve()
            if not path.is_relative_to(root) or sha(path) != row['file_sha256']: raise ValueError('Photo changed or escaped root.')
            with Image.open(path) as image: tensor = transform(ImageOps.exif_transpose(image).convert('RGB')).unsqueeze(0)
            features.append(model.encode(tensor))
        features = torch.cat(features)
        results, regional = {}, {}
        for name, task in tasks.items():
            logits = heads.heads[name](features)
            results[name] = evaluate_head(logits, rows, name, task) if any(value(r, name) is not None for r in rows) else {'status': 'no_known_truth'}
            regional[name] = {}
            for region in ('sri_lankan', 'global', 'unknown'):
                indices = [i for i, row in enumerate(rows) if row['region'] == region]
                subset = [rows[i] for i in indices]
                regional[name][region] = evaluate_head(logits[indices], subset, name, task) if any(value(r, name) is not None for r in subset) else {'status': 'no_known_truth'}
    result = {'split': args.split, 'identity': identity, 'manifest_sha256': sha(args.manifest),
              'n': len(rows), 'specimens': len({r['specimen_id'] for r in rows}), 'heads': results, 'by_region': regional, 'final_receipt': receipt,
              'notice': 'Unknown labels masked, unsupported known values counted as errors. Image-weighted metrics; permissions/labels/independence require expert review. Head probabilities remain uncalibrated. No activation or tuning on final outcomes.'}
    output.mkdir(parents=True); (output / 'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'split': args.split, 'n': len(rows), 'heads': list(results)}))


if __name__ == '__main__': main()
