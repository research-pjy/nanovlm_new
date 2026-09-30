"""Prepare an isolated, deterministic small-data input for the existing pipeline."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

from .artifacts import publish
from .export_metadata import build_metadata
from .verify_selection import integer, read_json


def subset(metadata, num_images, seed=42):
    integer(num_images,'num_images',3)
    if seed != 42 or metadata['random_seed'] != seed:
        raise ValueError('Small-data mode uses project seed 42')
    pools={g:[r for r in metadata['records'] if r['assignment']==g] for g in ('train','val','held_out')}
    if num_images>len(metadata['records']):
        raise ValueError('Requested more images than the verified dataset contains')
    held=max(1,round(num_images*len(pools['held_out'])/len(metadata['records'])))
    train=round((num_images-held)*0.9)
    counts={'train':train,'val':num_images-held-train,'held_out':held}
    if any(n<1 or n>len(pools[g]) for g,n in counts.items()):
        raise ValueError('Requested subset cannot represent all three existing assignments')
    rng=random.Random(seed)
    records=[]
    for group in pools:
        pool=sorted(pools[group],key=lambda r:r['image_id'])
        chosen={r['image_id'] for r in rng.sample(pool,counts[group])}
        records.extend(r for r in pools[group] if r['image_id'] in chosen)
    return {**metadata,'records':records,'assignments':counts,
            'small_data':{'num_images':num_images,'seed':seed,
                          'method':'seeded sampling within existing assignments; no reassignment'}}


def prepare(config, num_images, output):
    output=Path(output)
    root=Path(config['data_root']).expanduser().resolve()
    if output.resolve().is_relative_to(root):
        raise ValueError('Small-data output must be outside original data_root')
    metadata=subset(build_metadata(config),num_images,config['random_seed'])
    raw=(json.dumps(metadata,indent=2,ensure_ascii=False)+'\n').encode()
    splits={'schema_version':1,'random_seed':42,'train_fraction':config['train_fraction'],
            'source_manifest_sha256':metadata['source_manifest_sha256'],
            'source_annotation_sha256':metadata['source_annotation_sha256'],
            'metadata_sha256':hashlib.sha256(raw).hexdigest(),'counts':metadata['assignments'],
            'image_ids':{g:[r['image_id'] for r in metadata['records'] if r['assignment']==g]
                         for g in metadata['assignments']},'small_data':metadata['small_data']}
    payloads={'metadata.json':raw,'splits.json':(json.dumps(splits,indent=2)+'\n').encode()}
    for name,payload in payloads.items():
        path=output/name
        if path.exists() and path.read_bytes()!=payload:
            raise ValueError('Small-data inputs already differ; choose a new directory')
    for name,payload in payloads.items(): publish(output/name,payload)
    return {'num_images':num_images,'counts':metadata['assignments'],'output':str(output)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-config',required=True,type=Path)
    parser.add_argument('--num-images',required=True,type=int)
    parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args()
    try:
        config,_=read_json(args.data_config)
        print(json.dumps(prepare(config,args.num_images,args.output_dir),indent=2))
    except (OSError,ValueError,KeyError,TypeError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr)
        return 1
    return 0


if __name__=='__main__': sys.exit(main())
