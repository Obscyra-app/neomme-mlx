"""Download and verify the exact v0.1.0 model package."""
import argparse, hashlib, json
from pathlib import Path
from huggingface_hub import snapshot_download

REPO = 'Nurymanau/NeoMME-260M-Retriever-MLX'
REVISION = 'a2f11bf79b1a8a08c068abacf2f1f06fa056a28a'
MANIFEST_SHA256 = 'ec3bfdd601f714fb0c70acd6854f9e2096f328886ef65ef7b65130dd5a72fa48'

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output', default='model', type=Path)
    args=p.parse_args()
    path=Path(snapshot_download(REPO, revision=REVISION, local_dir=args.output))
    data=(path/'manifest.json').read_bytes()
    assert hashlib.sha256(data).hexdigest()==MANIFEST_SHA256, 'Manifest mismatch'
    for name, expected in json.loads(data)['files'].items():
        file=path/name
        with file.open('rb') as f:
            digest=hashlib.file_digest(f,'sha256').hexdigest()
        if digest!=expected['sha256'] or file.stat().st_size!=expected['bytes']:
            raise RuntimeError('Checksum mismatch: '+name)
    print('Verified model package:',path)

if __name__=='__main__':
    main()
