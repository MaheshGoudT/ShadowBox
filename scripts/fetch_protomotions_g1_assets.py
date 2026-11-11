"""Materialize pinned G1 Git LFS assets in a local qualification copy."""
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'tmp/playground/protomotions-source/protomotions/data/assets'
DESTINATION = ROOT/'tmp/playground/protomotions-assets'
REVISION = '13de0cccd06f53ce5c9cf9ec7d1932c7da73610f'


def main():
    if not DESTINATION.exists():
        shutil.copytree(SOURCE, DESTINATION)
    verified = []
    for pointer in sorted((SOURCE/'mesh/G1').rglob('*')):
        if not pointer.is_file():
            continue
        prefix = pointer.read_bytes()[:256]
        if not prefix.startswith(b'version https://git-lfs.github.com/spec/v1'):
            continue
        lines = prefix.decode().splitlines()
        digest = next(line.split('sha256:')[1] for line in lines if line.startswith('oid '))
        size = int(next(line.split()[1] for line in lines if line.startswith('size ')))
        target = DESTINATION/pointer.relative_to(SOURCE)
        if target.stat().st_size == size and hashlib.sha256(target.read_bytes()).hexdigest() == digest:
            verified.append(dict(path=str(target.relative_to(DESTINATION)),sha256=digest,size=size))
            continue
        rel = pointer.relative_to(ROOT/'tmp/playground/protomotions-source')
        url = f'https://media.githubusercontent.com/media/NVlabs/ProtoMotions/{REVISION}/{rel.as_posix()}'
        with urllib.request.urlopen(url, timeout=60) as response:
            content = response.read(size+1)
        if len(content) != size or hashlib.sha256(content).hexdigest() != digest:
            raise ValueError(f'LFS size/hash mismatch: {rel}')
        temporary = target.with_suffix(target.suffix+'.partial')
        temporary.write_bytes(content)
        temporary.replace(target)
        verified.append(dict(path=str(target.relative_to(DESTINATION)),sha256=digest,size=size))
        print('Verified',rel,flush=True)
    (DESTINATION/'g1-asset-manifest.json').write_text(json.dumps(dict(source_revision=REVISION,files=verified),indent=2)+'\n')
    print('Verified G1 LFS assets:',len(verified))


if __name__ == '__main__':
    main()
