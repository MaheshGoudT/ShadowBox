"""Download the licensed G1 model and only its meshes from MuJoCo Menagerie."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

from ..config import PROJECT_ROOT

REPOSITORY = 'google-deepmind/mujoco_menagerie'


def fetch(url):
    request = Request(url, headers={'User-Agent': 'ShadowBox-asset-setup'})
    with urlopen(request, timeout=60) as response:
        return response.read()


def download(revision=None, destination=None):
    if revision is None:
        revision = json.loads(fetch(f'https://api.github.com/repos/{REPOSITORY}/commits/main'))['sha']
    destination = Path(destination or PROJECT_ROOT / 'assets' / 'unitree_g1')
    destination.mkdir(parents=True, exist_ok=True)
    base = f'https://raw.githubusercontent.com/{REPOSITORY}/{revision}/unitree_g1'
    xml = fetch(base + '/g1.xml')
    model = ET.fromstring(xml)
    files = sorted({f"assets/{mesh.attrib['file']}" for mesh in model.findall('asset/mesh')})
    files += ['LICENSE', 'README.md', 'scene.xml']

    def save(name):
        if '..' in Path(name).parts or Path(name).is_absolute():
            raise ValueError('Unexpected asset path')
        data = fetch(base + '/' + quote(name))
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())

    (destination / 'g1.xml').write_bytes(xml)
    with ThreadPoolExecutor(max_workers=4) as workers:
        records = list(workers.map(save, files))
    records.append(dict(path='g1.xml', bytes=len(xml), sha256=hashlib.sha256(xml).hexdigest()))
    manifest = dict(repository=REPOSITORY, revision=revision, license='BSD-3-Clause', files=records)
    (destination / 'provenance.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', help='pin a Menagerie Git commit')
    args = parser.parse_args()
    manifest = download(args.revision)
    print(f"G1 assets: {len(manifest['files'])} files, "
          f"{sum(item['bytes'] for item in manifest['files'])/1e6:.1f} MB, revision {manifest['revision']}")


if __name__ == '__main__':
    main()
