#!/usr/bin/env python3
"""Download official benchmark bytes and verify every extracted raw file."""
import argparse
import hashlib
import io
import json
import os
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'electricity': ['https://openml.org/data/v1/download/2419/electricity.arff',
                    'https://api.openml.org/data/v1/download/2419/electricity.arff'],
    'bank': ['https://archive.ics.uci.edu/static/public/222/bank+marketing.zip',
             'https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip'],
    'gas': ['https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip',
            'https://archive.ics.uci.edu/ml/machine-learning-databases/00224/Dataset.zip'],
}
EXPECTED = json.loads((ROOT / 'configs/raw_manifest.json').read_text())


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def members(blob, depth=0):
    if depth > 3:
        raise ValueError('Unexpected nested archive depth')
    result = {}
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        if sum(i.file_size for i in archive.infolist()) > 150_000_000:
            raise ValueError('Archive exceeds expected benchmark size')
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename).name
            if name.endswith('.zip'):
                child = members(archive.read(info), depth + 1)
                for k, v in child.items():
                    if k in result and result[k] != v:
                        raise ValueError('Conflicting duplicate archive member: ' + k)
                    result[k] = v
            elif name in EXPECTED:
                value = archive.read(info)
                if name in result and result[name] != value:
                    raise ValueError('Conflicting duplicate archive member: ' + name)
                result[name] = value
    return result


def verify(name, blob):
    expected = EXPECTED[name]
    if len(blob) != expected['bytes'] or digest(blob) != expected['sha256']:
        raise ValueError(f'Version/checksum mismatch for {name}; no existing raw file was changed.')


def fetch(url):
    with urlopen(Request(url, headers={'User-Agent': 'DTRA-research/0.1'}), timeout=45) as response:
        blob = response.read(60_000_001)
        if len(blob) > 60_000_000:
            raise ValueError('Download exceeds expected size')
        return blob


def download(destination, local_source=None):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    events = []
    names = {'electricity': ['electricity-normalized.arff'], 'bank': ['bank-full.csv'],
             'gas': [f'batch{i}.dat' for i in range(1, 11)]}
    local_names = {'electricity': 'electricity-normalized(1).arff',
                   'bank': 'bank+marketing.zip', 'gas': 'gas+sensor+array+drift+dataset.zip'}
    for dataset, wanted in names.items():
        if all((destination / n).exists() and digest((destination / n).read_bytes()) == EXPECTED[n]['sha256'] for n in wanted):
            print(f'VERIFIED_EXISTING dataset={dataset}', flush=True)
            events.append({'dataset': dataset, 'status': 'verified_existing'})
            continue
        candidates = [str(Path(local_source) / local_names[dataset])] if local_source else SOURCES[dataset]
        errors = []
        received = None
        for url in candidates:
            attempts = 1 if local_source else 2
            for attempt in range(attempts):
                try:
                    print(f'DOWNLOAD dataset={dataset} source={url}', flush=True)
                    blob = Path(url).read_bytes() if local_source else fetch(url)
                    files = {'electricity-normalized.arff': blob} if dataset == 'electricity' else members(blob)
                    for name in wanted:
                        if name not in files:
                            raise ValueError('Expected file not found: ' + name)
                        verify(name, files[name])
                    received = files
                    events.append({'dataset': dataset, 'status': 'downloaded_and_verified', 'source': url,
                                   'container_sha256': digest(blob)})
                    break
                except Exception as exc:
                    errors.append(f'{url}: {exc}')
                    if attempt + 1 < attempts:
                        time.sleep(1)
            if received is not None:
                break
        if received is None:
            raise RuntimeError('All verified download routes failed:\n' + '\n'.join(errors))
        for name in wanted:
            path = destination / name
            if path.exists() and digest(path.read_bytes()) != EXPECTED[name]['sha256']:
                raise FileExistsError(f'{path} differs from the fixed benchmark. Choose a new data directory.')
            temporary = path.with_suffix(path.suffix + '.part')
            temporary.write_bytes(received[name])
            os.replace(temporary, path)
        print(f'DATA_READY dataset={dataset} files={len(wanted)}', flush=True)
    manifest = {'utc': datetime.now(timezone.utc).isoformat(), 'files': EXPECTED, 'events': events}
    (destination / 'download_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/raw')
    parser.add_argument('--from-local', type=Path, help='Optional pre-existing benchmark archives for offline verification')
    args = parser.parse_args()
    try:
        download(args.data_dir, args.from_local)
    except Exception as exc:
        print(f'DOWNLOAD_FAILED: {exc}', file=sys.stderr)
        sys.exit(1)
