"""Collect and verify the frozen iNaturalist CC0 comparison dataset.

Requires Python >= 3.10, Pillow and numpy. Run from any directory:
  python final/collect_dataset.py collect
  python final/collect_dataset.py verify
  python final/collect_dataset.py restore  # Download the frozen manifest, no resampling.
API responses and spare downloads are cached under /tmp/inat-dataset-cache.
"""
import argparse
import collections
import concurrent.futures
import datetime as dt
import gzip
import hashlib
import io
import json
from pathlib import Path
import random
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
CACHE = Path('/tmp/inat-dataset-cache')
SEED = 20261115
API = 'https://api.inaturalist.org/v1/'
MANIFEST = ROOT / 'images_manifest.json'
UA = 'SHAD-educational-dataset/1.0 (400-photo CC0 comparison; no training)'
GROUP_NAMES = {
    'butterflies': 'Дневные бабочки (Papilionoidea)',
    'moths': 'Другие взрослые чешуекрылые',
    'other_insects': 'Другие насекомые',
    'immature': 'Гусеницы и куколки',
    'plants': 'Растения и природный фон',
}


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def get(url):
    for attempt in range(5):
        try:
            request = urllib.request.Request(url, headers={
                'User-Agent': UA, 'Accept-Encoding': 'gzip',
            })
            with urllib.request.urlopen(request, timeout=50) as response:
                data = response.read()
                if response.headers.get('Content-Encoding') == 'gzip':
                    data = gzip.decompress(data)
                return data
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt == 4:
                raise
            print(f'Retrying {url}: {exc}', flush=True)
            time.sleep(min(2 ** (attempt + 1), 20))


def api(endpoint, params):
    url = API + endpoint + '?' + urllib.parse.urlencode(params)
    digest = hashlib.sha256(url.encode()).hexdigest()
    path = CACHE / 'api' / (digest + '.json.gz')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        envelope = json.loads(gzip.decompress(path.read_bytes()))
    else:
        response = json.loads(get(url))
        envelope = {'request_url': url,
                    'retrieved_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                    'response': response}
        path.write_bytes(gzip.compress(json.dumps(envelope).encode()))
        time.sleep(1.2)
    return envelope


def taxon(name):
    envelope = api('taxa', {'q': name, 'is_active': 'true', 'per_page': 30})
    matches = [x for x in envelope['response']['results'] if x['name'] == name]
    if len(matches) != 1:
        raise ValueError(f'Ambiguous taxon {name}: {[x["name"] for x in matches]}')
    return matches[0]['id']


def sampling_groups():
    butterfly = taxon('Papilionoidea')
    lepidoptera = taxon('Lepidoptera')
    specs = []
    for year, bquota, mquota in zip(range(2020, 2026), [34,34,33,33,33,33], [14,14,13,13,13,13]):
        dates = {'d1': f'{year}-01-01', 'd2': f'{year}-12-31', 'term_id': 1, 'term_value_id': 2}
        specs.append({'key': f'butterflies_{year}', 'group': 'butterflies', 'quota': bquota,
                      'params': {'taxon_id': butterfly, **dates}})
        specs.append({'key': f'moths_{year}', 'group': 'moths', 'quota': mquota,
                      'params': {'taxon_id': lepidoptera, 'without_taxon_id': butterfly, **dates}})
    dates = {'d1': '2020-01-01', 'd2': '2025-12-31'}
    for name in ['Diptera', 'Hymenoptera', 'Odonata', 'Coleoptera', 'Hemiptera', 'Neuroptera']:
        specs.append({'key': name, 'group': 'other_insects', 'quota': 10,
                      'params': {'taxon_id': taxon(name), **dates}})
    for name, value, quota in [('larva', 6, 12), ('pupa', 4, 8)]:
        specs.append({'key': name, 'group': 'immature', 'quota': quota,
                      'params': {'taxon_id': lepidoptera, 'term_id': 1, 'term_value_id': value, **dates}})
    for name in ['Asteraceae', 'Fabaceae', 'Poaceae', 'Bryophyta']:
        specs.append({'key': name, 'group': 'plants', 'quota': 10,
                      'params': {'taxon_id': taxon(name), **dates}})
    return specs


def candidates(spec, page):
    # Page order is stable within the cached snapshot; selection is shuffled below.
    params = {'photos': 'true', 'photo_license': 'cc0', 'quality_grade': 'research',
              'captive': 'false', 'per_page': 200, 'page': page,
              'order_by': 'id', 'order': 'desc', **spec['params']}
    envelope = api('observations', params)
    result = []
    for obs in envelope['response']['results']:
        tax = obs.get('taxon') or {}
        ancestors = tax.get('ancestor_ids', [])
        if int(spec['params']['taxon_id']) not in ancestors + [tax.get('id')]:
            continue
        if spec['group'] == 'moths' and 47224 in ancestors:
            continue
        annotations = obs.get('annotations', [])
        if 'term_value_id' in spec['params'] and not any(
                a.get('controlled_attribute_id') == 1
                and a.get('controlled_value_id') == spec['params']['term_value_id']
                and a.get('vote_score', 0) >= 0 for a in annotations):
            continue
        photos = [p for p in obs.get('photos', []) if p.get('license_code') == 'cc0'
                  and not p.get('hidden') and not p.get('flags')]
        random.Random(SEED + obs['id']).shuffle(photos)
        for photo in photos:
            dimensions = photo.get('original_dimensions') or {}
            width, height = dimensions.get('width', 0), dimensions.get('height', 0)
            if min(width, height) < 256 or max(width, height) < 512:
                continue
            url = re.sub(r'/square\.', '/original.', photo['url'])
            if not url.startswith('https://') or '/original.' not in url:
                continue
            user = obs['user']
            result.append({
                'group': spec['group'], 'sampling_group': spec['key'],
                'observation_id': obs['id'], 'photo_id': photo['id'],
                'observation_url': obs['uri'],
                'photo_page_url': f'https://www.inaturalist.org/photos/{photo["id"]}',
                'download_url': url, 'photo_license': photo['license_code'],
                'photo_attribution': photo.get('attribution'),
                'observer_login': user['login'], 'observer_name': user.get('name'),
                'observer_url': f'https://www.inaturalist.org/people/{user["login"]}',
                'taxon_id': tax['id'], 'taxon_name': tax['name'],
                'species_id': tax.get('min_species_taxon_id') or tax['id'],
                'ancestor_taxon_ids': ancestors,
                'observed_on': obs.get('observed_on'),
                'quality_grade': obs['quality_grade'],
                'api_request_url': envelope['request_url'],
                'metadata_retrieved_at_utc': envelope['retrieved_at_utc'],
                'api_photo_record': photo,
                'life_stage_annotations': [
                    {k: a.get(k) for k in ['controlled_attribute_id', 'controlled_value_id', 'vote_score']}
                    for a in annotations if a.get('controlled_attribute_id') == 1],
                'alive_or_dead_annotations': [
                    {k: a.get(k) for k in ['controlled_attribute_id', 'controlled_value_id', 'vote_score']}
                    for a in annotations if a.get('controlled_attribute_id') == 17],
            })
            break  # At most one candidate photograph per observation.
    random.Random(f'{SEED}:{spec["key"]}:{page}').shuffle(result)
    # Filter after shuffling so that adding an exclusion preserves the order of
    # the other candidates in the cached API snapshot.
    if spec['group'] in {'butterflies', 'moths', 'other_insects'}:
        result = [r for r in result if not any(
            a.get('controlled_value_id') == 19 and a.get('vote_score', 0) >= 0
            for a in r['alive_or_dead_annotations'])]
    print(f'Candidates {spec["key"]} page={page}: {len(result)} / total {envelope["response"]["total_results"]}', flush=True)
    return result


def fingerprint(path):
    data = path.read_bytes()
    with Image.open(io.BytesIO(data)) as im:
        im.verify()
    with Image.open(io.BytesIO(data)) as im:
        im.load()  # Full decode, not just headers.
        fmt = im.format
        orientation = im.getexif().get(274, 1)
        im = ImageOps.exif_transpose(im).convert('RGB')
        width, height = im.size
        if min(width, height) < 256 or max(width, height) < 512:
            raise ValueError(f'Image too small: {im.size}')
        pixels = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
        small = np.asarray(im.resize((32, 32), Image.Resampling.LANCZOS).convert('L'), dtype=float)
        # 2D DCT without scipy; 63 low-frequency AC coefficients.
        basis = np.cos(np.pi * (2*np.arange(32)[None, :] + 1) * np.arange(8)[:, None] / 64)
        coeff = (basis @ small @ basis.T).flatten()[1:]
        bits = coeff > np.median(coeff)
        phash = sum(int(bit) << i for i, bit in enumerate(bits))
        gray = np.asarray(im.resize((9, 8), Image.Resampling.LANCZOS).convert('L'))
        dbits = (gray[:, 1:] > gray[:, :-1]).flatten()
        dhash = sum(int(bit) << i for i, bit in enumerate(dbits))
    return {'sha256': hashlib.sha256(data).hexdigest(), 'pixel_sha256': pixels,
            'phash63': f'{phash:016x}', 'dhash64': f'{dhash:016x}',
            'width': width, 'height': height, 'format': fmt,
            'exif_orientation': orientation, 'bytes': len(data)}


def download(record):
    suffix = Path(urllib.parse.urlparse(record['download_url']).path).suffix.lower()
    if suffix not in {'.jpg', '.jpeg', '.png', '.webp'}:
        raise ValueError(f'Unsupported extension {suffix}')
    path = CACHE / 'photos' / f'{record["photo_id"]}{suffix}'
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        payload = get(record['download_url'])
        path.write_bytes(payload)
    return {**record, **fingerprint(path), 'cache_path': str(path)}


def duplicate(a, b):
    if a['sha256'] == b['sha256'] or a['pixel_sha256'] == b['pixel_sha256']:
        return 'exact_duplicate'
    ph = (int(a['phash63'], 16) ^ int(b['phash63'], 16)).bit_count()
    dh = (int(a['dhash64'], 16) ^ int(b['dhash64'], 16)).bit_count()
    if ph <= 6 and dh <= 8:
        return f'near_duplicate_phash={ph}_dhash={dh}'
    return None


def collect():
    specs = sampling_groups()
    dump(ROOT / 'dataset_sampling.json', {'seed': SEED, 'sampling_groups': specs,
         'max_per_observer': 6, 'max_per_species': 6,
         'exclude_explicitly_dead': {'groups': ['butterflies', 'moths', 'other_insects'],
                                    'controlled_attribute_id': 17, 'controlled_value_id': 19},
         'min_short_side': 256, 'min_long_side': 512})
    rejects_file = ROOT / 'dataset_exclusions.json'
    exclusions = json.loads(rejects_file.read_text()) if rejects_file.exists() else {}
    selected, events = [], []
    authors, species = collections.Counter(), collections.Counter()
    obsids, photoids, encounters = set(), set(), set()
    for spec in specs:
        count = 0
        for page in range(1, 16):
            pool = candidates(spec, page)
            if not pool:
                break
            offset = 0
            while count < spec['quota'] and offset < len(pool):
                batch = []
                while len(batch) < min(16, spec['quota'] - count) and offset < len(pool):
                    rec = pool[offset]; offset += 1
                    encounter = (rec['observer_login'], rec['observed_on'], rec['species_id'])
                    if str(rec['photo_id']) in exclusions:
                        events.append({'photo_id': rec['photo_id'], 'reason': exclusions[str(rec['photo_id'])]})
                        continue
                    if (rec['observation_id'] in obsids or rec['photo_id'] in photoids
                        or authors[rec['observer_login']] >= 6 or species[rec['species_id']] >= 6
                        or encounter in encounters):
                        continue
                    batch.append(rec)
                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                    futures = [executor.submit(download, rec) for rec in batch]
                    for rec, future in zip(batch, futures):
                        try:
                            item = future.result()
                        except Exception as exc:
                            events.append({'photo_id': rec['photo_id'], 'reason': f'download_or_decode: {exc}'})
                            continue
                        encounter = (item['observer_login'], item['observed_on'], item['species_id'])
                        if (authors[item['observer_login']] >= 6 or species[item['species_id']] >= 6
                                or encounter in encounters):
                            continue
                        match = next(((other, duplicate(item, other)) for other in selected
                                      if duplicate(item, other)), None)
                        if match:
                            events.append({'photo_id': item['photo_id'], 'reason': match[1],
                                           'kept_photo_id': match[0]['photo_id']})
                            continue
                        selected.append(item)
                        authors[item['observer_login']] += 1
                        species[item['species_id']] += 1
                        encounters.add(encounter)
                        obsids.add(item['observation_id']); photoids.add(item['photo_id'])
                        count += 1
                print(f'Selected {spec["key"]}: {count}/{spec["quota"]}; total={len(selected)}', flush=True)
            if count >= spec['quota']:
                break
        if count != spec['quota']:
            dump(CACHE / 'partial.json', selected)
            raise RuntimeError(f'Insufficient candidates for {spec["key"]}: {count}')
        dump(CACHE / 'progress.json', selected)
    assert len(selected) == 400
    dest = ROOT / 'images'
    dest.mkdir(exist_ok=True)
    for index, item in enumerate(selected, 1):
        source = Path(item.pop('cache_path'))
        item['image_id'] = f'inat_{item["photo_id"]}'
        item['file'] = f'images/{item["image_id"]}{source.suffix}'
        shutil.copyfile(source, ROOT / item['file'])
    # Only remove obsolete files produced by this script, never unrelated files.
    keep = {Path(x['file']).name for x in selected}
    for path in dest.glob('inat_*'):
        if path.name not in keep and path.is_file():
            path.unlink()
    dump(MANIFEST, selected)
    dump(ROOT / 'dataset_collection_events.json', events)
    print(f'COMPLETE: {len(selected)} files, {sum(x["bytes"] for x in selected)/1e6:.1f} MB', flush=True)


def verify():
    records = json.loads(MANIFEST.read_text())
    errors, near_pairs = [], []
    assert len(records) == 400
    assert len({r['photo_id'] for r in records}) == 400
    assert len({r['observation_id'] for r in records}) == 400
    sampling = json.loads((ROOT / 'dataset_sampling.json').read_text())
    actual_sampling_groups = collections.Counter(r['sampling_group'] for r in records)
    assert actual_sampling_groups == {s['key']: s['quota'] for s in sampling['sampling_groups']}
    assert max(collections.Counter(r['observer_login'] for r in records).values()) <= 6
    assert max(collections.Counter(r['species_id'] for r in records).values()) <= 6
    assert len({(r['observer_login'], r['observed_on'], r['species_id']) for r in records}) == 400
    excluded_path = ROOT / 'dataset_exclusions.json'
    if excluded_path.exists():
        excluded = json.loads(excluded_path.read_text())
        assert not any(str(r['photo_id']) in excluded for r in records), 'Excluded image still selected'
    actual_files = {p.name for p in (ROOT/'images').iterdir() if p.is_file()}
    assert actual_files == {Path(r['file']).name for r in records}
    for rec in records:
        assert rec['photo_license'] == rec['api_photo_record']['license_code'] == 'cc0'
        assert rec['quality_grade'] == 'research'
        assert '2020-01-01' <= rec['observed_on'] <= '2025-12-31'
        if rec['group'] in {'butterflies', 'moths', 'other_insects'}:
            assert not any(a.get('controlled_value_id') == 19 and a.get('vote_score', 0) >= 0
                           for a in rec['alive_or_dead_annotations'])
        fp = fingerprint(ROOT / rec['file'])
        if any(fp[k] != rec[k] for k in fp):
            errors.append(rec['file'])
    for i, a in enumerate(records):
        for b in records[i+1:]:
            reason = duplicate(a, b)
            if reason:
                near_pairs.append([a['photo_id'], b['photo_id'], reason])
    report = {'count': len(records), 'all_cc0': True, 'fully_decoded': len(records),
              'mismatches': errors, 'duplicate_pairs': near_pairs,
              'unique_observations': len({r['observation_id'] for r in records}),
              'unique_observers': len({r['observer_login'] for r in records}),
              'unique_species': len({r['species_id'] for r in records}),
              'groups': dict(collections.Counter(r['group'] for r in records)),
              'sampling_groups': dict(collections.Counter(r['sampling_group'] for r in records)),
              'bytes': sum(r['bytes'] for r in records),
              'max_per_observer': max(collections.Counter(r['observer_login'] for r in records).values()),
              'max_per_species': max(collections.Counter(r['species_id'] for r in records).values()),
              'exif_orientations': dict(collections.Counter(str(r['exif_orientation']) for r in records)),
              'manifest_sha256': hashlib.sha256(MANIFEST.read_bytes()).hexdigest()}
    dump(ROOT / 'dataset_validation.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    assert not errors and not near_pairs


def restore():
    records = json.loads(MANIFEST.read_text())
    for index, rec in enumerate(records, 1):
        path = ROOT / rec['file']
        if path.exists():
            assert hashlib.sha256(path.read_bytes()).hexdigest() == rec['sha256'], path
            continue
        assert rec['photo_license'] == 'cc0'
        data = get(rec['download_url'])
        assert hashlib.sha256(data).hexdigest() == rec['sha256'], f'Source changed: {rec["photo_id"]}'
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(data)
        print(f'Restored {index}/400: {path.name}', flush=True)
    verify()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['collect','verify','restore'])
    command = parser.parse_args().command
    globals()[command]()
