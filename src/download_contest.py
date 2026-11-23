"""Download and fully decode contest originals in parallel, without loading models."""

import argparse
import hashlib
import json
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent


def inspect_image(path):
    from PIL import Image

    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        image.load()
        width, height = image.size
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    return {"width": width, "height": height, "bytes": path.stat().st_size, "sha256": digest}


def image_path(record, directory):
    extension = Path(urlsplit(record["photo_url"]).path).suffix
    return directory / f"{record['photo_id']}{extension}"


def download(record, directory):
    url = record["photo_url"]
    path = image_path(record, directory)
    result = {"photo_id": record["photo_id"], "photo_url": url, "file": str(path.resolve())}
    if path.exists():
        try:
            return {**result, **inspect_image(path), "status": "cached", "error": None}
        except (OSError, ValueError):
            pass  # Replace an unreadable cache entry only after a valid download.
    for attempt in range(1, 4):
        temporary = None
        try:
            # A unique temporary name also allows predict.py to use this cache concurrently.
            with tempfile.NamedTemporaryFile(dir=directory, prefix=path.name + ".", suffix=".download", delete=False) as target:
                temporary = Path(target.name)
                request = Request(url, headers={"User-Agent": "ButterflyProject/1.0"})
                started = time.perf_counter()
                with urlopen(request, timeout=15) as response:
                    expected_size = response.headers.get("Content-Length")
                    while True:
                        if time.perf_counter() - started > 60:
                            raise TimeoutError("Original download exceeded 60 seconds")
                        chunk = response.read1(64 * 1024)
                        if not chunk:
                            break
                        target.write(chunk)
            if expected_size is not None and temporary.stat().st_size != int(expected_size):
                raise OSError("Downloaded size differs from Content-Length")
            info = inspect_image(temporary)
            temporary.replace(path)
            return {**result, **info, "status": "downloaded", "attempts": attempt, "error": None}
        except Exception as exc:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            if attempt == 3:
                return {**result, "status": "failed", "attempts": attempt, "error": f"{type(exc).__name__}: {exc}"}
            time.sleep(attempt)


def download_images(records, directory):
    """Resolve contest URLs to local files; fetch missing originals before inference."""
    for record in records:
        if "photo_url" not in record:
            continue
        path = image_path(record, directory)
        if not path.exists():
            directory.mkdir(parents=True, exist_ok=True)
            print(f"Скачивание фото {record['photo_id']}…", flush=True)
            result = download(record, directory)
            if result["error"]:
                raise RuntimeError(f"Не удалось получить фото {record['photo_id']}: {record['photo_url']}. Причина: {result['error']}")
        record["file"] = str(path.resolve())


def main():
    parser = argparse.ArgumentParser(description="Параллельно скачать оригиналы фотографий Контеста")
    parser.add_argument("--dataset", type=Path, default=ROOT / "contest/dataset.jsonl")
    parser.add_argument("--images-dir", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    directory = args.images_dir or args.dataset.parent / "images"
    directory.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    results = {}
    counts = {"cached": 0, "downloaded": 0, "failed": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download, record, directory) for record in records]
        for future in as_completed(futures):
            result = future.result()
            results[result["photo_id"]] = result
            counts[result["status"]] += 1
            if result["error"]:
                print(f"Ошибка {result['photo_id']}: {result['photo_url']} — {result['error']}", flush=True)
            if len(results) % 10 == 0 or len(results) == len(records):
                print(f"[{len(results)}/{len(records)}] кэш={counts['cached']}, скачано={counts['downloaded']}, "
                      f"ошибок={counts['failed']} ({time.perf_counter() - started:.1f} с)", flush=True)
    report = {
        "dataset": str(args.dataset.resolve()), "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "started_at_utc": started_at, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started, "workers": args.workers,
        "total_images": len(records), "counts": counts, "complete": counts["failed"] == 0,
        "total_bytes": sum(r.get("bytes", 0) for r in results.values()),
        "images": [results[r["photo_id"]] for r in records],
    }
    report_path = args.dataset.parent / "download_report.json"
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    print(f"Отчёт: {report_path}")
    if counts["failed"]:
        print("Недоступные исходные ссылки перечислены в отчёте; о них нужно сообщить преподавателю.")
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
