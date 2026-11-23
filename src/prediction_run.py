"""Command-line settings, prediction files and contest export for a local run."""

import argparse
import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def parse_args(model_ids):
    parser = argparse.ArgumentParser(description="Локальная разметка бабочек одной из четырёх моделей")
    parser.add_argument("--model", choices=model_ids, help="Модель; при --retry-errors берётся из предыдущего запуска")
    parser.add_argument("--retry-errors", action="store_true", help="Продолжить --output: повторить ошибки и отсутствующие ответы, сохранив успешные")
    parser.add_argument("--contest", action="store_true", help="Датасет и формат ответа Контеста; итоговый файл только для 400 успешных ответов")
    parser.add_argument("--manifest", type=Path, help="JSON manifest или JSONL с photo_id/photo_url; с --contest по умолчанию contest/dataset.jsonl")
    parser.add_argument("--images-dir", type=Path, help="Кэш скачанных оригиналов; по умолчанию images рядом с manifest")
    parser.add_argument("--output", type=Path, help="JSONL; по умолчанию predictions/<model>.jsonl или contest/submission-<model>.jsonl")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--revision", default="main", help="Ревизия весов Hugging Face, предпочтительно commit SHA")
    parser.add_argument("--labels", nargs="+", default=["butterfly", "moth"], help="Текстовые запросы DINO/OWLv2/YOLO-World")
    parser.add_argument("--threshold", type=float, default=0.25, help="Порог уверенности DINO/OWLv2/YOLO-World")
    parser.add_argument("--text-threshold", type=float, default=0.25, help="Порог текста Grounding DINO")
    parser.add_argument("--nms-iou", type=float, help="Дополнительный общий NMS; по умолчанию выключен")
    parser.add_argument("--yolo-weights", type=Path, default=ROOT / "weights" / model_ids["yolo-world"],
                        help="Файл весов YOLO-World; стандартные веса скачиваются при первом запуске")
    parser.add_argument("--imgsz", type=int, default=640, help="Сторона входа YOLO-World с сохранением пропорций и полями")
    parser.add_argument("--yolo-nms-iou", type=float, default=0.5, help="Встроенный NMS YOLO-World; 1.0 отключает подавление по IoU")
    parser.add_argument("--prompt", help="Инструкция Qwen; ответ должен содержать bbox_2d в шкале 0–1000")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Лимит ответа Qwen")
    parser.add_argument("--repetition-penalty", type=float, default=1.0,
                        help="Штраф повторов Qwen: 1.0 выключен; при зацикливании можно попробовать 1.1")
    parser.add_argument("--max-pixels", type=int, default=1024 * 1024, help="Лимит площади изображения для Qwen")
    parser.add_argument("--limit", type=int, help="Первые N изображений для пробного запуска")
    args = parser.parse_args()

    if args.model is None and not (args.retry_errors and args.output):
        parser.error("Укажите --model или --retry-errors --output <предыдущий файл>")
    output = args.output or (ROOT / "contest" / f"submission-{args.model}.jsonl" if args.contest
                             else ROOT / "predictions" / f"{args.model}.jsonl")
    metadata_path = output.with_suffix(".run.json")
    previous_run = None
    if args.retry_errors:
        previous_run = json.loads(metadata_path.read_text(encoding="utf-8"))
        restored = dict(previous_run["arguments"])
        restored.update({"output": output, "limit": None, "retry_errors": True,
                         "revision": previous_run.get("resolved_revision") or restored.get("revision", "main"),
                         "device": previous_run["device"]})
        if restored["model"] == "qwen":
            restored["prompt"] = previous_run["prompt"]
        for name in ("manifest", "images_dir", "yolo_weights"):
            if restored.get(name) is not None:
                restored[name] = Path(restored[name])
        parser.set_defaults(**restored)
        # Explicit CLI options override the saved settings, and are logged as a new attempt.
        args = parser.parse_args()
        if args.model != previous_run["arguments"]["model"]:
            parser.error("При --retry-errors нельзя менять модель; для другой модели нужен отдельный запуск")
        if args.contest != previous_run["arguments"].get("contest", False):
            parser.error("При --retry-errors сохраняется исходный режим --contest")

    args.manifest = args.manifest or ROOT / ("contest/dataset.jsonl" if args.contest else "images_manifest.json")
    args.images_dir = args.images_dir or args.manifest.parent / "images"
    manifest_sha256 = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if previous_run and previous_run["manifest_sha256"] != manifest_sha256:
        parser.error("Датасет отличается от предыдущего запуска; используйте исходный manifest")
    return args, output, previous_run, manifest_sha256


def read_results(path):
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as stream:
        return {row["photo_id"]: row for row in map(json.loads, stream)}


def successful(result):
    return result is not None and result.get("error") is None and result.get("boxes") is not None


def atomic_write(path, text):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def save_results(path, results):
    atomic_write(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results.values()))


def read_records(path):
    """Read the project JSON manifest or the contest JSONL dataset/template."""
    text = path.read_text(encoding="utf-8-sig")
    records = [json.loads(line) for line in text.splitlines() if line.strip()] if path.suffix == ".jsonl" else json.loads(text)
    return [{"photo_id": r["photo_id"], **({"photo_url": r["photo_url"]} if "photo_url" in r else {"file": r["file"]})}
            for r in records]


def write_submission(predictions_path, output, expected_ids):
    """Export only allowed fields; incomplete/failed runs must not become submissions."""
    rows = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines()]
    ids = [r["photo_id"] for r in rows]
    if len(rows) != 400 or len(set(ids)) != 400 or set(ids) != set(expected_ids):
        raise ValueError("Для отправки нужны ровно все 400 ID датасета, каждый один раз")
    failed = [r["photo_id"] for r in rows if r.get("error") is not None or r["boxes"] is None]
    if failed:
        raise ValueError(f"Есть ошибки обработки фотографий: {failed}; подробности в {predictions_path}")
    payload = "".join(json.dumps({"photo_id": r["photo_id"], "boxes": r["boxes"], "error": None},
                                ensure_ascii=False, allow_nan=False) + "\n" for r in rows).encode("utf-8")
    if len(payload) > 5 * 1024 * 1024:
        raise ValueError("Размер ответа превышает 5 МиБ")
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(output)


def start_run(output, args, previous_run, model, settings):
    """Save reproducibility metadata before inference so an interrupted run can resume."""
    is_yolo = args.model == "yolo-world"
    packages = ["torch", "torchvision", "Pillow"] + (
        ["ultralytics", "clip", "numpy", "opencv-python"] if is_yolo else ["transformers", "scipy", "accelerate"]
    )
    started = datetime.now(timezone.utc).isoformat()
    run = {
        **settings,
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "dtype": str(next(model.parameters()).dtype),
        "started_at_utc": started, "run_id": started, "completed": False,
        "versions": {name: importlib.metadata.version(name) for name in packages},
    }
    if previous_run:
        run["previous_runs"] = previous_run.get("previous_runs", []) + [
            {k: v for k, v in previous_run.items() if k != "previous_runs"}
        ]
    if args.contest:
        run.update(submission_file=str(output.resolve()), submission_ready=False)
    if is_yolo:
        weights = Path(model.ckpt_path).resolve()
        with weights.open("rb") as source:
            run["weights_sha256"] = hashlib.file_digest(source, "sha256").hexdigest()
        run["weights_file"] = str(weights)
        run["clip_installation"] = json.loads(importlib.metadata.distribution("clip").read_text("direct_url.json") or "null")
    save_run(output, run)
    return run


def save_run(output, run):
    atomic_write(output.with_suffix(".run.json"), json.dumps(run, ensure_ascii=False, indent=2, default=str) + "\n")


def finish_run(run, predictions_path, output, expected_ids):
    results = read_results(predictions_path)
    errors = sum(not successful(results[i]) for i in expected_ids if i in results)
    missing = [i for i in expected_ids if i not in results]
    run.update({"completed": True, "errors": errors, "missing_photo_ids": missing,
                "finished_at_utc": datetime.now(timezone.utc).isoformat()})
    exit_code = 1 if errors or (run["arguments"].get("retry_errors") and missing) else 0
    if "submission_file" in run:
        run["submission_ready"] = False
        try:
            write_submission(predictions_path, output, expected_ids)
            run["submission_ready"] = True
            run.pop("submission_error", None)
            print(f"Файл для Контеста готов: {output}")
        except ValueError as exc:
            exit_code = 1
            run["submission_error"] = str(exc)
            print(f"Файл для Контеста не создан/не обновлён: {exc}")
    save_run(output, run)
    print(f"Готово. Осталось ошибок: {errors}, без ответа: {len(missing)}.")
    return exit_code
