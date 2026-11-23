"""Run one of four comparison models locally and save normalized boxes."""

import json
import math
import os
import time
from datetime import datetime, timezone

from download_contest import download_images
from evaluate import iou, parse_box
from prediction_run import (
    ROOT, finish_run, parse_args, read_records, read_results, save_results,
    start_run, successful, write_submission,
)


MODEL_IDS = {
    "dino": "IDEA-Research/grounding-dino-tiny",
    "owlv2": "google/owlv2-base-patch16-ensemble",
    "qwen": "Qwen/Qwen3-VL-2B-Instruct",
    "yolo-world": "yolov8s-worldv2.pt",
}
QWEN_PROMPT = """Find every adult butterfly and adult moth in this image, including small moths.
Use one tight, axis-aligned bounding box per animal, enclosing its visible wings,
body, antennae and legs. Include only visible parts of occluded or cropped animals.
Do not mark caterpillars, pupae, other insects, plants or shadows.
Return ONLY a JSON array of objects, each with a bbox_2d field:
[{"bbox_2d": [xmin, ymin, xmax, ymax]}].
Coordinates must be relative to the whole image, scaled from 0 to 1000 on both axes.
Return [] if no adult butterflies or moths are visible. Do not include explanations."""


def parse_qwen(text):
    """Decode generated JSON; malformed model output must not become []."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("Qwen must return a JSON array")
    return [item["bbox_2d"] for item in data]


def normalize_boxes(coordinates, scores, width, height, nms_iou=None):
    """Clip model boxes to the image, discard zero-area boxes, optionally apply NMS."""
    boxes, kept_scores = [], []
    for coords, score in zip(coordinates, scores):
        x1, y1, x2, y2 = (float(value) for value in coords)
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x2 < x1 or y2 < y1:
            raise ValueError(f"Invalid model rectangle: {coords}")
        x1, x2 = (min(1.0, max(0.0, x / width)) for x in (x1, x2))
        y1, y2 = (min(1.0, max(0.0, y / height)) for y in (y1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append({"left": x1, "top": y1, "width": x2 - x1, "height": y2 - y1})
        kept_scores.append(score)
    if nms_iou is None:
        return boxes, kept_scores

    # All target objects form one class, even when separate prompts are used.
    order = sorted(range(len(boxes)), key=lambda i: kept_scores[i] or 0, reverse=True)
    keep = []
    for candidate in order:
        if all(iou(parse_box(boxes[candidate]), parse_box(boxes[k])) <= nms_iou for k in keep):
            keep.append(candidate)
    return [boxes[i] for i in keep], [kept_scores[i] for i in keep]


def synchronize(torch, device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def load_model(args):
    # Lazy imports allow --help and postprocessing tests without ML dependencies
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import torch

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if args.model == "yolo-world":
        from ultralytics import YOLOWorld

        args.yolo_weights.parent.mkdir(parents=True, exist_ok=True)
        model = YOLOWorld(str(args.yolo_weights)).to(device)
        model.model.eval()
        model.set_classes(args.labels)
        return torch, model, None, device, None

    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Qwen3VLForConditionalGeneration

    dtype = torch.float16 if args.model == "qwen" and device != "cpu" else torch.float32
    model_id = MODEL_IDS[args.model]
    options = {"revision": args.revision, "dtype": dtype, "use_safetensors": True}
    if args.model == "dino":
        options["disable_custom_kernels"] = True
    if args.model == "qwen":
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, attn_implementation="sdpa", **options)
    else:
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, **options)
    model = model.to(device).eval()
    revision = model.config._commit_hash or args.revision
    processor = AutoProcessor.from_pretrained(model_id, revision=revision, use_fast=False)
    if args.model == "qwen":
        processor.image_processor.min_pixels = 256 * 32 * 32
        processor.image_processor.max_pixels = args.max_pixels
        processor.image_processor.size = {
            "shortest_edge": processor.image_processor.min_pixels,
            "longest_edge": args.max_pixels,
        }
    return torch, model, processor, device, revision


def yolo_options(args, device):
    return {
        "device": device, "conf": args.threshold, "imgsz": args.imgsz,
        "agnostic_nms": True, "iou": args.yolo_nms_iou, "max_det": 300,
        "rect": False, "quantize": 32, "augment": False, "save": False, "verbose": False,
    }


def qwen_generation_options(args):
    return {"do_sample": False, "max_new_tokens": args.max_new_tokens,
            "repetition_penalty": args.repetition_penalty}


def infer_image(image, args, model, processor, device):
    """Return adapter output before shared postprocessing; no dataset labels enter here."""
    if args.model == "yolo-world":
        result = model.predict(source=image, **yolo_options(args, device))[0]
        # Ultralytics has already undone resizing/padding and applied its own NMS
        return {"boxes_xyxy": result.boxes.xyxy.detach().cpu().tolist(),
                "scores": result.boxes.conf.detach().cpu().tolist(),
                "coordinate_size": [image.width, image.height]}

    if args.model == "qwen":
        prompt = args.prompt or QWEN_PROMPT
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
        generated = model.generate(**inputs, **qwen_generation_options(args))
        tokens = generated[0, inputs["input_ids"].shape[1]:]
        raw_text = processor.decode(tokens, skip_special_tokens=True)
        eos_ids = model.generation_config.eos_token_id
        eos_ids = eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids]
        ended = len(tokens) > 0 and tokens[-1].item() in eos_ids
        finish_reason = "eos" if ended else "length" if len(tokens) >= args.max_new_tokens else "unknown"
        return {"raw_text": raw_text, "coordinate_size": [1000, 1000],
                "generated_tokens": len(tokens), "finish_reason": finish_reason}

    labels = args.labels
    text = ". ".join(label.lower().rstrip(".") for label in labels) + "." if args.model == "dino" else [labels]
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    outputs = model(**inputs)
    if args.model == "dino":
        result = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=args.threshold, text_threshold=args.text_threshold,
            target_sizes=[(image.height, image.width)],
        )[0]
    else:
        result = processor.post_process_object_detection(
            outputs=outputs, threshold=args.threshold, target_sizes=[(image.height, image.width)],
        )[0]
    return {"boxes_xyxy": result["boxes"].detach().cpu().tolist(),
            "scores": result["scores"].detach().cpu().tolist(),
            "coordinate_size": [image.width, image.height]}


def process_dataset(records, image_root, output, predict, sync, image_loader, nms_iou=None, raw_path=None,
                    resume=False, attempts_path=None, run_id=None):
    """Checkpoint after each image; retries preserve successes and append attempt history."""
    errors = 0
    raw_path = raw_path or output.with_suffix(".raw.jsonl")
    attempts_path = attempts_path or output.with_suffix(".attempts.jsonl")
    results = read_results(output) if resume else {}
    raw_results = read_results(raw_path) if resume else {}
    if not resume:
        save_results(output, results)
        save_results(raw_path, raw_results)
    with attempts_path.open("a" if resume else "w", encoding="utf-8") as attempts:
        for index, record in enumerate(records, 1):
            photo_id = record["photo_id"]
            previous = results.get(photo_id)
            if resume and successful(previous):
                continue
            sync()
            started = time.perf_counter()
            raw = {"photo_id": photo_id}
            try:
                image = image_loader(image_root / record["file"])
                raw.update(predict(image))
                if "raw_text" in raw:
                    if raw.get("finish_reason") == "length":
                        raise ValueError(
                            f"Qwen reached max_new_tokens ({raw['generated_tokens']}) without EOS; "
                            "inspect raw_text for repetition before increasing the limit"
                        )
                    raw["boxes_xyxy"] = parse_qwen(raw["raw_text"])
                    raw["scores"] = [None] * len(raw["boxes_xyxy"])
                boxes, scores = normalize_boxes(raw["boxes_xyxy"], raw["scores"], *raw["coordinate_size"], nms_iou)
                sync()
                result = {"photo_id": photo_id, "boxes": boxes, "scores": scores, "error": None}
            except Exception as exc:
                errors += 1
                result = {"photo_id": photo_id, "boxes": None, "error": f"{type(exc).__name__}: {exc}"}
            result["elapsed_seconds"] = time.perf_counter() - started
            raw["error"] = result["error"]
            attempt = {
                "run_id": run_id, "at_utc": datetime.now(timezone.utc).isoformat(), "photo_id": photo_id,
                "result": result, "raw": raw,
            }

            # Results from before the attempts journal need their original error archived once.
            if previous is not None and "attempts" not in previous:
                attempt.update(previous_result=previous, previous_raw=raw_results.get(photo_id))
            attempts.write(json.dumps(attempt, ensure_ascii=False) + "\n")
            attempts.flush()

            # Include failed attempts in the time needed to obtain the final result.
            results[photo_id] = {
                **result,
                "elapsed_seconds": result["elapsed_seconds"] + ((previous.get("elapsed_seconds") or 0) if previous else 0),
                "attempts": (previous.get("attempts", 1) if previous else 0) + 1,
            }
            raw_results[photo_id] = raw
            save_results(raw_path, raw_results)
            save_results(output, results)
            status = f"boxes={len(result['boxes'])}" if result["error"] is None else result["error"]
            print(f"[{index}/{len(records)}] {photo_id} {status} ({result['elapsed_seconds']:.2f}s)", flush=True)
    return errors


def main():
    args, output, previous_run, manifest_sha256 = parse_args(MODEL_IDS)
    records = read_records(args.manifest)
    expected_ids = [r["photo_id"] for r in records]
    predictions_path = output.with_suffix(".details.jsonl") if args.contest else output
    if args.retry_errors:
        previous_results = read_results(predictions_path)
        records = [r for r in records if not successful(previous_results.get(r["photo_id"]))]
        if not records:
            print("Все фотографии уже обработаны успешно; модель не загружается.")
            return finish_run(previous_run, predictions_path, output, expected_ids)
        print(f"К повторной обработке: {len(records)} из {len(expected_ids)} фотографий.", flush=True)
    if args.limit is not None:
        records = records[:args.limit]
    output.parent.mkdir(parents=True, exist_ok=True)
    download_started = time.perf_counter()
    download_images(records, args.images_dir)
    download_seconds = time.perf_counter() - download_started
    print(f"Загрузка {MODEL_IDS[args.model]}…", flush=True)
    load_started = time.perf_counter()
    torch, model, processor, device, revision = load_model(args)
    load_seconds = time.perf_counter() - load_started
    from PIL import Image, ImageOps

    def read_image(path):
        with Image.open(path) as image:
            return ImageOps.exif_transpose(image).convert("RGB")

    settings = {
        "model_id": MODEL_IDS[args.model], "resolved_revision": revision, "device": device,
        "load_seconds": load_seconds, "download_seconds": download_seconds,
        "predictions_file": str(predictions_path.resolve()), "manifest_sha256": manifest_sha256,
        "photo_ids": [record["photo_id"] for record in records],
        "prompt": (args.prompt or QWEN_PROMPT) if args.model == "qwen" else args.labels,
        "generation": qwen_generation_options(args) if args.model == "qwen" else None,
        "image_processor": processor.image_processor.to_dict() if processor is not None else None,
    }
    if args.model == "yolo-world":
        settings["yolo_predict"] = yolo_options(args, device)
    run = start_run(output, args, previous_run, model, settings)
    print(f"Устройство: {device}. Изображений: {len(records)}. Предсказания: {predictions_path}", flush=True)
    with torch.inference_mode():
        run["attempt_errors"] = process_dataset(
            records, args.manifest.parent, predictions_path,
            lambda image: infer_image(image, args, model, processor, device),
            lambda: synchronize(torch, device), read_image, args.nms_iou,
            raw_path=output.with_suffix(".raw.jsonl"),
            resume=args.retry_errors, attempts_path=output.with_suffix(".attempts.jsonl"), run_id=run["run_id"],
        )
    return finish_run(run, predictions_path, output, expected_ids)


if __name__ == "__main__":
    raise SystemExit(main())
