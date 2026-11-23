"""Evaluate normalized butterfly boxes, assuming all input data is valid."""

import argparse
import hashlib
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass
class ImageResult:
    boxes: list
    elapsed_seconds: float | None = None


def parse_box(box):
    """Convert left/top/width/height to left/top/right/bottom."""
    x, y = box["left"], box["top"]
    return (x, y, x + box["width"], y + box["height"])


def load_jsonl(path):
    records = {}
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line in stream:
            row = json.loads(line)
            records[row["photo_id"]] = ImageResult(
                [parse_box(box) for box in row["boxes"]], row.get("elapsed_seconds"))
    return records


def load_groups(path):
    rows = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return {row["photo_id"]: row["group"] for row in rows}


def iou(a, b):
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area_a + area_b - intersection)


def maximum_matching(scores, threshold):
    """Maximum cardinality matching via augmenting paths; rows=predictions, columns=gold.

    This maximizes the number of valid pairs, not their total IoU. Iterative BFS
    avoids recursion limits. Input order deterministically breaks ties.
    """
    edges = [[g for g, score in enumerate(row) if score >= threshold] for row in scores]
    pred_to_gold, gold_to_pred = {}, {}
    for start in range(len(edges)):
        queue, visited_pred, parent = deque([start]), {start}, {}
        free_gold = None
        while queue and free_gold is None:
            pred = queue.popleft()
            for gold in edges[pred]:
                if gold in parent:
                    continue
                parent[gold] = pred
                owner = gold_to_pred.get(gold)
                if owner is None:
                    free_gold = gold
                    break
                if owner not in visited_pred:
                    visited_pred.add(owner)
                    queue.append(owner)
        while free_gold is not None:
            pred = parent[free_gold]
            previous = pred_to_gold.get(pred)
            pred_to_gold[pred] = free_gold
            gold_to_pred[free_gold] = pred
            free_gold = previous
    return [{"prediction_index": p, "golden_index": g, "iou": scores[p][g]}
            for p, g in sorted(pred_to_gold.items())]


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def summarize(rows, key):
    tp, fp, fn = (sum(row["metrics"][key][name] for row in rows) for name in ("tp", "fp", "fn"))
    negatives = [row for row in rows if row["golden_boxes"] == 0]
    false_positive_images = sum(row["predicted_boxes"] > 0 for row in negatives)
    return {
        "images": len(rows), "golden_boxes": tp + fn, "predicted_boxes": tp + fp,
        "tp": tp, "fp": fp, "fn": fn, "object_errors": fp + fn,
        "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "images_with_errors": sum(row["metrics"][key]["fp"] + row["metrics"][key]["fn"] > 0 for row in rows),
        "negative_images": len(negatives),
        "false_positive_boxes_on_negatives": sum(row["predicted_boxes"] for row in negatives),
        "false_positive_images_on_negatives": false_positive_images,
        "false_positive_image_rate_on_negatives": ratio(false_positive_images, len(negatives)),
    }


def evaluate(golden, predictions, thresholds=(0.5, 0.75), groups=None):
    thresholds = tuple(thresholds)
    details = []
    for photo_id in sorted(golden):
        truth, predicted = golden[photo_id].boxes, predictions[photo_id].boxes
        scores = [[iou(p, g) for g in truth] for p in predicted]
        row = {"photo_id": photo_id, "golden_boxes": len(truth),
               "predicted_boxes": len(predicted), "metrics": {}}
        if groups is not None:
            row["group"] = groups[photo_id]
        for threshold in thresholds:
            matches = maximum_matching(scores, threshold)
            matched_gold = {m["golden_index"] for m in matches}
            matched_pred = {m["prediction_index"] for m in matches}
            row["metrics"][str(float(threshold))] = {
                "tp": len(matches), "fp": len(predicted) - len(matches), "fn": len(truth) - len(matches),
                "matches": matches,
                "unmatched_golden_indices": sorted(set(range(len(truth))) - matched_gold),
                "unmatched_prediction_indices": sorted(set(range(len(predicted))) - matched_pred),
            }
        details.append(row)
    keys = [str(float(t)) for t in thresholds]
    times = [r.elapsed_seconds for r in predictions.values() if r.elapsed_seconds is not None]
    return {
        "iou_thresholds": list(thresholds),
        "overall": {key: summarize(details, key) for key in keys},
        "by_group": {group: {key: summarize([r for r in details if r["group"] == group], key)
                              for key in keys} for group in sorted(set(groups.values()))} if groups is not None else {},
        "timing": {"source": "prediction records: elapsed_seconds", "timed_images": len(times),
                   "total_images": len(predictions), "complete": len(times) == len(predictions),
                   "sum_recorded_seconds": sum(times) if times else None,
                   "mean_recorded_seconds": ratio(sum(times), len(times))},
        "per_image": details,
    }


def show_summary(report):
    print(f"Модель / запуск: {report['name']}")
    print(f"{'Группа':<18} {'IoU':>5} {'Фото':>5} {'TP':>5} {'FP':>5} {'FN':>5} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    sections = [("ALL", report["overall"]), *report["by_group"].items()]
    for group, metrics in sections:
        for threshold, values in metrics.items():
            numbers = ["—" if values[k] is None else f"{values[k]:.4f}" for k in ("precision", "recall", "f1")]
            print(f"{group:<18} {threshold:>5} {values['images']:>5} {values['tp']:>5} {values['fp']:>5} {values['fn']:>5} "
                  + " ".join(f"{n:>10}" for n in numbers))
    first = next(iter(report["overall"].values()))
    print(f"Без бабочек: {first['negative_images']} фото; лишних рамок: {first['false_positive_boxes_on_negatives']}; "
          f"фото с ложными срабатываниями: {first['false_positive_images_on_negatives']}.")
    timing = report["timing"]
    if timing["timed_images"]:
        print(f"Время записано для {timing['timed_images']}/{timing['total_images']} фото: "
              f"сумма {timing['sum_recorded_seconds']:.3f} с, среднее {timing['mean_recorded_seconds']:.3f} с.")
    else:
        print("Время модели не указано: для его учёта добавьте elapsed_seconds в предсказания.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Сравнение рамок модели с golden set (object F1, precision, recall).")
    parser.add_argument("--golden", type=Path, default=ROOT / "golden_set.jsonl", help="Эталон JSONL; по умолчанию рядом со скриптом")
    parser.add_argument("--predictions", type=Path, required=True, help="Предсказания одной модели в JSONL")
    parser.add_argument("--manifest", type=Path, help="images_manifest.json для результатов по группам")
    parser.add_argument("--output", type=Path, help="Сохранить подробный JSON-отчёт; существующий отчёт будет заменён")
    parser.add_argument("--name", help="Название модели / конфигурации; по умолчанию имя файла предсказаний")
    parser.add_argument("--iou-thresholds", type=float, nargs="+", default=[0.5, 0.75], help="Пороги IoU; по умолчанию 0.5 0.75")
    args = parser.parse_args(argv)
    inputs = {"golden": args.golden, "predictions": args.predictions}
    if args.manifest:
        inputs["manifest"] = args.manifest
    golden, predictions = load_jsonl(args.golden), load_jsonl(args.predictions)
    groups = load_groups(args.manifest) if args.manifest else None
    report = {"schema_version": 1, "name": args.name or args.predictions.stem,
              "inputs": {name: {"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                         for name, p in inputs.items()},
              **evaluate(golden, predictions, args.iou_thresholds, groups)}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    show_summary(report)
    if args.output:
        print(f"Отчёт: {args.output}")


if __name__ == "__main__":
    main()
