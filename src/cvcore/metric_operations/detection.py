import copy
import json
import operator
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tabulate import tabulate

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable


class ImageEvaluator:

    def __init__(self, gt_path_or_data, pred_path_or_data, threshold=0.5, metric_type="iou"):
        self.threshold = threshold
        self.metric_type = metric_type.lower().strip()

        if isinstance(gt_path_or_data, (str, Path)):
            self.gt_path = Path(gt_path_or_data)
            self.gt_data = self._load_json(self.gt_path)
        else:
            self.gt_path = None
            self.gt_data = gt_path_or_data

        if isinstance(pred_path_or_data, (str, Path)):
            self.pred_path = Path(pred_path_or_data)
            self.pred_data = self._load_json(self.pred_path)
        else:
            self.pred_path = None
            self.pred_data = pred_path_or_data

        gt_id = self.gt_data.get("image_id") or (self.gt_path.stem if self.gt_path else None)
        pred_id = self.pred_data.get("image_id") or (self.pred_path.stem if self.pred_path else None)

        if gt_id and pred_id and gt_id != pred_id:
            raise ValueError(f"Mismatched image pair! GT is '{gt_id}' but Pred is '{pred_id}'.")

        self.label_mapping = self._build_label_mapping()
        self.results = None

    @staticmethod
    def _load_json(path):
        with open(path, "r") as file:
            return json.load(file)

    def _build_label_mapping(self):
        mapping = {}
        for annotation in self.gt_data.get("annotations", []):
            if "class_id" in annotation and "label" in annotation:
                mapping[annotation["class_id"]] = annotation["label"]

        for pred in self.pred_data.get("predictions", []):
            if "class_id" in pred and "label" in pred:
                mapping.setdefault(pred["class_id"], pred["label"])
        return mapping

    @staticmethod
    def calculate_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
        area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def calculate_ios(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
        area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

        min_area = min(area1, area2)
        return intersection / min_area if min_area > 0 else 0.0

    def match_predictions(self, confidence_threshold, threshold=None, metric_type=None):
        eff_threshold = threshold if threshold is not None else self.threshold
        eff_metric = metric_type.lower().strip() if metric_type is not None else self.metric_type
        eval_fn = self.calculate_ios if eff_metric == "ios" else self.calculate_iou

        predictions = [
            (index, prediction)
            for index, prediction in enumerate(self.pred_data.get("predictions", []))
            if prediction.get("confidence", 0.0) >= confidence_threshold
        ]

        predictions.sort(
            key=lambda item: item[1].get("confidence", 0.0),
            reverse=True
        )

        ground_truths = self.gt_data.get("annotations", [])
        matched_gt_indices = set()
        matched_scores = []
        matches = []
        tp = fp = 0

        for pred_index, prediction in predictions:
            best_score = 0.0
            best_gt_index = None

            for gt_index, ground_truth in enumerate(ground_truths):
                if gt_index in matched_gt_indices:
                    continue

                if prediction["class_id"] != ground_truth["class_id"]:
                    continue

                score = eval_fn(prediction["bbox"], ground_truth["bbox"])

                if score > best_score:
                    best_score = score
                    best_gt_index = gt_index

            if best_gt_index is not None and best_score >= eff_threshold:
                matched_gt_indices.add(best_gt_index)
                matched_scores.append(best_score)
                tp += 1
                status = "TP"
            else:
                fp += 1
                status = "FP"

            matches.append({
                "prediction_index": pred_index,
                "gt_index": best_gt_index if status == "TP" else None,
                "confidence": prediction.get("confidence", 0.0),
                "score": best_score,
                "status": status,
                "class_id": prediction["class_id"]
            })

        fn = len(ground_truths) - len(matched_gt_indices)

        return {
            "gt_count": len(ground_truths),
            "prediction_count": len(predictions),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "matched_scores": matched_scores,
            "matches": matches,
            "metric_type": eff_metric
        }

    def match_class_predictions(self, class_id, confidence_threshold, threshold=None, metric_type=None):
        eff_threshold = threshold if threshold is not None else self.threshold
        eff_metric = metric_type.lower().strip() if metric_type is not None else self.metric_type
        eval_fn = self.calculate_ios if eff_metric == "ios" else self.calculate_iou

        ground_truths = [
            (index, ground_truth)
            for index, ground_truth in enumerate(self.gt_data.get("annotations", []))
            if ground_truth["class_id"] == class_id
        ]

        predictions = [
            (index, prediction)
            for index, prediction in enumerate(self.pred_data.get("predictions", []))
            if prediction["class_id"] == class_id
            and prediction.get("confidence", 0.0) >= confidence_threshold
        ]

        predictions.sort(
            key=lambda item: item[1].get("confidence", 0.0),
            reverse=True
        )

        matched_gt_indices = set()
        matched_scores = []
        tp = fp = 0

        for _, prediction in predictions:
            best_score = 0.0
            best_gt_index = None

            for gt_index, ground_truth in ground_truths:
                if gt_index in matched_gt_indices:
                    continue

                score = eval_fn(prediction["bbox"], ground_truth["bbox"])

                if score > best_score:
                    best_score = score
                    best_gt_index = gt_index

            if best_gt_index is not None and best_score >= eff_threshold:
                matched_gt_indices.add(best_gt_index)
                matched_scores.append(best_score)
                tp += 1
            else:
                fp += 1

        return {
            "class_id": class_id,
            "class_name": self.label_mapping.get(class_id, str(class_id)),
            "gt_count": len(ground_truths),
            "prediction_count": len(predictions),
            "tp": tp,
            "fp": fp,
            "fn": len(ground_truths) - len(matched_gt_indices),
            "matched_scores": matched_scores,
            "metric_type": eff_metric
        }

    @staticmethod
    def calculate_metrics(match_result):
        tp = match_result["tp"]
        fp = match_result["fp"]
        fn = match_result["fn"]

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1_score = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0 else 0.0
        )

        matched_scores = match_result.get("matched_scores", [])
        mean_score = (
            sum(matched_scores) / len(matched_scores)
            if matched_scores else 0.0
        )

        return {
            "gt_count": match_result["gt_count"],
            "prediction_count": match_result["prediction_count"],
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "mean_score": mean_score,
            "metric_type": match_result.get("metric_type", "iou").upper()
        }

    def get_per_class_stat(self, confidence_threshold, threshold=None, metric_type=None):
        filtered_predictions = [
            prediction
            for prediction in self.pred_data.get("predictions", [])
            if prediction.get("confidence", 0.0) >= confidence_threshold
        ]

        class_ids = {
            ground_truth["class_id"]
            for ground_truth in self.gt_data.get("annotations", [])
        }

        class_ids.update(
            prediction["class_id"]
            for prediction in filtered_predictions
        )

        results = []
        for class_id in sorted(class_ids):
            result = self.match_class_predictions(
                class_id,
                confidence_threshold,
                threshold=threshold,
                metric_type=metric_type
            )
            metrics = self.calculate_metrics(result)
            metrics["class_id"] = class_id
            metrics["class_name"] = result["class_name"]
            metrics["matched_scores"] = result["matched_scores"]
            results.append(metrics)

        return results

    def get_stat(self, threshold=0.5, metric_type=None, confidence_thresholds=None, return_stat=True):
        if confidence_thresholds is None:
            confidence_thresholds = [0.25, 0.5, 0.75]

        self.threshold = threshold
        if metric_type is not None:
            self.metric_type = metric_type.lower().strip()

        metric_name = self.metric_type.upper()
        overall_results = []
        per_class_results = []

        for confidence_threshold in confidence_thresholds:
            overall_match = self.match_predictions(
                confidence_threshold,
                threshold=self.threshold,
                metric_type=self.metric_type
            )
            overall_metrics = self.calculate_metrics(overall_match)
            overall_metrics["confidence_threshold"] = confidence_threshold
            overall_metrics["matched_scores"] = overall_match["matched_scores"]
            overall_results.append(overall_metrics)

            class_results = self.get_per_class_stat(
                confidence_threshold,
                threshold=self.threshold,
                metric_type=self.metric_type
            )
            for result in class_results:
                result["confidence_threshold"] = confidence_threshold
                per_class_results.append(result)

        self.results = {
            "overall": overall_results,
            "per_class": per_class_results,
            "metric_type": self.metric_type,
            "threshold": self.threshold
        }

        if return_stat:
            headers = [
                "Confidence", "GT", "Predictions", "TP", "FP", "FN",
                "Precision", "Recall", "F1", f"Mean {metric_name}"
            ]

            overall_table = [
                [
                    result["confidence_threshold"],
                    result["gt_count"],
                    result["prediction_count"],
                    result["tp"],
                    result["fp"],
                    result["fn"],
                    round(result["precision"], 4),
                    round(result["recall"], 4),
                    round(result["f1_score"], 4),
                    round(result["mean_score"], 4)
                ]
                for result in overall_results
            ]

            print(f"\nSINGLE-IMAGE OVERALL DETECTION METRICS ({metric_name}@{self.threshold})\n")
            print(tabulate(overall_table, headers=headers, tablefmt="grid"))

            per_class_table = [
                [
                    result["confidence_threshold"],
                    result["class_name"],
                    result["gt_count"],
                    result["prediction_count"],
                    result["tp"],
                    result["fp"],
                    result["fn"],
                    round(result["precision"], 4),
                    round(result["recall"], 4),
                    round(result["f1_score"], 4),
                    round(result["mean_score"], 4)
                ]
                for result in per_class_results
            ]

            print(f"\nSINGLE-IMAGE PER-CLASS DETECTION METRICS ({metric_name}@{self.threshold})\n")
            print(tabulate(
                per_class_table,
                headers=["Confidence", "Class Name", *headers[1:]],
                tablefmt="grid"
            ))

        return None

    def plot(
    self,
    confidence_threshold=0.25,
    title_prefix="",
    show=True,
    show_stats=False,
    threshold=None,
    metric_type=None,
    class_names=None  # <-- Added: filter visualizer to specific classes
):
        eff_threshold = threshold if threshold is not None else self.threshold
        eff_metric = metric_type.lower().strip() if metric_type is not None else self.metric_type

        image_path = self.gt_data.get("image_path")
        if image_path and Path(image_path).exists():
            image = Image.open(image_path).convert("RGB")
        else:
            image = Image.new("RGB", (640, 640), color=(245, 245, 245))

        # Resolve target_cids if class_names is provided
        target_cids = None
        if class_names is not None:
            if not isinstance(class_names, (list, tuple, set)):
                class_names = [class_names]
            name_to_cid = {str(v).lower(): k for k, v in self.label_mapping.items()}
            target_cids = set()
            for c in class_names:
                if isinstance(c, int):
                    target_cids.add(c)
                elif str(c).isdigit():
                    target_cids.add(int(c))
                elif str(c).lower() in name_to_cid:
                    target_cids.add(name_to_cid[str(c).lower()])

        # Filter GT annotations and predictions by target_cids
        all_annotations = self.gt_data.get("annotations", [])
        annotations = [
            ann for ann in all_annotations
            if target_cids is None or ann.get("class_id") in target_cids
        ]

        all_predictions = self.pred_data.get("predictions", [])
        predictions = [
            p for p in all_predictions
            if p.get("confidence", 0.0) >= confidence_threshold
            and (target_cids is None or p.get("class_id") in target_cids)
        ]

        # Calculate metrics: class-specific if a single target class is chosen, else overall
        if target_cids is not None and len(target_cids) == 1:
            target_cid = list(target_cids)[0]
            matched_res = self.match_class_predictions(
                target_cid,
                confidence_threshold,
                threshold=eff_threshold,
                metric_type=eff_metric
            )
        else:
            matched_res = self.match_predictions(
                confidence_threshold,
                threshold=eff_threshold,
                metric_type=eff_metric
            )
        img_metrics = self.calculate_metrics(matched_res)

        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        axes[0].imshow(image)
        axes[0].set_title(f"{title_prefix}Ground Truth".strip())
        axes[0].axis("off")

        for annotation in annotations:
            x1, y1, x2, y2 = annotation["bbox"]
            class_id = annotation["class_id"]
            class_name = annotation.get("label", str(class_id))
            color = colors[class_id % len(colors)]

            rectangle = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False, linewidth=2, edgecolor=color
            )
            axes[0].add_patch(rectangle)
            axes[0].text(
                x1, max(0, y1 - 5), class_name,
                fontsize=10, color="white",
                bbox=dict(facecolor=color, alpha=0.7, pad=2)
            )

        axes[1].imshow(image)
        axes[1].set_title(f"{title_prefix}Predictions (Confidence >= {confidence_threshold})".strip())
        axes[1].axis("off")

        for prediction in predictions:
            x1, y1, x2, y2 = prediction["bbox"]
            class_id = prediction["class_id"]
            class_name = prediction.get("label", str(class_id))
            confidence = prediction.get("confidence", 0.0)
            color = colors[class_id % len(colors)]

            rectangle = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False, linewidth=2, edgecolor=color
            )
            axes[1].add_patch(rectangle)
            axes[1].text(
                x1, max(0, y1 - 5), f"{class_name} {confidence:.2f}",
                fontsize=10, color="white",
                bbox=dict(facecolor=color, alpha=0.7, pad=2)
            )

        if show_stats:
            m_label = eff_metric.upper()
            scope_label = f" ({class_names[0]})" if target_cids and len(target_cids) == 1 else ""
            stat_box = (
                f"[{m_label}@{eff_threshold}{scope_label}]\n"
                f"TP: {img_metrics['tp']} | FP: {img_metrics['fp']} | FN: {img_metrics['fn']}\n"
                f"Precision: {img_metrics['precision']:.3f}\n"
                f"Recall: {img_metrics['recall']:.3f}"
            )
            axes[1].text(
                0.02, 0.98, stat_box,
                transform=axes[1].transAxes,
                fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.8, edgecolor="none"),
                color="lime", weight="bold"
            )
            print(f"\n--- Metrics for image: {title_prefix} (Conf: {confidence_threshold}, {m_label}@{eff_threshold}) ---")
            print(tabulate([[
                confidence_threshold,
                img_metrics["gt_count"], img_metrics["prediction_count"],
                img_metrics["tp"], img_metrics["fp"], img_metrics["fn"],
                round(img_metrics["precision"], 4), round(img_metrics["recall"], 4)
            ]], headers=["Conf_Threshold", "GT", "Preds", "TP", "FP", "FN", "Precision", "Recall"], tablefmt="grid"))

        plt.tight_layout()
        if show:
            plt.show()
        plt.close(fig)

class DatasetEvaluator:

    OPS = {
        "<": operator.lt,
        "<=": operator.le,
        ">": operator.gt,
        ">=": operator.ge,
        "==": operator.eq,
        "=": operator.eq,
        "!=": operator.ne,
    }

    def __init__(self, gt_dir, pred_dir, threshold=0.5, metric_type="iou", num_workers: Optional[int] = None):
        self.gt_dir = Path(gt_dir)
        self.pred_dir = Path(pred_dir)
        self.threshold = threshold
        self.metric_type = metric_type.lower().strip()
        self.num_workers = num_workers

        self.image_files = self._discover_pairs()
        self.image_ids = list(self.image_files.keys())
        self.active_image_ids = list(self.image_ids)

        self.label_mapping = {}
        self.overall_df = None
        self.per_class_df = None
        self.ap_df = None
        self.map_50_95 = None
        self.map_50 = None
        self.all_image_object = {}
        self.results = None

    def _discover_pairs(self):
        pairs = {}
        for gt_path in sorted(self.gt_dir.glob("*.json")):
            image_id = gt_path.stem
            pred_path = self.pred_dir / f"{image_id}.json"
            if pred_path.exists():
                pairs[image_id] = (gt_path, pred_path)
        return pairs

    @staticmethod
    def _compute_single_metrics(gt_count, prediction_count, tp, fp, fn, matched_scores, metric_name="IOU"):
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1_score = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0 else 0.0
        )
        mean_score = (
            sum(matched_scores) / len(matched_scores)
            if matched_scores else 0.0
        )
        return {
            "gt_count": gt_count,
            "prediction_count": prediction_count,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "mean_score": mean_score,
            "metric_type": metric_name.upper()
        }

    def calculate_metrics(self, confidence_thresholds=None, image_objects=None):
        if confidence_thresholds is None:
            confidence_thresholds = [0.25, 0.5, 0.75]

        if image_objects is None:
            image_objects = self.all_image_object

        overall_results = []
        per_class_results = []

        all_class_ids = set()
        for img_obj in image_objects.values():
            all_class_ids.update(img_obj.label_mapping.keys())
        sorted_cids = sorted(all_class_ids)

        for ct in confidence_thresholds:
            ov_gt = 0
            ov_pred = 0
            ov_tp = 0
            ov_fp = 0
            ov_fn = 0
            ov_matched_scores = []

            class_accumulators = {
                cid: {
                    "gt_count": 0,
                    "prediction_count": 0,
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                    "matched_scores": []
                }
                for cid in sorted_cids
            }

            for img_obj in image_objects.values():
                res = img_obj.results
                if res is None:
                    continue

                for ov_entry in res.get("overall", []):
                    if ov_entry.get("confidence_threshold") == ct:
                        ov_gt += ov_entry["gt_count"]
                        ov_pred += ov_entry["prediction_count"]
                        ov_tp += ov_entry["tp"]
                        ov_fp += ov_entry["fp"]
                        ov_fn += ov_entry["fn"]
                        ov_matched_scores.extend(ov_entry.get("matched_scores", []))
                        break

                for cls_entry in res.get("per_class", []):
                    if cls_entry.get("confidence_threshold") == ct:
                        cid = cls_entry["class_id"]
                        if cid in class_accumulators:
                            class_accumulators[cid]["gt_count"] += cls_entry["gt_count"]
                            class_accumulators[cid]["prediction_count"] += cls_entry["prediction_count"]
                            class_accumulators[cid]["tp"] += cls_entry["tp"]
                            class_accumulators[cid]["fp"] += cls_entry["fp"]
                            class_accumulators[cid]["fn"] += cls_entry["fn"]
                            class_accumulators[cid]["matched_scores"].extend(cls_entry.get("matched_scores", []))

            ov_metrics = self._compute_single_metrics(
                ov_gt, ov_pred, ov_tp, ov_fp, ov_fn, ov_matched_scores, metric_name=self.metric_type
            )
            ov_metrics["confidence_threshold"] = ct
            overall_results.append(ov_metrics)

            for cid in sorted_cids:
                c_data = class_accumulators[cid]
                c_metrics = self._compute_single_metrics(
                    c_data["gt_count"], c_data["prediction_count"],
                    c_data["tp"], c_data["fp"], c_data["fn"], c_data["matched_scores"],
                    metric_name=self.metric_type
                )
                c_metrics["confidence_threshold"] = ct
                c_metrics["class_id"] = cid
                c_metrics["class_name"] = self.label_mapping.get(cid, str(cid))
                per_class_results.append(c_metrics)

        return {
            "overall": overall_results,
            "per_class": per_class_results
        }

    _calculate_metrics = calculate_metrics

    @staticmethod
    def _compute_ap_from_matches(predictions_info, total_gt):
        if total_gt == 0:
            return None
        if not predictions_info:
            return 0.0

        predictions_info.sort(key=lambda x: x[0], reverse=True)
        tp = np.array([x[1] for x in predictions_info])
        fp = 1 - tp

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        recall = tp_cumsum / total_gt
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)

        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))

        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])

        indices = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])
        return float(ap)

    @staticmethod
    def _process_single_image(args):
        image_id, gt_path, pred_path, thresh, metric_t, conf_threshs = args
        img_eval = ImageEvaluator(gt_path, pred_path, threshold=thresh, metric_type=metric_t)
        img_eval.get_stat(threshold=thresh, metric_type=metric_t, confidence_thresholds=conf_threshs, return_stat=False)
        return image_id, img_eval

    def get_stat(self, threshold=0.5, metric_type=None, confidence_thresholds=None, return_stat=True, num_workers=None):
        if confidence_thresholds is None:
            confidence_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9]

        self.threshold = threshold
        if metric_type is not None:
            self.metric_type = metric_type.lower().strip()

        eval_fn = ImageEvaluator.calculate_ios if self.metric_type == "ios" else ImageEvaluator.calculate_iou
        workers = num_workers if num_workers is not None else self.num_workers

        tasks = [
            (
                img_id,
                self.image_files[img_id][0],
                self.image_files[img_id][1],
                self.threshold,
                self.metric_type,
                confidence_thresholds,
            )
            for img_id in self.active_image_ids
        ]

        self.all_image_object = {}

        if workers == 1:
            for task in tqdm(tasks, desc=f"Evaluating ({self.metric_type.upper()})", unit="img"):
                image_id, img_eval = self._process_single_image(task)
                self.all_image_object[image_id] = img_eval
                self.label_mapping.update(img_eval.label_mapping)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for image_id, img_eval in tqdm(
                    executor.map(self._process_single_image, tasks),
                    total=len(tasks),
                    desc=f"Evaluating ({self.metric_type.upper()})",
                    unit="img",
                ):
                    self.all_image_object[image_id] = img_eval
                    self.label_mapping.update(img_eval.label_mapping)

        metric_results = self.calculate_metrics(confidence_thresholds=confidence_thresholds)
        overall_results = metric_results["overall"]
        per_class_results = metric_results["per_class"]

        map_thresholds = [round(float(x), 2) for x in np.arange(0.5, 1.0, 0.05)]
        ap_preds = {}
        class_gt_counts = {}
        map_ap_preds = {t: {} for t in map_thresholds}
        sorted_cids = sorted(self.label_mapping.keys())

        for img_id, img_obj in self.all_image_object.items():
            gt_data = img_obj.gt_data
            pred_data = img_obj.pred_data

            gt_by_class = {}
            for ann in gt_data.get("annotations", []):
                cid = ann["class_id"]
                gt_by_class.setdefault(cid, []).append(ann["bbox"])

            pred_by_class = {}
            for p in pred_data.get("predictions", []):
                cid = p["class_id"]
                pred_by_class.setdefault(cid, []).append(p)

            for cid, bboxes in gt_by_class.items():
                class_gt_counts[cid] = class_gt_counts.get(cid, 0) + len(bboxes)

            classes_in_image = set(gt_by_class.keys()).union(pred_by_class.keys())

            for cid in classes_in_image:
                gts = gt_by_class.get(cid, [])
                preds = pred_by_class.get(cid, [])
                preds_sorted = sorted(preds, key=lambda x: x.get("confidence", 0.0), reverse=True)

                score_matrix = np.zeros((len(preds_sorted), len(gts)), dtype=float)
                for p_idx, p in enumerate(preds_sorted):
                    for g_idx, g in enumerate(gts):
                        score_matrix[p_idx, g_idx] = eval_fn(p["bbox"], g)

                matched_gts_default = set()
                ap_preds.setdefault(cid, [])
                for p_idx, p in enumerate(preds_sorted):
                    best_score = 0.0
                    best_gt_idx = None
                    for g_idx in range(len(gts)):
                        if g_idx in matched_gts_default:
                            continue
                        score_val = score_matrix[p_idx, g_idx]
                        if score_val > best_score:
                            best_score = score_val
                            best_gt_idx = g_idx

                    is_tp = 0
                    if best_gt_idx is not None and best_score >= self.threshold:
                        matched_gts_default.add(best_gt_idx)
                        is_tp = 1
                    ap_preds[cid].append((p.get("confidence", 0.0), is_tp))

                for t in map_thresholds:
                    map_ap_preds[t].setdefault(cid, [])
                    matched_gts_t = set()
                    for p_idx, p in enumerate(preds_sorted):
                        best_score_t = 0.0
                        best_gt_idx_t = None
                        for g_idx in range(len(gts)):
                            if g_idx in matched_gts_t:
                                continue
                            score_val = score_matrix[p_idx, g_idx]
                            if score_val > best_score_t:
                                best_score_t = score_val
                                best_gt_idx_t = g_idx

                        is_tp_t = 1 if (best_gt_idx_t is not None and best_score_t >= t) else 0
                        if is_tp_t:
                            matched_gts_t.add(best_gt_idx_t)
                        map_ap_preds[t][cid].append((p.get("confidence", 0.0), is_tp_t))

        ap_results = []
        for cid in sorted_cids:
            ap_val = self._compute_ap_from_matches(ap_preds.get(cid, []), class_gt_counts.get(cid, 0))
            ap_results.append({
                "class_id": cid,
                "class_name": self.label_mapping.get(cid, str(cid)),
                "threshold": self.threshold,
                "metric_type": self.metric_type.upper(),
                "ap": ap_val,
            })

        valid_aps = [r["ap"] for r in ap_results if r["ap"] is not None]
        map_50 = float(np.mean(valid_aps)) if valid_aps else 0.0

        map_scores = []
        for t in map_thresholds:
            t_aps = []
            for cid in sorted_cids:
                ap_t = self._compute_ap_from_matches(map_ap_preds[t].get(cid, []), class_gt_counts.get(cid, 0))
                if ap_t is not None:
                    t_aps.append(ap_t)
            if t_aps:
                map_scores.append(float(np.mean(t_aps)))
        map_50_95 = float(np.mean(map_scores)) if map_scores else 0.0

        metric_name = self.metric_type.upper()
        self.results = {
            "overall": overall_results,
            "per_class": per_class_results,
            "ap": ap_results,
            "map_50": map_50,
            "map_50_95": map_50_95,
        }

        col_score = f"Mean {metric_name}"
        self.overall_df = pd.DataFrame([
            [
                r["confidence_threshold"], r["gt_count"], r["prediction_count"],
                r["tp"], r["fp"], r["fn"],
                round(r["precision"], 4), round(r["recall"], 4),
                round(r["f1_score"], 4), round(r["mean_score"], 4),
            ]
            for r in overall_results
        ], columns=["Confidence", "GT", "Predictions", "TP", "FP", "FN", "Precision", "Recall", "F1", col_score])

        self.per_class_df = pd.DataFrame([
            [
                r["confidence_threshold"], r["class_name"], r["gt_count"], r["prediction_count"],
                r["tp"], r["fp"], r["fn"],
                round(r["precision"], 4), round(r["recall"], 4),
                round(r["f1_score"], 4), round(r["mean_score"], 4),
            ]
            for r in per_class_results
        ], columns=["Confidence", "Class Name", "GT", "Predictions", "TP", "FP", "FN", "Precision", "Recall", "F1", col_score])

        self.ap_df = pd.DataFrame([
            [r["class_name"], r["metric_type"], r["threshold"], round(r["ap"], 4) if r["ap"] is not None else None]
            for r in ap_results
        ], columns=["Class Name", "Metric", "Threshold", "AP"])

        if return_stat:
            print(f"\nDATASET-LEVEL OVERALL METRICS ({metric_name}@{self.threshold})\n")
            print(tabulate(self.overall_df.values, headers=self.overall_df.columns, tablefmt="grid"))
            print(f"\nDATASET-LEVEL PER-CLASS METRICS ({metric_name}@{self.threshold})\n")
            print(tabulate(self.per_class_df.values, headers=self.per_class_df.columns, tablefmt="grid"))
            print(f"\nDATASET-LEVEL AP ({metric_name})\n")
            print(tabulate(self.ap_df.values, headers=self.ap_df.columns, tablefmt="grid"))
            print(f"\nDATASET-LEVEL mAP ({metric_name})\n")
            print(tabulate([
                [f"mAP@{self.threshold}", round(map_50, 4)],
                ["mAP@0.5:0.95", round(map_50_95, 4)]
            ], headers=["Metric", "Score"], tablefmt="grid"))

        return None

    def filter(self, class_names, model_confidence_threshold, metric, eq, class_confidence_threshold):
        if eq not in self.OPS:
            raise ValueError(f"Unsupported operator '{eq}'. Supported: {list(self.OPS.keys())}")
        op_func = self.OPS[eq]

        if not self.all_image_object:
            self.get_stat(return_stat=False)

        target_class_ids = set()
        if not isinstance(class_names, (list, tuple, set)):
            class_names = [class_names]

        name_to_cid = {v.lower(): k for k, v in self.label_mapping.items()}
        for c in class_names:
            if isinstance(c, int):
                target_class_ids.add(c)
            elif str(c).isdigit():
                target_class_ids.add(int(c))
            elif isinstance(c, str):
                c_lower = c.strip().lower()
                if c_lower in name_to_cid:
                    target_class_ids.add(name_to_cid[c_lower])

        metric_key = metric.lower().strip()
        if metric_key == "f1":
            metric_key = "f1_score"
        elif metric_key in ("iou", "ios", "overlap"):
            metric_key = "mean_score"

        filtered_objects = {}
        for image_id, img_obj in self.all_image_object.items():
            matched = False
            for cid in target_class_ids:
                class_res = img_obj.match_class_predictions(
                    cid,
                    model_confidence_threshold,
                    threshold=self.threshold,
                    metric_type=self.metric_type
                )
                if class_res["gt_count"] == 0 and class_res["prediction_count"] == 0:
                    continue

                class_metrics = img_obj.calculate_metrics(class_res)
                val = class_metrics.get(metric_key, None)
                if val is not None and op_func(val, class_confidence_threshold):
                    matched = True
                    break

            if matched:
                filtered_objects[image_id] = img_obj

        print(f"Filter matched {len(filtered_objects)} / {len(self.all_image_object)} images.")

        new_evaluator = copy.copy(self)
        new_evaluator.all_image_object = filtered_objects
        new_evaluator.active_image_ids = list(filtered_objects.keys())
        new_evaluator.overall_df = None
        new_evaluator.per_class_df = None
        new_evaluator.ap_df = None
        return new_evaluator
def plot(
    self,
    num_samples=5,
    confidence_threshold=None,  # Defaults dynamically to self.threshold
    random_sample=True,
    class_names=None,
    model_confidence_threshold=None,
    show=True,
    show_stats=False,
    threshold=None,
    metric_type=None
):
    if not self.all_image_object:
        self.get_stat(return_stat=False)

    if not self.all_image_object:
        print("No image objects available to plot.")
        return

    # Choose plotting confidence threshold
    if model_confidence_threshold is not None:
        plot_conf = model_confidence_threshold
    elif confidence_threshold is not None:
        plot_conf = confidence_threshold
    else:
        plot_conf = 0.2  # Match typical lower evaluation bound

    eff_threshold = threshold if threshold is not None else self.threshold
    eff_metric = metric_type if metric_type is not None else self.metric_type

    candidate_items = list(self.all_image_object.items())

    if class_names is not None:
        if not isinstance(class_names, (list, tuple, set)):
            class_names_list = [class_names]
        else:
            class_names_list = list(class_names)

        name_to_cid = {v.lower(): k for k, v in self.label_mapping.items()}
        target_class_ids = set()
        for c in class_names_list:
            if isinstance(c, int):
                target_class_ids.add(c)
            elif str(c).isdigit():
                target_class_ids.add(int(c))
            elif isinstance(c, str):
                c_lower = c.strip().lower()
                if c_lower in name_to_cid:
                    target_class_ids.add(name_to_cid[c_lower])

        filtered_candidates = []
        for image_id, img_obj in candidate_items:
            has_gt = any(
                ann.get("class_id") in target_class_ids
                for ann in img_obj.gt_data.get("annotations", [])
            )
            has_pred = any(
                p.get("class_id") in target_class_ids
                and p.get("confidence", 0.0) >= plot_conf
                for p in img_obj.pred_data.get("predictions", [])
            )
            if has_gt or has_pred:
                filtered_candidates.append((image_id, img_obj))

        candidate_items = filtered_candidates
        if not candidate_items:
            print(f"No images found containing class '{class_names}' at confidence >= {plot_conf}")
            return

    num_samples = min(num_samples, len(candidate_items))
    if random_sample:
        sampled_items = random.sample(candidate_items, num_samples)
    else:
        sampled_items = candidate_items[:num_samples]

    title_suffix = f" ({class_names})" if class_names is not None else ""
    for image_id, img_obj in sampled_items:
        img_obj.plot(
            confidence_threshold=plot_conf,
            title_prefix=f"[{image_id}]{title_suffix} ",
            show=show,
            show_stats=show_stats,
            threshold=eff_threshold,
            metric_type=eff_metric,
            class_names=class_names  # <-- Forwarded class_names to ImageEvaluator
        )