# 2 classes 
# 1st class - -init(gt-path,path pred)images data - plot() ,get_Stat(), _calculate metrics
# 
import json
from ..plot_operations import image_row 
from copy import deepcopy
from tabulate import tabulate
class Image:
    def __init__(self, gt_json_path, pred_json_path,unique_classes,classification_type):
        self.gt_json = self._load_json(gt_json_path)
        self.pred_json = self._load_json(pred_json_path)

        self.image_path = self.gt_json.get("image_path", "")
        if not isinstance(self.image_path, str):
            raise TypeError("'image_path' must be a string.")

        if self.image_path.strip() == "":
            raise ValueError("'image_path' cannot be empty.")
        
        if not isinstance(unique_classes, (list, tuple, set)):
            raise TypeError(
                "'unique_classes' must be a list, tuple, or set."
            )

        if len(unique_classes) == 0:
            raise ValueError("'unique_classes' cannot be empty.")

        self.unique_classes = list(unique_classes)
        if not isinstance(classification_type, str):
            raise TypeError(
                f"'classification_type' must be a string, got {type(classification_type).__name__}."
            )

        self.classification_type = classification_type.lower()
        self.labels = self.gt_json.get("labels")
        self.predictions = self.pred_json.get("predictions")
        self.confidence_scores = self.pred_json.get("confidence_scores", [])

        if self.labels is None:
            raise ValueError("Missing required key 'labels' in ground-truth JSON.")

        if self.predictions is None:
            raise ValueError("Missing required key 'predictions' in prediction JSON.")

        self._validate_inputs(self.labels, self.predictions)
        
        metrics = self._calculate_confusion_matrix(
            self.labels,
            self.predictions
        )

        self.statistics = self._calculate_classification_metrics(
            metrics
        )
    def _validate_inputs(self, labels, predictions):
        """
        Validate the labels and predictions for a single image.

        Parameters
        ----------
        labels : list
            Ground-truth labels.

        predictions : list
            Predicted labels.

        Raises
        ------
        TypeError
            If labels or predictions are not lists.

        ValueError
            If the inputs are invalid.
        """

        # ---------------------------------------------------------
        # Type Validation
        # ---------------------------------------------------------

        if not isinstance(labels, list):
            raise TypeError(
                f"'labels' must be a list, got {type(labels).__name__}."
            )

        if not isinstance(predictions, list):
            raise TypeError(
                f"'predictions' must be a list, got {type(predictions).__name__}."
            )

        # ---------------------------------------------------------
        # Empty Validation
        # ---------------------------------------------------------

        if len(labels) == 0:
            raise ValueError("'labels' cannot be empty.")

        if len(predictions) == 0:
            raise ValueError("'predictions' cannot be empty.")

        # ---------------------------------------------------------
        # Duplicate Validation
        # ---------------------------------------------------------

        if len(labels) != len(set(labels)):
            raise ValueError(
                "Duplicate labels found in ground-truth."
            )

        if len(predictions) != len(set(predictions)):
            raise ValueError(
                "Duplicate labels found in predictions."
            )

        # ---------------------------------------------------------
        # Label Validation
        # ---------------------------------------------------------

        for label in labels:
            if label not in self.unique_classes:
                raise ValueError(
                    f"Unknown ground-truth label '{label}'. "
                    f"Expected one of {self.unique_classes}."
                )

        for prediction in predictions:
            if prediction not in self.unique_classes:
                raise ValueError(
                    f"Unknown prediction label '{prediction}'. "
                    f"Expected one of {self.unique_classes}."
                )

        # ---------------------------------------------------------
        # Classification-Type Validation
        # ---------------------------------------------------------

        if self.classification_type in ("binary", "multiclass"):

            if len(labels) != 1:
                raise ValueError(
                    f"{self.classification_type} classification "
                    "requires exactly one ground-truth label."
                )

            if len(predictions) != 1:
                raise ValueError(
                    f"{self.classification_type} classification "
                    "requires exactly one predicted label."
                )

        elif self.classification_type == "multilabel":

            if len(labels) < 1:
                raise ValueError(
                    "Multilabel classification requires at least "
                    "one ground-truth label."
                )

            if len(predictions) < 1:
                raise ValueError(
                    "Multilabel classification requires at least "
                    "one predicted label."
                )

        else:
            raise ValueError(
                "classification_type must be one of "
                "['binary', 'multiclass', 'multilabel']."
            )
    def _load_json(self, path):
        try:
            with open(path, "r") as file:
                data = json.load(file)

        except FileNotFoundError:
            raise FileNotFoundError(f"JSON file not found: {path}")

        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format: {path}")

        if not isinstance(data, dict):
            raise ValueError(
                f"Expected JSON object (dictionary), got {type(data).__name__}."
            )

        return data

    def plot(self):
        """
        Display the image along with its ground-truth labels,
        predicted labels, and confidence scores.

        Returns
        -------
        None
        """
        print(f"Image Path: {self.image_path}")
        print(f"Ground Truth: {self.labels}")

        for label, score in zip(self.predictions, self.confidence_scores):
            print(f"{label} : {score}")

        image_row(image=self.image_path)
    def get_stat(self, display_stats=True):
        """
        Calculate and display image-level and class-level statistics.
        """

        labels = self.labels
        predictions = self.predictions

        metrics = self._calculate_confusion_matrix(
            labels,
            predictions
        )

        metrics = self._calculate_classification_metrics(
            metrics
        )

        self.statistics = metrics

        if display_stats:
            self._calculate_metrics(self.statistics)

        return None

    def _calculate_metrics(self, metrics):
        """
        Display image-level and class-level statistics in tabular format.

        Parameters
        ----------
        metrics : dict
            Dictionary containing image-level and class-level metrics.

        Returns
        -------
        None
        """

        # ---------------------------------------------------------
        # Image-Level Statistics
        # ---------------------------------------------------------

        image = metrics["image"]

        image_table = [[
            image["tp"],
            image["fp"],
            image["fn"],
            image["tn"],
            round(image["precision"], 4),
            round(image["recall"], 4),
            round(image["f1"], 4),
            round(image["accuracy"], 4)
        ]]

        print("\nImage-Level Statistics\n")

        print(
            tabulate(
                image_table,
                headers=[
                    "TP",
                    "FP",
                    "FN",
                    "TN",
                    "Precision",
                    "Recall",
                    "F1",
                    "Accuracy"
                ],
                tablefmt="grid"
            )
        )

        # ---------------------------------------------------------
        # Class-Level Statistics
        # ---------------------------------------------------------

        class_table = []

        for class_name, values in metrics["classes"].items():

            class_table.append([
                class_name,
                values["tp"],
                values["fp"],
                values["fn"],
                values["tn"],
                round(values["precision"], 4),
                round(values["recall"], 4),
                round(values["f1"], 4),
                round(values["accuracy"], 4)
            ])

        print("\nClass-Level Statistics\n")

        print(
            tabulate(
                class_table,
                headers=[
                    "Class",
                    "TP",
                    "FP",
                    "FN",
                    "TN",
                    "Precision",
                    "Recall",
                    "F1",
                    "Accuracy"
                ],
                tablefmt="grid"
            )
        )


    def _calculate_confusion_matrix(self, labels, predictions):
        """
        Calculate image-level and class-level confusion matrices.

        Parameters
        ----------
        labels : list
            Ground-truth labels for a single image.

        predictions : list
            Predicted labels for a single image.

        Returns
        -------
        dict
            {
                "image": {...},
                "classes": {
                    class_name: {...}
                }
            }
        """

        # ------------------------------------------------------------------
        # Metric Template
        # ------------------------------------------------------------------

        metric_template = {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "accuracy": 0
        }

        # ------------------------------------------------------------------
        # Initialize Metrics Dictionary
        # ------------------------------------------------------------------

        metrics = {
            "image": deepcopy(metric_template),
            "classes": {}
        }

        for class_name in self.unique_classes:
            metrics["classes"][class_name] = deepcopy(metric_template)
        gt_set = set(labels)
        pred_set = set(predictions)

        # image - level confusion matrix

        image_metric = metrics["image"]

        if self.classification_type in ("binary","multiclass"):

            if labels[0] == predictions[0] :
                image_metric["tp"] = 1
            else:
                image_metric["fp"] = 1
                image_metric["fn"] = 1
        else: # multilabel

            image_metric["tp"] = len(gt_set & pred_set)
            image_metric["fp"] = len(pred_set - gt_set)
            image_metric["fn"] = len(gt_set - pred_set)
            image_metric["tn"] = (
                len(self.unique_classes)
                -image_metric["tp"]
                -image_metric["fp"]
                -image_metric["fn"]
                )
        # class-level confusion matrix

        for class_name in self.unique_classes:
            class_metric = metrics["classes"][class_name]

            gt_present = class_name in gt_set
            pred_present = class_name in pred_set

            if gt_present and  pred_present:
                class_metric["tp"] = 1
            elif (not gt_present) and pred_present :
                class_metric["fp"] = 1
            elif gt_present and (not pred_present) :
                class_metric["fn"] = 1
            else:
                class_metric["tn"] = 1

        return metrics
  
    def _calculate_classification_metrics(self, metrics):
        """
        Calculate precision, recall, F1-score and accuracy from
        confusion matrix values.

        Parameters
        ----------
        metrics : dict
            Output from _calculate_confusion_matrix().

        Returns
        -------
        dict
            Metrics dictionary with classification metrics filled.
        """

        for metric_values in metrics.values():
            #skip the classes dictionary
            if isinstance(metric_values,dict) and "tp" not in metric_values:
                for class_metric in metric_values.values():

                    tp=class_metric["tp"]
                    fp=class_metric["fp"]
                    fn=class_metric["fn"]
                    tn=class_metric["tn"]

                    #precison
                    if tp+fp > 0:
                        class_metric["precision"] = tp/ (tp+fp)
                    #recall

                    if tp+fn>0:
                        class_metric["recall"] = tp / (tp+fn)

                    #f1
                    precision = class_metric["precision"]
                    recall = class_metric["recall"]

                    if precision + recall > 0:
                        class_metric["f1"] = (2*precision*recall) / (precision+recall)

                    #Accuracy
                    total = fp+tp+fn+tn
                    if total > 0:
                        class_metric["accuracy"] =(tp+tn)/total

            else:
                tp = metric_values["tp"]
                fp = metric_values["fp"]
                fn = metric_values["fn"]
                tn = metric_values["tn"]

                if tp + fp > 0:
                    metric_values["precision"] = tp / (tp + fp)

                # Recall
                if tp + fn > 0:
                    metric_values["recall"] = tp / (tp + fn)

                # F1
                precision = metric_values["precision"]
                recall = metric_values["recall"]

                if precision + recall > 0:
                    metric_values["f1"] = (
                        2 * precision * recall
                    ) / (precision + recall)

                # Accuracy
                total = tp + fp + fn + tn

                if total > 0:
                    metric_values["accuracy"] = (
                        tp + tn
                    ) / total
        return metrics

                        