from torchmetrics import Accuracy, F1, Precision, Recall  # , Specificity
from typing import Tuple

# class BaseLogger:
#     def __init__(
#         self, classes, average: str = None, multilabel: bool = True, **kwargs
#     ):
#         self.classes = classes
#         self.num_classes = len(classes)
#         self.average = average
#         self.multilabel = multilabel

#         self.metrics = {
#             "F1": {},
#             # "Specificity": {},
#             "Accuracy": {},
#             "Precision": {},
#             "Recall": {},
#         }

#         (
#             self.metrics["F1"]["train"],
#             self.metrics["F1"]["val"],
#             self.metrics["F1"]["test"],
#         ) = get_f1(self.num_classes, self.average, self.multilabel)
#         # self.metrics["Specificity"]["train"], self.metrics["Specificity"]["val"], self.metrics["Specificity"]["test"] = get_specificity(num_classes)(self.num_classes, self.average, self.multilabel)
#         (
#             self.metrics["Accuracy"]["train"],
#             self.metrics["Accuracy"]["val"],
#             self.metrics["Accuracy"]["test"],
#         ) = get_accuracy(self.num_classes, self.average, self.multilabel)
#         (
#             self.metrics["Precision"]["train"],
#             self.metrics["Precision"]["val"],
#             self.metrics["Precision"]["test"],
#         ) = get_precision(self.num_classes, self.average, self.multilabel)
#         (
#             self.metrics["Recall"]["train"],
#             self.metrics["Recall"]["val"],
#             self.metrics["Recall"]["test"],
#         ) = get_recall(self.num_classes, self.average, self.multilabel)

#         self.metric_hist = {}

#     def calculate_metrics(self, preds, targets, dataset_type: str):
#         f1 = self.metric_to_dict(self.metrics["F1"][dataset_type](preds, targets))
#         precision = self.metric_to_dict(
#             self.metrics["Precision"][dataset_type](preds, targets)
#         )
#         recall = self.metric_to_dict(
#             self.metrics["Recall"][dataset_type](preds, targets)
#         )
#         # specificity = self.metric_to_dict(self.metrics["Specifitiy"][dataset_type](preds, targets))
#         accuracy = self.metric_to_dict(
#             self.metrics["Accuracy"][dataset_type](preds, targets)
#         )

#         metric_dict = {
#             f"{dataset_type}/f1": f1,
#             f"{dataset_type}/accuracy": accuracy,
#             f"{dataset_type}/recall": recall,
#             # f"{dataset_type}/specificity": specificity,
#             f"{dataset_type}/precision": precision,
#         }

#         return metric_dict

def metric_to_dict(classes, metric):
    try:
        metric_dict = {classes[i]: _ for i, _ in enumerate(metric)}
    except:
        metric_dict = None
        print("Classes and Metrics dont match")
        print("Classes:", classes)
        print("Metrics:", metrics)

    return metric_dict


def get_precision(
    num_classes: int, average: str = None, multilabel: bool = True, **kwargs
) -> Tuple[Precision, Precision, Precision]:
    train_precision = Precision(
        num_classes=num_classes, average=average  # , subset_accuracy=multilabel
    )
    val_precision = Precision(
        num_classes=num_classes, average=average  # , subset_accuracy=multilabel
    )
    test_precision = Precision(
        num_classes=num_classes, average=average  # , subset_accuracy=multilabel
    )

    return train_precision, val_precision, test_precision


def get_recall(
    num_classes, average: str = None, multilabel: bool = True, **kwargs
) -> Tuple[Recall, Recall, Recall]:
    train_recall = Recall(
        num_classes=num_classes, average=average  # , subset_accuracy=multilabel
    )
    val_recall = Recall(
        num_classes=num_classes, average=average
    )  # , subset_accuracy=multilabel)
    test_recall = Recall(
        num_classes=num_classes, average=average  # , subset_accuracy=multilabel
    )
    return train_recall, val_recall, test_recall


def get_accuracy(
    num_classes, average: str = None, multilabel: bool = True, **kwargs
) -> Tuple[Accuracy, Accuracy, Accuracy]:
    train_acc = Accuracy(
        num_classes=num_classes, average=average, subset_accuracy=multilabel
    )
    val_acc = Accuracy(
        num_classes=num_classes, average=average, subset_accuracy=multilabel
    )
    test_acc = Accuracy(
        num_classes=num_classes, average=average, subset_accuracy=multilabel
    )
    return train_acc, val_acc, test_acc


def get_f1(
    num_classes, average: str = None, multilabel: bool = True, **kwargs
) -> Tuple[F1, F1, F1]:
    train_f1 = F1(
        num_classes=num_classes, average=average
    )  # , subset_accuracy=multilabel)
    val_f1 = F1(
        num_classes=num_classes, average=average
    )  # , subset_accuracy=multilabel)
    test_f1 = F1(
        num_classes=num_classes, average=average
    )  # , subset_accuracy=multilabel)
    return train_f1, val_f1, test_f1


# def get_specificity(
#     num_classes, average: str = None, multilabel: bool = True, **kwargs
# ) -> (Specificity, Specificity, Specificity):
#     train_specificity = Specificity(
#         num_classes=num_classes, average=average, multilabel=multilabel
#     )
#     val_specificity = Specificity(
#         num_classes=num_classes, average=average, multilabel=multilabel
#     )
#     test_specificity = Specificity(
#         num_classes=num_classes, average=average, multilabel=multilabel
#     )
#     return train_specificity, val_specificity, test_specificity
