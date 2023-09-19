import csv
import os
from pprint import pprint

from utlis.plot_figure import plot_figure


def save_csv(dir_path, values):
    save_path = f'{dir_path}/loss_acc.csv'
    if not os.path.isfile(save_path):
        headers = ['acc', 'loss']
        with open(f'{save_path}', 'w', encoding='utf-8', newline='') as fp:
            writer = csv.writer(fp)
            writer.writerow(headers)
    with open(f'{save_path}', 'a', encoding='utf-8', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(values)


def classifaction_report_csv(save_path, report, names):
    for c in names:
        target_values = report[f'{c}']
        save_csv_path = save_path + f"/{c}.csv"
        if not os.path.isfile(save_path):
            with open(save_csv_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['precision', 'recall', 'f1-score'])
        with open(save_csv_path, 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([target_values["precision"], target_values["recall"], target_values["f1-score"]])


def print_result(report):
    report.pop('weighted avg')
    report.pop('macro avg')
    pprint(report)


def save_metric(save_path, report, loss, names):
    os.makedirs(save_path, exist_ok=True)

    classifaction_report_csv(save_path, report, names)
    acc = report['accuracy']
    save_csv(save_path, [acc, loss])
    plot_figure(save_path)
