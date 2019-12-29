import csv
import json
import os
import pickle
from datetime import datetime
import torch
import shutil


def load_from_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_to_pickle(data, path):
    with open(ensure_path(path), "wb") as f:
        pickle.dump(data, f)


def read_json(input_path):
    with open(input_path, "r") as f:
        return json.load(f)


def save_list_as_text(data, output_path):
    output_path = ensure_path(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        for word in data:
            f.write("{}\n".format(word))


def load_text_as_list(input_path):
    with open(input_path, 'r', encoding="utf-8") as f:
        return f.read().splitlines()


def ensure_path(path):
    directory = os.path.dirname(path)
    if len(directory) > 0 and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    return path


def write_csv(content, header, path, delimiter=","):
    path = ensure_path(path)
    with open(path, 'w', encoding="utf-8", newline='') as f:
        csv_writer = csv.writer(f, delimiter=delimiter, quotechar='"', quoting=csv.QUOTE_MINIMAL)
        if header is not None:
            csv_writer.writerow(header)

        for row in content:
            csv_writer.writerow(row)


def read_csv(path, load_header=False, delimiter=",", quotechar='"'):
    content = []
    with open(path, "r", encoding='ISO-8859-1') as f:
        csv_reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
        if load_header:
            [content.append(row) for row in csv_reader]
        else:
            [content.append(row) for i, row in enumerate(csv_reader) if i > 0]
    return content


def write_json(content, path):
    with open(path, "w") as f:
        json.dump(content, f)


def get_exp_name(model_name):
    return "{}-{}".format(model_name, datetime.now().strftime("%D-%H-%M-%S").replace("/", "_"))


def save_model_checkpoint(state, is_best, output_dir, exp_name, step):
    file_name = "model_ckpt_{}.tar".format(step)
    output_path = os.path.join(output_dir, exp_name, file_name)
    directory = os.path.dirname(output_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(state, output_path)
    if is_best:
        best_output_path = os.path.join(output_dir, exp_name, "model_ckpt_best.tar".format(step))
        shutil.copyfile(output_path, best_output_path)
        return best_output_path
    return output_path
