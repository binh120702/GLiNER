import glob
import json
import os
import os
import numpy as np
import argparse
import torch
from tqdm import tqdm
import random

def open_content(path):
    paths = glob.glob(os.path.join(path, "*.json"))
    print(f"Found {len(paths)} files in the path {path}")
    train, dev, test = None, None, None
    for p in paths:
        if "train" in p:
            with open(p, "r", encoding='utf-8') as f:
                train = json.load(f)
        elif "dev" in p:
            with open(p, "r", encoding='utf-8') as f:
                dev = json.load(f)
        elif "test" in p:
            with open(p, "r", encoding='utf-8') as f:
                test = json.load(f)
    return train, dev, test

# create dataset
def create_dataset(path):
    train_dataset, dev_dataset, test_dataset = open_content(path)
    # lower case the entity types
    entity_types = ['organization', 'location', 'person', 'miscellaneous']
    return train_dataset, dev_dataset, test_dataset, entity_types


@torch.no_grad()
def get_for_one_path_vlsp(path, model):
    # load the dataset
    _, _, test_dataset, entity_types = create_dataset(path)

    # eval only on 10% of the test dataset
    # shuffle the dataset with fixed seed

    data_name = path.split("/")[-1]  # get the name of the dataset

    # check if the dataset is flat_ner
    flat_ner = True

    # evaluate the model
    results, f1 = model.evaluate(test_dataset, flat_ner=flat_ner, threshold=0.5, batch_size=4,
                                 entity_types=entity_types)
    return data_name, results, f1


def get_for_all_path_vlsp(model, steps, log_dir, data_paths):
    all_paths = glob.glob(f"{data_paths}/*")

    all_paths = sorted(all_paths)

    # move the model to the device
    device = next(model.parameters()).device
    model.to(device)
    # set the model to eval mode
    model.eval()

    # log the results
    save_path = os.path.join(log_dir, "results.txt")

    with open(save_path, "a") as f:
        f.write("##############################################\n")
        # write step
        f.write("step: " + str(steps) + "\n")

    all_results = {} 
    
    for p in tqdm(all_paths):
        data_name, results, f1 = get_for_one_path_vlsp(p, model)
        # write to file
        with open(save_path, "a") as f:
            f.write(data_name + "\n")
            f.write(str(results) + "\n")

        all_results[data_name] = f1

    avg_all = sum(all_results.values()) / len(all_results)

    save_path_table = os.path.join(log_dir, "tables.txt")

    # results for all datasets
    table_bench_all = ""
    for k, v in all_results.items():
        table_bench_all += f"{k:20}: {v:.1%}\n"
    # (20 size aswesave_path_tablell for average i.e. :20)
    table_bench_all += f"{'Average':20}: {avg_all:.1%}"

    # write to file
    with open(save_path_table, "a") as f:
        f.write("##############################################\n")
        f.write("step: " + str(steps) + "\n")
        f.write("Table for all datasets except crossNER\n")
        f.write(table_bench_all + "\n\n")
        f.write("##############################################\n\n")