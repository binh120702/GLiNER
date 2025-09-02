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
    paths = glob.glob(os.path.join(path, "*.conll"))
    print(f"Found {len(paths)} files in the path {path}")
    train, dev, test = None, None, None
    for p in paths:
        if "train" in p:
            with open(p, "r", encoding='utf-8') as f:
                train = f.readlines()
        elif "dev" in p:
            with open(p, "r", encoding='utf-8') as f:
                dev = f.readlines()
        elif "test" in p:
            with open(p, "r", encoding='utf-8') as f:
                test = f.readlines()
    return train, dev, test


def transform_multiconer_data(data):
    processed_data = []
    tokenized_text = []
    tags = []
    for i in data:
        if i.startswith('# id'):
            if tokenized_text:
                processed_data.append({"tokenized_text": tokenized_text, "tag": tags})
                tokenized_text = []
                tags = []
        else:
            if i.strip() != '':
                token = i.split(' _ _ ')[0].strip()
                tag = i.split(' _ _ ')[1].strip()
                tokenized_text.append(token)
                tags.append(tag)

    for sent in processed_data:
        ner = []
        current_ner = ''
        pos = 0
        for token, tag in zip(sent["tokenized_text"], sent["tag"]):
            if tag == 'O':
                if current_ner != '':
                    ner.append(current_ner)
                    current_ner = ''
            elif current_ner == '':
                current_ner = [pos, pos, tag.split('-')[1].lower()]
            else:
                current_ner[1] += 1 
            pos += 1
        if current_ner != '':
            ner.append(current_ner)
        sent["ner"] = ner
    return processed_data

# create dataset
def create_dataset(path):
    '''
    Location (LOC) : Facility, OtherLOC, HumanSettlement, Station
    Creative Work (CW) : VisualWork, MusicalWork, WrittenWork, ArtWork, Software
    Group (GRP) : MusicalGRP, PublicCORP, PrivateCORP, AerospaceManufacturer, SportsGRP, CarManufacturer, ORG
    Person (PER) : Scientist, Artist, Athlete, Politician, Cleric, SportsManager, OtherPER
    Product (PROD) : Clothing, Vehicle, Food, Drink, OtherPROD
    Medical (MED) : Medication/Vaccine, MedicalProcedure, AnatomicalStructure, Symptom, Disease
    '''
    train, dev, test = open_content(path)
    train_dataset = transform_multiconer_data(train)
    dev_dataset = transform_multiconer_data(dev)
    test_dataset = transform_multiconer_data(test)
    # lower case the entity types
    entity_types = ['loc', 'cw', 'grp', 'per', 'prod', 'med']
    return train_dataset, dev_dataset, test_dataset, entity_types


@torch.no_grad()
def get_for_one_path_multiconer(path, model):
    # load the dataset
    _, _, test_dataset, entity_types = create_dataset(path)

    # eval only on 10% of the test dataset
    # shuffle the dataset with fixed seed
    random.seed(111)
    random.shuffle(test_dataset)
    test_dataset = test_dataset[:len(test_dataset)//10]

    data_name = path.split("/")[-1]  # get the name of the dataset

    # check if the dataset is flat_ner
    flat_ner = True

    # evaluate the model
    results, f1 = model.evaluate(test_dataset, flat_ner=flat_ner, threshold=0.5, batch_size=4,
                                 entity_types=entity_types)
    return data_name, results, f1


def get_for_all_path_multiconer(model, steps, log_dir, data_paths):
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
        if "Multilingual" not in p and "MIX" not in p:
            data_name, results, f1 = get_for_one_path_multiconer(p, model)
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