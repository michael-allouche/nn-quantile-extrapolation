import pandas as pd
import torch
import os
import numpy as np
from pathlib import Path
from collections import defaultdict
import yaml


def get_best_order_model(df_order_summary):
    """returns the best model with gamma=0.5 and rho=-1 for a given order condition"""
    best_loss = None  # init the best loss
    for file in df_order_summary.iterrows():
        filename = file[1]["model_filename"]
        pt_ckpt = torch.load(Path("ckpt",  "{}.pt".format(filename)), map_location="cpu")
        loss = pt_ckpt["metrics"]["loss"]["value"]

        if best_loss is None or loss < best_loss:
            best_loss = loss
            best_epoch = pt_ckpt["metrics"]["loss"]["epoch"]
            best_filename = filename
    return best_filename, best_epoch, best_loss.detach().numpy()

def get_best_models():
    """returns the best models with gamma=0.5 and rho=-1 for each order condition"""
    df_summary = load_summary()
    dict_best_models = defaultdict(list)

    order_conditions = np.sort(df_summary["n_orders"].unique())
    for order in order_conditions:
        df_order_summary = df_summary[df_summary["n_orders"]==order]
        filename, epoch, loss = get_best_order_model(df_order_summary)
        # save it
        dict_best_models["order"].append(order)
        dict_best_models["filename"].append(filename)
        dict_best_models["epoch"].append(epoch)
        dict_best_models["loss"].append(loss)
    return pd.DataFrame.from_dict(dict_best_models)


def load_summary():
    # cleaning
    df_summary = pd.read_csv(Path("ckpt", "_config_summary.csv"), sep=";")
    # remove model_filename replications duplicates
    # ---
    df_summary.drop_duplicates(subset=["model_filename"], inplace=True, keep="last")
    df_summary.dropna(axis=0, how="all", inplace=True)
    df_summary.to_csv(Path("ckpt", "_config_summary.csv"), header=True, index=False, sep=";")  # save it again
    # ----
    return df_summary

def load_summary_file(filename):
    df_summary = load_summary()
    file_summary = df_summary[df_summary["model_filename"] == filename].to_dict('records')[0]
    file_summary["params"] = yaml.safe_load(file_summary["params"])
    return file_summary

def get_config():
    """
    load the .yaml config file
    Returns
    -------
    dict:
        config file
    """

    with(open(Path("configs", "config_file.yaml"))) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader) 
    return config

def save_config(config, model, name_file, name_model):
    pt_pathdir= Path("ckpt", "{}".format(name_model))

    # add new items to the dict config
    config["generator_params"] = model.generator.state_dict()
    config["discriminator_params"] = model.discriminator.state_dict()
    config["generator_opt"] = model.optimizerG.state_dict()
    config["discriminator_opt"] = model.optimizerD.state_dict()

    Path(pt_pathdir).mkdir(parents=True, exist_ok=True)
    torch.save(config, os.path.join(pt_pathdir, "{}.pt".format(name_file)))
    return

def save_config_summary(config, model_filename):
    """save the config summary in a .csv file for each model"""
    pathdir = Path("ckpt")
    pathdir.mkdir(parents=True, exist_ok=True)  # check if the directory exists
    csv_pathfile = Path(pathdir,  "_config_summary.csv")

    flatten_config = flatten_dict(config)
    flatten_config["n_gpu"] = torch.cuda.device_count()  # count number of GPU available during training
    flatten_config["model_filename"] = model_filename
    df = pd.DataFrame(flatten_config)

    if csv_pathfile.is_file():  # if csv file exists
        df.to_csv(csv_pathfile, mode="a", header=False, index=False, sep=";")  # append
    else:
        df.to_csv(csv_pathfile, header=True, index=False, sep=";")  # add
    return

def flatten_dict(nested_dict):
    """flatten a 2 level nested dict"""
    out = {}
    list_subdict = nested_dict.values()
    for subdict in list_subdict:
        for key, val in subdict.items():
            out[key] = [val] #str(val)
    return out

def nested_dict_to_df(nested_dict):
    """convert a nested dict to a multi index Dataframe"""
    new_dict = {}
    for outerKey, innerDict in nested_dict.items():
        for innerKey, values in innerDict.items():
            new_dict[(outerKey, innerKey)] = values
    return pd.DataFrame(new_dict)
