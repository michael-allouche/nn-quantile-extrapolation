import datetime

import pandas as pd
from utils import get_config, save_config_summary
from extreme.data_management import DataSampler
from models import ExtrapolateNN

from multiprocessing import Pool
from pathlib import Path
import torch
import argparse

parser = argparse.ArgumentParser(description='Runner')
parser.add_argument('--processes', '-p',
                    help="number of processes. No multiprocessing by default",
                    default=1,
                    type=int)

args = parser.parse_args()
n_processes = args.processes

config = get_config()  # load .yaml configuration file
condition = len(config["data"]["params"]["rho"]) + 1  # data order condition
config["data"]["order_condition"] = condition
data_sampler = DataSampler(**config["data"])

n_data = config["data"]["n_data"]
n_data = int(n_data - (n_data * config["data"]["percentile"]))

# utilities
verbose = config["training"]["verbose"]
ckpt_epochs = [i for i in range(verbose, config["training"]["n_epochs"] + verbose, verbose)]  # check if all epochs have been trained
csv_pathfile = Path("ckpt", "_config_summary.csv")

now = datetime.datetime.now()
model_filename = now.strftime("%Y-%m-%d_%H-%M-%S")  # schema name model

def train_a_replication(replication=1):
    """
    Training of just one replication
    Parameters
    ----------
    replication : int
        associated to the seed for the generating simulated data

    Returns
    -------

    """
    print("REPLICATION #{}".format(replication))
    model_filename_rep = model_filename + "-rep{}".format(replication)
    pathfile = Path("ckpt", config["data"]["distribution"], "training", "{}.pt".format(model_filename_rep))
    if pathfile.is_file():  # if the file exists, don't train it again !
        try:
            pt_ckpt = torch.load(pathfile, map_location="cpu")
            for chkpt_epoch in ckpt_epochs:  # check that all epochs have been trained
                pt_ckpt["epoch{}".format(chkpt_epoch)]["params"]
            return  # if it is the case, move to the next replication file
        except (EOFError, KeyError):  # if one is missing, remove the file and the csv line
            pathfile.unlink()  # file .pt
            df_summary = pd.read_csv(csv_pathfile, sep=";")   # associated csv row
            df_summary.drop(df_summary.index[(df_summary["model_filename"]==model_filename_rep)], inplace=True)
            df_summary.dropna(axis=0, how="all", inplace=True)
            df_summary.to_csv(csv_pathfile, header=True, index=False, sep=";")

    # regular training
    data, X_order = data_sampler.load_simulated_data(n_data=n_data, rep=replication)
    total_data = data.shape[0]  # nb of log-spacings
    config["data"]["total_data"] = total_data

    save_config_summary(config=config, model_filename=model_filename_rep)  # save the config in a csv file

    print("="*10, "Training on {} distribution ({}) with a total of {} data: {}".format(config["data"]["distribution"].upper(),
                                                                                        config["data"]["params"],
                                                                                        total_data, model_filename_rep), "="*10)
    model = ExtrapolateNN(**config["model"], model_filename=model_filename_rep)
    model.train(data_train=data, X=X_order, distribution=config["data"]["distribution"], **config["training"])
    return



if __name__ == '__main__':
    replications = list(range(1, config["training"]["replications"] + 1))
    if n_processes > 1:  # multiprocessing
        pool = Pool(n_processes)
        pool.map(train_a_replication, replications)
        pool.close()
        pool.join()
    else:
        # train_a_replication(replication=config["training"]["replications"])  # for just one replication
        for replication in replications:
            train_a_replication(replication=replication)


