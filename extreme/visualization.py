import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import display
import re
from pathlib import Path

from utils import load_summary_file
from extreme.data_management import DataSampler, load_quantiles
from extreme.estimators import evt_estimators
from models import load_model, model_evaluation




def training_plot(k_anchor, epoch=None, show_as_video=False, saved=False, **model_filenames):
    """
    Regression plot

    Parameters
    ----------
    k_anchor : int
        anchor point
    epoch : int
        NN iteration
    show_as_video : bool
        visualize all iterations up to 'epoch'
    saved : bool
        save the figure
    model_filenames : dict
        name of the models to plot; {"label": name_model}

    Returns
    -------

    """
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    colors = plt.cm.rainbow(np.linspace(0, 1, len(model_filenames)))

    _, model_filename = list(model_filenames.items())[0]
    summary_file = load_summary_file(model_filename)
    rep = int("".join(re.findall('rep([0-9]+)*$', model_filename)))

    n_data = summary_file["n_data"]
    beta = k_anchor / n_data
    z = np.log(1/beta).reshape(-1, 1)
    data_sampler = DataSampler(**summary_file)
    if epoch is None:  # by defaut the epoch selected is the last one
        epoch = summary_file["n_epochs"]

    alpha = np.arange(1, k_anchor)[::-1] / n_data  # k-1/n, ..., 1/n
    i_indices = np.arange(1, k_anchor)[::-1]
    x = np.log(k_anchor / i_indices).reshape(-1, 1)
    inputs = np.float32(np.concatenate([x, z * np.ones_like(x)], axis=1))

    X_order = load_quantiles(**summary_file, rep=rep)  # load quantiles X_1,n, ..., X_n,n
    real_quantiles = [data_sampler.ht_dist.tail_ppf(n_data/_i) for _i in np.arange(1, k_anchor)[::-1]]  # simulate the real quantile

    X_anchor = X_order[-k_anchor]  # anchor point estimated with order statistics
    real_anchor = data_sampler.ht_dist.tail_ppf(n_data/k_anchor)

    y_order = np.log(X_order[-(k_anchor-1):]) - np.log(X_anchor)
    y_real = np.log(real_quantiles) - np.log(real_anchor)  # real_anchor ou X_anchor

    def _training_plot(epoch):
        fig, axes = plt.subplots(1, 1, figsize=(12, 7), sharex=False, squeeze=False)
        for idx_model, (order_trunc, model_filename) in enumerate(model_filenames.items()):
            if idx_model == 0:  # plot reference line
                plt.plot(x.ravel(), y_real.ravel(),  color='black', linewidth=2, label="real function")  # real function
                sns.scatterplot(x=x.ravel(), y=y_order.ravel(),  marker="o", color='C2', s=50, label="Order stat")

            # NN predictions
            model = load_model(filename=model_filename, epoch=epoch, distribution=summary_file["distribution"])
            y_pred = model.net(torch.tensor(inputs)).detach().numpy()
            sns.scatterplot(x=x.ravel(), y=y_pred.ravel(), color=colors[idx_model], marker="o", s=50, label="NN")
            # plt.plot(x.ravel(), y_pred.ravel(), color=colors[idx_model], linewidth=2)

        axes[0, 0].legend()
        # axes[0, 0].set_xlabel(r"$x$")
        # axes[0, 0].set_ylabel("log spaces $Y$")
        axes[0, 0].spines["left"].set_color("black")
        axes[0, 0].spines["bottom"].set_color("black")
        axes[0, 0].set_title("Regression plot\n{}: {} \n(epoch={})".format(summary_file["distribution"].capitalize(),
                                                                           str(summary_file["params"]).upper(),
                                                                            epoch), fontweight="bold")
        # axes[0, 0].set_ylim(-5, 2)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        fig.tight_layout()
        sns.despine()
        if saved:
            plt.savefig("imgs/f_funcNN-{}-{}.eps".format(summary_file["distribution"], str(summary_file["params"])), format="eps")
        return

    if show_as_video:
        save_freq = summary_file["verbose"]
        ckpt_epochs = [save_freq] + [i for i in range(save_freq, epoch + save_freq, save_freq)]
        for chkpt_epoch in ckpt_epochs:
            _training_plot(chkpt_epoch)
            plt.show()
            display.clear_output(wait=True)
            #time.sleep(1)
    else:
        _training_plot(epoch)
    return


def xquantile_plot(criteria="mad", metric="median", **model_filenames):
    """extreme quantile plot at level 1/2n for different replications"""

    # take the first one to load the config summary and extract data infos
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    colors = plt.cm.rainbow(np.linspace(0, 1, len(model_filenames)))

    _, model_filename = list(model_filenames.items())[0]
    summary_file = load_summary_file(filename=model_filename+"-rep1")

    # assume all models have the same number of data and replications
    n_data = summary_file["n_data"]
    n_replications = summary_file["replications"]

    pathdir = Path("ckpt", summary_file["distribution"], "extrapolation", str(summary_file["params"]))
    pathdir.mkdir(parents=True, exist_ok=True)

    EXTREME_ALPHA = 1/(2*n_data)  # pick the extreme alpha
    anchor_points = np.arange(2, n_data)  # 1, ..., n-1

    # real data
    data_sampler = DataSampler(**summary_file)
    real_quantile = data_sampler.ht_dist.tail_ppf(1/EXTREME_ALPHA)  # real extreme quantile

    fig, axes = plt.subplots(2, 1, figsize=(15, 2 * 5), sharex=False, squeeze=False)  # 3 plots: quantile, var, mse

    try:
        dict_evt = np.load(Path(pathdir, "evt_estimators_rep{}_ndata{}.npy".format(n_replications, n_data)), allow_pickle=True)[()]
    except FileNotFoundError:
        print("Training EVT estimators ...")
        dict_evt = evt_estimators(n_replications, n_data, summary_file["distribution"], summary_file["params"], return_full=True,
                                  metric=metric)

    for idx_model, (trunc_condition, model_filename) in enumerate(model_filenames.items()):
        pathfile = Path(pathdir, "{}.npy".format(model_filename))

        try:
            dict_nn = np.load(pathfile, allow_pickle=True)[()]
        except FileNotFoundError:
            print("Model Selection ...")
            dict_nn = model_evaluation(model_filename)

        for replication in range(1, n_replications + 1):
            model_mean = dict_nn[criteria][metric]["series"]
            model_rmse = dict_nn[criteria][metric]["rmse"]  # series for different k
            model_rmse_bestK = dict_nn[criteria][metric]["rmse_bestK"]


        # plot NN
        axes[0, 0].plot(anchor_points, model_mean,  label="{}: {:.4f}".format(trunc_condition, model_rmse_bestK), color=colors[idx_model])
        axes[1, 0].plot(anchor_points, model_rmse, label="{}: {:.4f}".format(trunc_condition, model_rmse_bestK), color=colors[idx_model])

    for estimator in dict_evt.keys():
        axes[0, 0].plot(anchor_points, dict_evt[estimator][metric]["series"],
                        label="{}: {:.4f}".format(estimator, dict_evt[estimator][metric]["rmse_bestK"]), linestyle="-.")

        axes[1, 0].plot(anchor_points, dict_evt[estimator][metric]["rmse"],
                        label="{}: {:.4f}".format(estimator, dict_evt[estimator][metric]["rmse_bestK"]), linestyle="-.")

    axes[0, 0].hlines(y=real_quantile, xmin=0., xmax=n_data, label="reference line", color="black", linestyle="--")

    axes[0, 0].legend()
    axes[0, 0].spines["left"].set_color("black")
    axes[0, 0].spines["bottom"].set_color("black")

    # title / axis
    axes[0, 0].set_xlabel(r"anchor point $k$")
    axes[0, 0].set_ylabel("quantile")
    axes[0, 0].set_title("Median estimator")

    axes[1, 0].set_xlabel(r"anchor point $k$")
    axes[1, 0].set_ylabel("RMedSE")
    axes[1, 0].set_title("RMedSE")
    axes[1, 0].spines["left"].set_color("black")
    axes[1, 0].spines["bottom"].set_color("black")

    # y_lim
    axes[0, 0].set_ylim(real_quantile*0.5, real_quantile*2)  # QUANTILE
    axes[1, 0].set_ylim(0, 1)  # RMedSE



    fig.tight_layout()
    fig.suptitle("Estimator plot \n{}: {}".format(summary_file["distribution"].capitalize(), str(summary_file["params"]).upper()), fontweight="bold", y=1.04)
    sns.despine()
    return


