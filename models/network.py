import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from pathlib import Path
import datetime
from models.metrics import compute_criteria

# consider different optimizers
dict_optimizers = {"adam": torch.optim.Adam, "rmsp": torch.optim.RMSprop, "sgd": torch.optim.SGD}


class BlockOrder(nn.Module):
    def __init__(self, k_block):
        super(BlockOrder, self).__init__()
        self.k_block = k_block  # k-th block
        self.w2 = nn.Parameter(torch.distributions.Uniform(-1, 0).sample(), requires_grad=True)  # w_2
        self.w3 = nn.Parameter(torch.distributions.Uniform(-1, 0).sample(), requires_grad=True)  # w_3
        self.w4 = nn.Parameter(torch.distributions.Uniform(-1, 0).sample(), requires_grad=True)  # w_4
        return

    def forward(self, inputs):
        x1 = inputs[:, :1]
        x2 = inputs[:, 1:]

        fc1 = F.elu(self.w2 * x1 + self.w3 * x2, alpha=1)
        fc2 = F.elu(self.w4 * x2, alpha=1)

        out = fc1 - fc2
        return out

class fNet(nn.Module):
    def __init__(self, trunc, seed=123):
        super(fNet, self).__init__()
        torch.manual_seed(seed)
        self.n_blocks = int(trunc * (trunc - 1) / 2)  # J(J-1)/2
        self.order_blocks = nn.ModuleList([BlockOrder(k_block) for k_block in range(self.n_blocks)])  # parallel blocks
        self.w0 = nn.Parameter(torch.distributions.Uniform(0, 1).sample(), requires_grad=True)  # w_0 > 0 (gamma estimator)
        self.w1 = nn.Parameter(torch.distributions.Uniform(-1, 0).sample((self.n_blocks, 1)), requires_grad=True)
        return

    def forward(self, inputs):
        x1 = inputs[:, :1]
        varphi = 0.  # varphi
        for i, order_block in enumerate(self.order_blocks):
            varphi += order_block(inputs) * self.w1[i]

        return varphi + (self.w0 * x1) #+ self.biais

class ExtrapolateNN():
    def __init__(self, trunc, optimizer, lr, lamb=0., seed=123, model_filename=None, **kwargs):

        # init numpy and otrch seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.trunc = trunc

        # model
        self.device = self.get_device()
        self.net = fNet(trunc, seed).to(self.device)

        # optimization
        self.optimizer = dict_optimizers[optimizer](self.net.parameters(), lr=lr)
        self.loss_criterion = None
        self.lr = lr
        self.lamb = lamb # for regularization term

        # evaluation criteras
        self.running_loss = 0.
        self.running_variance = 0.
        self.running_r_variance = 0.
        self.running_mad = 0.
        self.running_r_mad = 0.
        self.running_aad = 0.
        self.running_r_aad = 0.
        self.n_batches = 0.

        # set model filename
        self.model_filename = model_filename
        if model_filename is None:
            now = datetime.datetime.now()
            model_filename = now.strftime("%Y-%m-%d_%H-%M-%S")
            self.model_filename = model_filename

        # paths
        self.pt_pathfile = None

        # dict
        self.dict_criteria = {"loss": None, "variance": None, "r_variance": None,
                              "mad": None,"r_mad": None,
                              "aad": None, "r_aad": None}
        self.init_dict_crit()
        return

    def train(self, data_train, X, n_epochs, loss, batch_size, verbose, distribution=None, y=None, **kwargs):
        ckpt_pathdir = Path("ckpt", distribution, "training")

        # init paths
        ckpt_pathdir.mkdir(parents=True, exist_ok=True)  # check if the directory exists
        self.pt_pathfile = Path(ckpt_pathdir, "{}.pt".format(self.model_filename))

        # build dataset and shuffle iid
        trainset = torch.utils.data.DataLoader(torch.from_numpy(data_train), batch_size=batch_size, shuffle=True, worker_init_fn=np.random.seed(0))
        self.n_batches = len(trainset)  # number of batches

        for epoch in range(1, n_epochs + 1):
            start_time = time.time()
            self.running_loss = 0.
            for i, data in enumerate(trainset):  # reshuffle each epoch
                data = data.to(self.device)

                # get inputs/outputs data
                inputs, z_real = data[:, 0:2], data[:, 2:]

                # zero the parameter gradients
                self.net.zero_grad()

                # forward + backward + update
                z_pred = self.net(inputs)
                reg = self.regularization()

                # in case some inf values, don't take the gradient into account
                if torch.isinf(z_pred).any():
                    self.net.zero_grad()
                    idx_inf = torch.isinf(z_pred)
                    z_pred = self.net(inputs[~idx_inf[:, 0]])
                    _loss = self.loss_function(z_real=z_real[~idx_inf].reshape(-1, 1), z_pred=z_pred, method=loss) + self.lamb * reg
                    print("\t WARNING: {} inf value(s) occurred during this iteration".format(idx_inf.sum()))  # TODO: put in log file
                    print("\t Inputs responsible:", inputs[idx_inf[:, 0]])
                else:
                    _loss = self.loss_function(z_real=z_real, z_pred=z_pred, method=loss) + self.lamb*reg
                _loss.backward()  # compute gradients
                self.optimizer.step()  # update weights
                self.running_loss += (_loss/self.n_batches)

                # break

            if epoch % verbose == 0 and epoch > 10:  # action every verbose epochs
                self.save_checkpoint(epoch, X, y)  # save parameters, losses
                time_epoch = time.time() - start_time
                print("Epoch {} ({:.2f} sec): Loss train={:.3f} | Variance={:.3f} | RVariance={:.3f} | MAD={:.3f} | RMAD={:.3f} | AAD={:.3f} | RAAD={:.3f}".format(
                    epoch, time_epoch, self.running_loss, self.running_variance, self.running_r_variance,
                    self.running_mad, self.running_r_mad, self.running_aad, self.running_r_aad))
        return

    def regularization(self):
        """
        Sign penalization of some parameters
        Returns
        -------

        """
        list_reg = []
        for i, parameter in enumerate(self.net.parameters()):
            if i == 0:  # w0 must be positive
                param = parameter.view(-1)  # flatten
                list_reg.append(torch.minimum(torch.zeros_like(param, device=self.device), param))
            if i > 1:  # don't take w0 and w1
                param = parameter.view(-1)  # flatten
                list_reg.append(torch.maximum(torch.zeros_like(param, device=self.device), param))
        return torch.sum(torch.cat(list_reg))

    # Evaluation
    # ======================
    def save_checkpoint(self, epoch, X, y):
        """save parameters and evaluation metrics"""
        self.eval_criterias(epoch, X, y)
        if not self.pt_pathfile.is_file():
            # first epoch
            dict_ckpt = {}
            self.write_params(dict_ckpt, epoch)  # save NN parameters
        else:
            # read and append the dictionary
            dict_ckpt = torch.load(self.pt_pathfile)
            self.write_params(dict_ckpt, epoch)  # # save NN parameters
        self.write_eval(dict_ckpt)  # save evaluation
        # erase the file and save parameters each epoch called: in case of crash during training
        torch.save(dict_ckpt, self.pt_pathfile)
        return

    def write_params(self, dict_ckpt, epoch):
        dict_ckpt["epoch{}".format(epoch)] = {}
        dict_ckpt["epoch{}".format(epoch)]["params"] = self.net.state_dict()
        dict_ckpt["epoch{}".format(epoch)]["optimizer"] = self.optimizer.state_dict()
        return dict_ckpt

    def write_eval(self, dict_ckpt):
        # save best loss and the associated epoch
        dict_ckpt["eval"] = self.dict_criteria
        return dict_ckpt

    def eval_criterias(self, epoch, X_order, *args, **kwargs):
        """criterias evaluation and save the best epoch"""
        with torch.no_grad():
            n_data = X_order.shape[0]
            EXTREME_ALPHA = 1/(2*n_data)  # extreme alpha TODO: put it as a parameter in the yaml file

            k_anchor = np.arange(2, n_data)  # k=2,...,n-1
            q_nn = self.extrapolate(alpha=EXTREME_ALPHA, k_anchor=k_anchor, X_order=X_order).ravel()  # extrapolated quantile neural network
            X = q_nn[13:int(3*n_data/4) + 1]  # for k=15,...,375 (i=13,...,373) HARD CODE

            self.running_variance = compute_criteria(X, "variance")  # variance
            self.running_r_variance = compute_criteria(X, "r_variance")  # relative variance
            self.running_mad = compute_criteria(X, "mad")  # median absolute deviation
            self.running_r_mad = compute_criteria(X, "r_mad")  # relative median absolute deviation
            self.running_aad = compute_criteria(X, "aad")  # absolute average deviation
            self.running_r_aad = compute_criteria(X, "r_aad")  # relative absolute average deviation

            self.update_eval(self.running_loss, "loss", epoch)
            self.update_eval(self.running_variance, "variance", epoch)
            self.update_eval(self.running_r_variance, "r_variance", epoch)
            self.update_eval(self.running_mad, "mad", epoch)
            self.update_eval(self.running_r_mad, "r_mad", epoch)
            self.update_eval(self.running_aad, "aad", epoch)
            self.update_eval(self.running_r_aad, "r_aad", epoch)
        return



    def update_eval(self, metric, criteria, epoch):
        """update the dict_criteria to get the best epoch associated with the smallest metric value"""
        if self.dict_criteria[criteria]["value"] is None or metric < self.dict_criteria[criteria]["value"]:
            self.dict_criteria[criteria]["value"] = metric
            self.dict_criteria[criteria]["epoch"] = epoch
        return

    def extrapolate(self, alpha, k_anchor, X_order, get_y=False, **kwargs):
        """
        extrapolate at the extreme order alpha based on the anchor point
        Parameters
        ----------
        alpha : ndarray
            float
        k_anchor : list
            k=[2, ..., n-1]
        X_anchor: quantile order X_{n-[n\beta]+1, n}
            float
        get_y: bool
            if True, returns the output of the function f^NN

        Returns
        -------

        """
        alphas = np.ones_like(k_anchor) * alpha
        betas = k_anchor / len(X_order)
        x = np.log(betas / alphas).reshape(-1, 1)
        z = np.log(1/betas).reshape(-1, 1)
        X_anchor = X_order[-k_anchor]

        inputs = torch.tensor(np.float32(np.concatenate([x, z], axis=1)), device=self.device)
        y_pred = self.net(inputs).detach().cpu().numpy()

        if get_y:
            return X_anchor * np.exp(y_pred), y_pred

        return X_anchor * np.exp(y_pred)


    def init_dict_crit(self):
        """intialize the criteria dictionary"""
        for k in self.dict_criteria.keys():
            self.dict_criteria[k] = {'value': None, 'epoch': None}
        return

    def loss_function(self, z_real, z_pred, method, **kwargs):
        if method=="l1":
            l1 = nn.L1Loss()
            return l1(z_real, z_pred)
        elif method=="l2":
            l2=nn.MSELoss()
            return l2(z_real, z_pred)
        else:
            print("Loss method is not defined")
        return


    @staticmethod
    def get_device():
        """
        run on CPU or GPU mode

        Parameters
        ----------
        n_gpu: int
            number of GPUs. 0 if run on CPU
        """
        if torch.cuda.is_available():
            return torch.device("cuda:0")  # on GPU
        else:
            return torch.device("cpu")  # on CPU





