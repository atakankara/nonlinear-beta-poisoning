from secml.array import CArray
from src.classifier.secml_autograd import as_tensor
import torch
import torch.nn as nn
import math

class OnFeatureSpace:
    def __init__(self, ds, clf, h_coeff, store_ds=False, kernel="gaussian"):
        self.data = ds.X.tondarray()
        self.labels = ds.Y.unique()
        self.clf = clf
        self.stat_register = {}
        self.ds_register = {}

        ylst = ds.Y.unique()
        for y in ylst:
            # embed data point with phi function
            ds_y = self.clf.transform_all(ds[ds.Y == y, :].X)
            if isinstance(ds_y, CArray):
                ds_y = as_tensor(ds_y).to(clf.device)
            # get stats on phi space
            mu = self._mu(ds_y)
            sigma = self._sigma(ds_y)
            self.stat_register[str(y)] = {"mu": mu, "sigma": sigma}
            if store_ds:
                # if h == "average_distance":
                h = (torch.cdist(ds_y, ds_y).mean() / 2)*h_coeff

                if kernel == "gaussian":   
                    self.ds_register[str(y)] = GKDE(ds_y, bw=h)
                elif kernel == "tophat":
                    self.ds_register[str(y)] = TophatKDE(ds_y, bw=h)
                elif kernel == "epanechnikov":
                    self.ds_register[str(y)] = EpanechnikovKDE(ds_y, bw=h)
                elif kernel == "gaussian2":
                    self.ds_register[str(y)] = GKDE2(ds_y, bw=h)
                elif kernel == "logistic":
                    self.ds_register[str(y)] = LogisticKDE(ds_y, bw=h)
                elif kernel == "sigmoid":
                    self.ds_register[str(y)] = SigmoidKDE(ds_y, bw=h)
                elif kernel == "mlp":
                    self.ds_register[str(y)] = MlpKDE(ds_y, bw=h)
                else:
                    raise Exception()

    def _mu(self, data):
        return data.mean(axis=0)

    def _sigma(self, data):
        return data.var(axis=0)


########################### Gausian (Laplacian) ######################

class KDEGaussian(OnFeatureSpace):
    def __init__(self, ds, clf, h):
        super().__init__(ds, clf, h, store_ds=True, kernel="gaussian")
        self.__name__ = "gaussian_kernel"
    def pr(self, x, y=None):
        kde = self.get_dist(y)
        phi_x = self.clf.transform(x)
        p = kde(phi_x)
        return p

    def get_dist(self, y):
        key = str(y)
        return self.ds_register[key]


class GKDE(nn.Module):
    def __init__(self, data, bw):
        super().__init__()
        self.data = data
        self.bw = bw

    def forward(self, x):
        n = self.data.shape[0]
        num = -(self.data - x).norm(2, dim=1) / self.bw
        p = torch.exp(num).sum()
        p *= 1 / n
        return p

########################### Tophat ######################

class KDETophat(OnFeatureSpace):
    def __init__(self, ds, clf, h):
        super().__init__(ds, clf, h, store_ds=True, kernel="tophat")

    def pr(self, x, y=None):
        kde = self.get_dist(y)
        phi_x = self.clf.transform(x)
        p = kde(phi_x)
        return p

    def get_dist(self, y):
        key = str(y)
        return self.ds_register[key]


class TophatKDE(nn.Module):
    def __init__(self, data, bw):
        super().__init__()
        self.data = data
        self.bw = bw

    def forward(self, x):
        n = self.data.shape[0]
        num = (self.data - x).norm(2, dim=1)
        number_of_elements = torch.tensor(torch.numel(num[num>self.bw]), dtype=torch.float)
        p = number_of_elements
        p *= 1 / n
        return p

########################### Epanechnikov ######################

class KDEEpanechnikov(OnFeatureSpace):
    def __init__(self, ds, clf, h):
        super().__init__(ds, clf, h, store_ds=True, kernel="epanechnikov")
        self.__name__ = "epanechnikov_kernel"

    def pr(self, x, y=None):
        kde = self.get_dist(y)
        phi_x = self.clf.transform(x)
        p = kde(phi_x)
        return p

    def get_dist(self, y):
        key = str(y)
        return self.ds_register[key]


class EpanechnikovKDE(nn.Module):
    def __init__(self, data, bw):
        super().__init__()
        self.data = data
        self.bw = bw

    def forward(self, x):
        n = self.data.shape[0]
        num = 1 - (torch.square((self.data - x).norm(2, dim=1)) / torch.square(self.bw*5))
        p = num.sum()
        p *= 1 / n
        return p

########################### Gaussian2 ######################

class KDEGaussian2(OnFeatureSpace):
    def __init__(self, ds, clf, h):
        super().__init__(ds, clf, h, store_ds=True, kernel="gaussian2")
        self.__name__ = "gaussian2_kernel"

    def pr(self, x, y=None):
        kde = self.get_dist(y)
        phi_x = self.clf.transform(x)
        p = kde(phi_x)
        return p

    def get_dist(self, y):
        key = str(y)
        return self.ds_register[key]


class GKDE2(nn.Module):
    def __init__(self, data, bw):
        super().__init__()
        self.data = data
        self.bw = bw

    def forward(self, x):
        n = self.data.shape[0]
        num = -torch.square((self.data - x).norm(2, dim=1)) / (2*torch.square(self.bw))
        p = torch.exp(num).sum()
        p *= 1 / n
        return p


########################### Logistic ######################

class KDELogistic(OnFeatureSpace):
    def __init__(self, ds, clf, h):
        super().__init__(ds, clf, h, store_ds=True, kernel="logistic")
        self.__name__ = "logistic_kernel"

    def pr(self, x, y=None):
        kde = self.get_dist(y)
        phi_x = self.clf.transform(x)
        p = kde(phi_x)
        return p

    def get_dist(self, y):
        key = str(y)
        return self.ds_register[key]


class LogisticKDE(nn.Module):
    def __init__(self, data, bw):
        super().__init__()
        self.data = data
        self.bw = bw
        n = self.data.shape[0]

        self.optimal_h = ((n*(math.pi**4)) / 63) ** (-1/5) 

    def forward(self, x):
        n = self.data.shape[0]
        num = torch.exp(-(self.data - x).norm(2, dim=1) / self.optimal_h) / torch.square(1 + torch.exp(-(self.data - x).norm(2, dim=1) / self.optimal_h))
        p = num.sum()
        p *= 1 / (n*self.optimal_h)
        return p


########################### Sigmoid ######################

class KDESigmoid(OnFeatureSpace):
    def __init__(self, ds, clf, h):
        super().__init__(ds, clf, h, store_ds=True, kernel="sigmoid")
        self.__name__ = "sigmoid_kernel"

    def pr(self, x, y=None):
        kde = self.get_dist(y)
        phi_x = self.clf.transform(x)
        p = kde(phi_x)
        return p

    def get_dist(self, y):
        key = str(y)
        return self.ds_register[key]


class SigmoidKDE(nn.Module):
    def __init__(self, data, bw):
        super().__init__()
        self.data = data
        self.bw = bw

    def forward(self, x):
        n = self.data.shape[0]
        alpha =  1 / self.data.shape[1]
        num = torch.matmul(self.data, x.T.double())*alpha
        p = torch.tanh(num).sum()
        p *= 1 / n
        return p

########################### MLP ######################

class KDEMlp(OnFeatureSpace):
    def __init__(self, ds, clf, h):
        super().__init__(ds, clf, h, store_ds=True, kernel="mlp")

    def pr(self, x, y=None):
        kde = self.get_dist(y)
        phi_x = self.clf.transform(x)
        p = kde(phi_x, y)
        return p

    def get_dist(self, y):
        key = str(y)
        return self.ds_register[key]
    
class MlpKDE(nn.Module):
    def __init__(self, data, bw):
        super().__init__()
        self.model = torch.load('binary_classifier.pt')
        self.model.eval()
        self.__name__ = "mlp_kernel"

    def forward(self, x, y):
        prob = self.model(x)
        return prob[0][0] / prob[0][0].sum() if y == 0 else prob[0][1] / prob[0][1].sum()


class Feedforward(torch.nn.Module):
        def __init__(self, input_size, hidden_size, pdrop=0.2):
            super(Feedforward, self).__init__()
            self.input_size = input_size
            self.hidden_size  = hidden_size
            self.dropout = torch.nn.Dropout(p=pdrop)
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(self.hidden_size, 2)
            self.softmax = torch.nn.Softmax()
        def forward(self, x):
            after_drop = self.dropout(x)
            hidden = self.fc1(after_drop)
            relu = self.relu(hidden)
            output = self.fc2(relu)
            return output