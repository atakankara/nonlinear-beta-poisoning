import torch
import torch.nn as nn

from secml.ml.classifiers import CClassifierSVM, CClassifierLogistic, CClassifierPyTorch
from secml.ml.classifiers.loss import CLossCrossEntropy
from src.classifier.secml_autograd import SecmlLayer, as_tensor
from src.classifier.torch_classifier import Mlp, ConvNN


def to_tensor(x, device="cpu"):
    x = as_tensor(x).unsqueeze(0)
    x = x.float()
    x = x.to(device)
    return x


class SecmlClassifier:
    def __init__(self, clf=None, preprocess=None, n_classes=2):
        self.clf = clf
        self.fitted = False
        self.n_labels = n_classes
        self.C = self.alpha = None
        self._preprocess = preprocess
        self.device = "cpu"

        if self._preprocess is None:
            self._preprocess = to_tensor
        self.device = "cpu"

    def loss(self, x, labels):
        scores = self.clf.decision_function(x)
        loss = self.clf._loss.loss(labels, scores).sum()
        return loss

    def ce_loss(self, x, labels):
        scores = self.clf.decision_function(x)
        prob = scores  # scores.exp() / (scores.exp()).sum(axis=1)
        ce = CLossCrossEntropy()
        return ce.loss(y_true=labels, score=prob).sum()

    def predict(self, x, return_decision_function=False):
        if not return_decision_function:
            predicted = self.clf.predict(x)
            return predicted
        else:
            predicted, scores = self.clf.predict(x, return_decision_function=True)
            return predicted, scores

    def _fit(self, ds):
        return self.fit(ds)

    def _forward(self, x):
        return self.predict(x)

    def fit(self, ds):
        self.clf.fit(ds.X, ds.Y)

    def init(self):
        pass

    def init_fit(self, ds, parameters):
        svm = CClassifierSVM(C=parameters["C"], store_dual_vars=True)
        self.clf = svm
        self.fit(ds.X, ds.Y)

    def preprocess(self, x):
        return self._preprocess(x)

    def deepcopy(self):
        pass

    def to_string(self):
        pass

    def is_fitted(self):
        return self.fitted

    @property
    def n_classes(self):
        """Number of classes of training dataset."""
        return self.n_labels

    def transform(self, x):
        return x

    def transform_all(self, x):
        return x

    def _to_torch_classifier(self):
        return SecmlLayer(self.clf)


class SVMClassifier(SecmlClassifier):
    __class_type = "svm"

    def __init__(self, clf=None, k=None):
        super().__init__(clf)
        if clf is not None:
            self.C = clf.C
            self.kernel = clf.kernel
        else:
            # init with params
            self.kernel = k

    def init_fit(self, ds, parameters):
        svm = CClassifierSVM(C=parameters["C"], kernel=self.kernel)
        self.clf = svm
        self.fit(ds)
        self.fitted = True
        self.C = parameters["C"]

    def loss(self, x, labels):
        return self.ce_loss(x, labels)

    def deepcopy(self):
        return SVMClassifier(self.clf.deepcopy())

    def to_string(self):
        return "svm"


class LogisticClassifier(SecmlClassifier):
    __class_type = "logistic"

    def __init__(self, clf=None):
        super().__init__(clf)

    def init_fit(self, ds, parameters):
        svm = CClassifierLogistic(C=parameters["C"])
        self.clf = svm
        self.fit(ds)
        self.fitted = True

    def deepcopy(self):
        return LogisticClassifier(self.clf.deepcopy())

    def to_string(self):
        return "logistic"

    def loss(self, x, labels):
        return self.ce_loss(x, labels)


class MlpClassifier(SecmlClassifier):
    __class_type = "mlp"

    def __init__(self, clf=None, outp=2, hidden_sizes=[128, ]):
        super().__init__(clf)
        self.outp = outp
        self.hidden_sizes = hidden_sizes

    def init_fit(self, ds, parameters):
        in_shape = ds.X.shape
        mlp = Mlp(in_shape[1], self.outp, self.hidden_sizes)
        clf = CClassifierPyTorch(
            model=mlp,
            batch_size=64,
            optimizer=torch.optim.Adam(mlp.parameters(), lr=1e-3),
            input_shape=in_shape[1:],
            loss=nn.CrossEntropyLoss()
        )
        self.clf = clf
        self.fit(ds)
        self.fitted = True

    def deepcopy(self):
        return MlpClassifier(self.clf.deepcopy(), self.outp, self.hidden_sizes)

    def to_string(self):
        return "mlp"

    def loss(self, x, labels):
        return self.ce_loss(x, labels)


class CnnClassifier(SecmlClassifier):
    __class_type = "cnn"

    def __init__(self, numChannels, classes, clf=None,):
        super().__init__(clf)
        self.numChannels = numChannels
        self.classes = classes

    def init_fit(self, ds, parameters):
        in_shape = ds.X.shape
        cnn = ConvNN(self.numChannels, self.classes)
        self.clf = CClassifierPyTorch(
            model=cnn,
            batch_size=64,
            optimizer=torch.optim.Adam(cnn.parameters(), lr=1e-3),
            input_shape=in_shape[1:],
            loss=nn.CrossEntropyLoss()
        )
        self.fit(ds)
        self.fitted = True

    def deepcopy(self):
        return CnnClassifier(self.numChannels, self.classes, clf=self.clf.deepcopy())

    def to_string(self):
        return "cnn"

    def loss(self, x, labels):
        return self.ce_loss(x, labels)