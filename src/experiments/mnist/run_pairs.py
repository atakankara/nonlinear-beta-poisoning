import sys

sys.path.extend(["./"])

from data.mnist_loader import *
from src.experiments.run_attack import *
from src.classifier.secml_classifier import SVMClassifier, LogisticClassifier, MlpClassifier, CnnClassifier
from src.optimizer.beta_optimizer import beta_poison, to_scaled_img
from src.optimizer.flip_poisoning import flip_batch_poison
from src.optimizer.white_poisoning import white_poison
import os
from src.optimizer.kde import *

if __name__ == "__main__":
    set_seed(444)
    d1, d2 = int(opts.ds[0]), int(opts.ds[2])
    digits = (d1, d2)
    tr, val, ts = load_mnist(digits=digits, n_tr=400, n_val=1000, n_ts=1000)

    if opts.classifier == "logistic":
        clf = LogisticClassifier()
    elif opts.classifier == "mlp1":
        clf = MlpClassifier(outp=2, hidden_sizes=[256, ])
    elif opts.classifier == "mlp2":
        clf = MlpClassifier(outp=2, hidden_sizes=[256, 128])
    elif opts.classifier == "mlp3":
        clf = MlpClassifier(outp=2, hidden_sizes=[256, 128, 64])
    elif opts.classifier == "cnn":
        clf = CnnClassifier(1, 2) #TODO:enter parameters
    elif opts.classifier == "svm-linear":
        clf = SVMClassifier(k="linear")
    elif opts.classifier == "svm-rbf":
        from secml.ml import CKernelRBF
        clf = SVMClassifier(k=CKernelRBF(gamma=float(opts.gamma)))
    elif opts.classifier == "svm-poly":
        clf = SVMClassifier(k="poly")
    else :
        clf = SVMClassifier(k="linear")


    for h in [1]:
        if opts.kernel == "gaussian":
            kernel = KDEGaussian(val, clf, h)
        elif opts.kernel == "tophat":
            kernel = KDETophat(val, clf, h)
        elif opts.kernel == "epanechnikov":
            kernel = KDEEpanechnikov(val, clf, h)
        elif opts.kernel == "gaussian2":
            kernel = KDEGaussian2(val, clf, h)
        elif opts.kernel == "logistic":
            kernel = KDELogistic(val, clf, h)
        elif opts.kernel == "sigmoid":
            kernel = KDESigmoid(val, clf, h)
        elif opts.kernel == "mlp_kernel":
            kernel = KDEMlp(val, clf, h)
        else:
            raise Exception()

        params = {
            "n_proto": opts.n_proto,
            "lb": 1,
            "y_target": None,
            "y_poison": None,
            "transform": to_scaled_img,
            # "gamma": opts.gamma
        }
        path = opts.path + "/mnist-{}-tr{}/{}/".format(
            opts.ds, tr.X.shape[0], opts.classifier
        )
        os.makedirs(path, exist_ok=True)

        if "beta" in opts.generator:
            name = path + "beta_poison_k" + str(opts.n_proto)
            run_attack(beta_poison, name, clf, tr, val, ts, h, params=params, kernel=kernel)
        if "white" in opts.generator:
            name = path + "white_poison"
            run_attack(white_poison, name, clf, tr, val, ts, h, params=params)
        if "flip" in opts.generator:
            name = path + "flip"
            run_attack(flip_batch_poison, name, clf, tr, val, ts, h, params=params)
