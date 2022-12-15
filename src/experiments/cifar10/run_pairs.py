import sys

sys.path.extend(["./"])

from data.cifar_loader import load_data
from src.experiments.run_attack import *
from src.classifier.secml_classifier import CnnClassifier, SVMClassifier, LogisticClassifier, MlpClassifier
from src.optimizer.beta_optimizer import beta_poison, to_scaled_img
from src.optimizer.flip_poisoning import flip_batch_poison
from src.optimizer.white_poisoning import white_poison
import os
from src.optimizer.kde import *

if __name__ == "__main__":
    set_seed(444)
    d1, d2 = int(opts.ds[0]), int(opts.ds[2])
    classes = (d1, d2)
    tr, val, ts = load_data(labels=classes, n_tr=300, n_val=500, n_ts=1000)

    if opts.classifier == "logistic":
        clf = LogisticClassifier()
    elif opts.classifier == "mlp1":
        clf = MlpClassifier(outp=2, hidden_sizes=[256, ])
    elif opts.classifier == "mlp2":
        clf = MlpClassifier(outp=2, hidden_sizes=[256, 128])
    elif opts.classifier == "mlp3":
        clf = MlpClassifier(outp=2, hidden_sizes=[256, 128, 64])
    elif opts.classifier == "cnn":
        clf = CnnClassifier(3, 2)
    elif opts.classifier == "svm-linear":
        clf = SVMClassifier(k="linear")
    else:
        raise Exception("Classifier not found")

    h = float(opts.h)
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
        raise Exception("Kernel not found")

    params = {
        "n_proto": opts.n_proto,
        "lb": 1,
        "y_target": None,
        "y_poison": None,
        "transform": to_scaled_img,
    }
    path = opts.path + "/cifar-{}-tr{}/{}/".format(
        opts.ds, tr.X.shape[0], opts.classifier
    )
    os.makedirs(path, exist_ok=True)

    if "beta" in opts.generator:
        name = path + "beta_poison_k" + str(opts.n_proto)
        run_attack(beta_poison, name, clf, tr, val, ts, h, params=params, kernel=kernel)
    if "white" in opts.generator:
        name = path + "white_poison"
        run_attack(white_poison, name, clf, tr, val, ts, params=params)
    if "flip" in opts.generator:
        name = path + "flip"
        run_attack(flip_batch_poison, name, clf, tr, val, ts, h, params=params)