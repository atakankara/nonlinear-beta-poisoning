import sys

sys.path.extend(["./"])

from data.cifar_loader import load_data
from src.experiments.run_attack import *
from src.classifier.secml_classifier import SVMClassifier, LogisticClassifier, MlpClassifier
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
    else:
        clf = SVMClassifier(k="linear")
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
        run_attack(beta_poison, name, clf, tr, val, ts, 1, params=params, kernel=KDEGaussian(val, clf, 1))
    if "white" in opts.generator:
        name = path + "white_poison"
        run_attack(white_poison, name, clf, tr, val, ts, params=params)
    if "flip" in opts.generator:
        name = path + "flip"
        run_attack(flip_batch_poison, name, clf, tr, val, ts, params=params)
