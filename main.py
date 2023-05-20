from sklearn.model_selection import train_test_split
import numpy as np


def import_data(file_name, t):
    X = []
    Y = []
    f = open(file_name)
    for line in f.readlines():
        vec = []
        acc = ""
        for char in line:
            acc += char
            if char == " " or char == '\n' or char == 0:
                vec.append(float(acc))
                acc = ""
                continue
        X.append(np.array(vec)[:-1])
        Y.append(vec[-1])

    X = np.array(X)
    Y = np.array(Y)
    mask = (Y == 0.0)
    Y[mask] = -1

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.5, random_state=t)
    return X_train, Y_train, X_test, Y_test


def make_rules(X):
    lines = []
    for i1 in range(len(X)):
        for i2 in range(i1 + 1, len(X)):
            if np.array_equal(X[i1], X[i2]):
                continue
            if (X[i2][0] - X[i1][0]) == 0 or (X[i2][1] - X[i1][1]) == 0:
                continue
            m = (X[i2][1] - X[i1][1]) / (X[i2][0] - X[i1][0])
            b = X[i1][1] - m*X[i1][0]
            lines.append((m, b))
    return lines


def guess(rule, p):
    pred = rule[0]*p[0] + rule[1]
    if pred > p[1]:
        return 1
    else:
        return -1


def is_rule_rigth(rule, p, label):
    return 1 if guess(rule, p) != label else 0


def predict(X, h):
    mask = X[:, 1] < (h[0]*X[:, 0] + h[1])
    ret = np.zeros_like(X[:, 0]) - 1
    ret[mask] = 1
    return ret


def adaboost(X, Y, k, H):
    D = np.ones_like(Y) * (1/len(X))
    a = []
    rh = []
    for t in range(k):
        et = []
        a.append(0)
        rh.append(0)
        for h in H:
            pred = predict(X, h)
            et.append(np.dot(D, np.not_equal(pred, Y)))

        rh[t] = np.argmin(et)
        a[t] = 0.5 * np.log((1-et[rh[t]])/et[rh[t]])

        pred = predict(X, H[rh[t]])
        new_weights = D * np.exp(-a[t]*pred*Y)
        Zt = np.sum(new_weights)
        D = new_weights / Zt
    return a, rh


def ada_error(X, Y, H, a, h, k):
    F = np.zeros_like(Y)
    for i in range(k):
        F += predict(X, H[h[i]])*a[i]
    Hk = np.sign(F)
    err = np.sum(np.not_equal(Y, Hk))
    return err / len(Y)


if __name__ == "__main__":
    ems = np.zeros(8)
    es = np.zeros(8)
    n_iters = 50
    for i in range(n_iters):
        print(i)
        Ks = []
        X, Y, Xt, Yt = import_data("squares.txt", 2*i)
        H = make_rules(X)
        a, h = adaboost(X, Y, 8, H)
        for k in range(8):
            em = ada_error(X, Y, H, a, h, k + 1)
            e = ada_error(Xt, Yt, H, a, h, k + 1)

            ems[k] += em
            es[k] += e

    ems /= n_iters
    es /= n_iters

    for i in range(8):
        print("empirical error mean of H", i, " is - ", ems[i])
        print("error mean of H", i, " is - ", es[i])
