import numpy
import scipy
import sklearn
from Solution import bayesRisk


def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def split_db_2to1(D, L, seed = 0):
    nTrain = int(D.shape[1] * 2.0 / 3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0: nTrain]
    idxTest = idx[nTrain :]

    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]

    return (DTR, LTR), (DVAL, LVAL)

def load_iris_binary():
    D, L = sklearn.datasets.load_iris()["data"].T, sklearn.datasets.load_iris()["target"]
    D = D[:, L != 0]
    L = L[L != 0]
    L[ L == 2] = 0

    return D, L

def trainLogRegBinary(DTR, LTR, l):
    ZTR = LTR * 2.0 - 1.0

    def logReg_obj_with_grad(v):
        w = v[: -1]
        b = v[-1]
        s = numpy.dot(vcol(w).T, DTR).ravel() + b

        loss = numpy.logaddexp(0, -ZTR * s)

        G = -ZTR / (1.0 + numpy.exp(ZTR * s))
        GW = (vrow(G) * DTR).mean(1) + l * w.ravel()
        Gb = G.mean()

        return loss.mean() + l / 2 * numpy.linalg.norm(w)**2, numpy.hstack([GW, numpy.array(Gb)])

    vf = scipy.optimize.fmin_l_bfgs_b(logReg_obj_with_grad, x0 = numpy.zeros(DTR.shape[0] + 1))[0]

    print("Logistic Regressione - Lambda = %e - J * (w, b) = %e" % (l, logReg_obj_with_grad(vf)[0]))

    return vf[: -1], vf[-1]

def trainWeigthedLogRegBinary(DTR, LTR, l, pT):

    ZTR = LTR * 2.0 - 1.0

    wTrue = pT / (ZTR > 0).sum()
    wFalse = (1 - pT) / (ZTR < 0).sum()

    def logReg_obj_with_grad(v):
        w = v[: -1]
        b = v[-1]
        s = numpy.dot(vcol(w).T, DTR).ravel() + b

        loss = numpy.logaddexp(0, -ZTR * s)
        loss[ZTR > 0] *= wTrue
        loss[ZTR < 0] *= wFalse

        G = -ZTR / (1.0 + numpy.exp(ZTR * s))
        G[ZTR > 0] *= wTrue
        G[ZTR < 0] *= wFalse

        GW = (vrow(G) * DTR).sum(1) + l * w.ravel()
        Gb = G.sum()

        return loss.sum() + l / 2 * numpy.linalg.norm(w)**2, numpy.hstack([GW, numpy.array(Gb)])

    vf = scipy.optimize.fmin_l_bfgs_b(logReg_obj_with_grad, x0 = numpy.zeros(DTR.shape[0] + 1))[0]

    print(("Weighted Logistic Regresion (pT %e) - Lambda = %e - J * (w, b) = %e" % (pT, l, logReg_obj_with_grad(vf)[0])))

    return vf[: -1], vf[-1]

if __name__ == "__main__":

    D, L = load_iris_binary()
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    for lamb in [1e-3, 1e-1, 1.0]:

        w, b = trainLogRegBinary(DTR, LTR, lamb)
        sVal = numpy.dot(w.T, DVAL) + b
        PVAL = (sVal > 0) * 1
        err = (PVAL != LVAL).sum() / float(LVAL.size)

        print ("Error rate: %.1f" % (err * 100))

        pEmp = (LTR == 1).sum() / LTR.size
        sValLLR = sVal - numpy.log(pEmp / (1 - pEmp))

        print ("minDCF - pT = 0.5: %.4f" % bayesRisk.compute_minDCF_binary_fast(sValLLR, LVAL, 0.5, 1.0, 1.0))
        print ("actDCF - pT = 0.5: %.4f" % bayesRisk.compute_actDCF_binary_fast(sValLLR, LVAL, 0.5, 1.0, 1.0))

        pT = 0.8
        w, b = trainWeigthedLogRegBinary(DTR, LTR, lamb, pT = pT)
        sVal = numpy.dot(w.T, DVAL) + b
        sValLLR = sVal - numpy.log(pT / (1 - pT))

        print("minDCF - pT = 0.8: %.4f" % bayesRisk.compute_minDCF_binary_fast(sValLLR, LVAL, pT, 1.0, 1.0))
        print("actDCF - pT = 0.8: %.4f" % bayesRisk.compute_actDCF_binary_fast(sValLLR, LVAL, pT, 1.0, 1.0))

        print()
