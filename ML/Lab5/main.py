import scipy.special
import sklearn.datasets
import numpy

def load_iris():
    return sklearn.datasets.load_iris()["data"].T, sklearn.datasets.load_iris()["target"]

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def split_db_2to1(D, L, seed = 0):

    nTrain = int(D.shape[1] * 2.0 / 3.0)
    numpy.random.seed(seed)

    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0: nTrain]
    idxTest =idx[nTrain:]

    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]

    return (DTR, LTR), (DVAL, LVAL)

def compute_mu_C(D):
    mu = vcol(D.mean(1))
    C = ((D - mu) @ (D - mu).T) / float(D.shape[1])
    return mu, C

def logpdf_GAU_ND(x, mu, C):
    P = numpy.linalg.inv(C)
    return -0.5*x.shape[0]*numpy.log(numpy.pi*2) - 0.5*numpy.linalg.slogdet(C)[1] - 0.5*((x - mu) * (P @ (x - mu))).sum(0)

def compute_log_likelihood_Gau(D, hParams):
    S = numpy.zeros((len(hParams), D.shape[1]))

    for lab in range(S.shape[0]):
        S[lab, :] = logpdf_GAU_ND(D, hParams[lab][0], hParams[lab][1])

    return S

def compute_logPosterior(S_logLikelihood, v_prior):
    SJoint = S_logLikelihood + vcol(numpy.log(v_prior))
    SMarginal = vrow(scipy.special.logsumexp(SJoint, axis = 0))
    SPost = SJoint - SMarginal

    return SPost

def GAU_MVG_ML_estimates(D, L):
    labelSet = set(L)
    hParams = {}

    for lab in labelSet:
        DX = D[:, L == lab]
        hParams[lab] = compute_mu_C(DX)

    return hParams

def GAU_NAIVE_ML_estimates(D, L):
    labelSet = set(L)
    hParams = {}

    for lab in labelSet:
        DX = D[:, L == lab]
        mu, C = compute_mu_C(DX)
        hParams[lab] = (mu, C * numpy.eye(D.shape[0]))

    return hParams

def GAU_TIED_ML_estimates(D, L):
    labelSet = set(L)
    hParams = {}
    hMeans = {}
    CGlobal = 0

    for lab in labelSet:
        DX = D[:, L == lab]
        mu, C_class = compute_mu_C(DX)
        CGlobal += C_class * DX.shape[1]
        hMeans[lab] = mu

    CGlobal = CGlobal / D.shape[1]
    for lab in labelSet:
        hParams[lab] = (hMeans[lab], CGlobal)

    return hParams

if __name__ == "__main__":
    D, L = load_iris()

    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    hParams_MVG = GAU_MVG_ML_estimates(DTR, LTR)
    for lab in [0, 1, 2]:
        print ("MVG - Class", lab)
        print((hParams_MVG[lab][0]))
        print(hParams_MVG[lab][1])
        print()

    S_logLikelihood = compute_log_likelihood_Gau(DVAL, hParams_MVG)
    S_logPost = compute_logPosterior(S_logLikelihood, numpy.ones(3)/3)

    PVAL = S_logPost.argmax(0)
    print("MVG Error Rate: %.1f" % ((PVAL != LVAL).sum() / float(LVAL.size) * 100))
    print()


    hParams_NAIVE = GAU_NAIVE_ML_estimates(DTR, LTR)
    for lab in [0, 1, 2]:
        print("NAIVE BAYES GAUSSIAN - Class", lab)
        print(hParams_NAIVE[lab][0])
        print(hParams_NAIVE[lab][1])
        print()

    S_logLikelihood = compute_log_likelihood_Gau(DVAL, hParams_NAIVE)
    S_logPost = compute_logPosterior(S_logLikelihood, numpy.ones(3)/3)
    PVAL = S_logPost.argmax(0)
    print("NAIVE BAYES GAUSSIAN Error Rate: %.1f" % ((PVAL != LVAL).sum() / float(LVAL.size) * 100))
    print()


    hParams_TIED = GAU_TIED_ML_estimates(DTR, LTR)
    for lab in [0, 1, 2]:
        print("TIED GAUSSIAN - Class", lab)
        print(hParams_TIED[lab][0])
        print(hParams_TIED[lab][1])
        print()

    S_logLikelihood = compute_log_likelihood_Gau(DVAL, hParams_TIED)
    S_logPost = compute_logPosterior(S_logLikelihood, numpy.ones(3)/3)
    PVAL = S_logPost.argmax(0)
    print("TIED GAUSSIAN Error Rate: %.1f" % ((PVAL != LVAL).sum() / float(LVAL.size) * 100))
    print()

    print("Binary Task")
    DBinary = D[:, L != 0]
    LBinary = L[L != 0]
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(DBinary, LBinary)

    hParams_MVG = GAU_MVG_ML_estimates(DTR, LTR)
    LLR = logpdf_GAU_ND(DVAL, hParams_MVG[2][0], hParams_MVG[2][1]) - logpdf_GAU_ND(DVAL, hParams_MVG[1][0], hParams_MVG[1][1])

    PVAL = numpy.zeros(DVAL.shape[1], dype = numpy.int32)
    TH = 0
    PVAL[LLR >= TH] = 2
    PVAL[LLR < TH] = 1
    print ("MVG Error rate: %.1f" % ((PVAL != LVAL).sum() / float(LVAL.size) * 100))
    print()

    hParams_Tied = GAU_TIED_ML_estimates(DTR, LTR)
    LLR = logpdf_GAU_ND(DVAL, hParams_Tied[2][0], hParams_Tied[2][1]) - logpdf_GAU_ND(DVAL, hParams_Tied[1][0], hParams_Tied[1][1])
    PVAL = numpy.zeros(DVAL.shape[1], dtype=numpy.int32)
    TH = 0
    PVAL[LLR >= TH] = 2
    PVAL[LLR < TH] = 1
    print("Tied - Error rate: %.1f%%" % ((PVAL != LVAL).sum() / float(LVAL.size) * 100))