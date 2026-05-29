from bdb import effective

import numpy
from matplotlib import pyplot as plt

from main import compute_empirical_Bayes_risk_binary_llr_optimal_decisions
from main import compute_minDCF_binary_fast


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

def load_data(file_name):

    dataset_list = []
    label_list = []

    with open(file_name) as file:
        for line in file:
            try:
                features = line.split(",")[0:-1]
                features = vcol(numpy.array([float(i) for i in features]))
                value = line.split(",")[-1].strip()
                label = int(value)

                dataset_list.append(features)
                label_list.append(label)
            except:
                pass

    return numpy.hstack(dataset_list), numpy.array(label_list, dtype=numpy.int32)

if __name__ == "__main__":

    D, L = load_data("Project/trainData.txt")
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    hParams_MVG = GAU_MVG_ML_estimates(DTR, LTR)
    LLR_MVG = logpdf_GAU_ND(DVAL, hParams_MVG[1][0], hParams_MVG[1][1]) - logpdf_GAU_ND(DVAL, hParams_MVG[0][0], hParams_MVG[0][1])

    hParams_TIED = GAU_TIED_ML_estimates(DTR, LTR)
    LLR_TIED = logpdf_GAU_ND(DVAL, hParams_TIED[1][0], hParams_TIED[1][1]) - logpdf_GAU_ND(DVAL, hParams_TIED[0][0], hParams_TIED[0][1])

    hParams_NB = GAU_NAIVE_ML_estimates(DTR, LTR)
    LLR_NB = logpdf_GAU_ND(DVAL, hParams_NB[1][0], hParams_NB[1][1]) - logpdf_GAU_ND(DVAL, hParams_NB[0][0], hParams_NB[0][1])

    effective_priors = [0.1, 0.5, 0.9]

    for prior in effective_priors:
        print(f"Effective prior: {prior}")

        act_MVG = compute_empirical_Bayes_risk_binary_llr_optimal_decisions(LLR_MVG, LVAL, prior, 1.0, 1.0)
        min_MVG = compute_minDCF_binary_fast(LLR_MVG, LVAL, prior, 1.0, 1.0)
        print(f"MVG - Actual DCF: {act_MVG:.3f} | Min DCF: {min_MVG:.3f}")

        act_TIED = compute_empirical_Bayes_risk_binary_llr_optimal_decisions(LLR_TIED, LVAL, prior, 1.0, 1.0)
        min_TIED = compute_minDCF_binary_fast(LLR_TIED, LVAL, prior, 1.0, 1.0)
        print(f"TIED - Actual DCF: {act_TIED:.3f} | Min DCF: {min_TIED:.3f}")

        act_NB = compute_empirical_Bayes_risk_binary_llr_optimal_decisions(LLR_NB, LVAL, prior, 1.0, 1.0)
        min_NB = compute_minDCF_binary_fast(LLR_NB, LVAL, prior, 1.0, 1.0)
        print(f"NAIVE - Actual DCF: {act_NB:.3f}   | Min DCF: {min_NB:.3f}")

        effPriorLogOdds = numpy.linspace(-4, 4, 30)
        effPriors = 1.0 / (1.0 + numpy.exp(-effPriorLogOdds))

        actDCF_MVG, minDCF_MVG = [], []
        actDCF_TIED, minDCF_TIED = [], []
        actDCF_NB, minDCF_NB = [], []

        for p in effPriors:
            actDCF_MVG.append(compute_empirical_Bayes_risk_binary_llr_optimal_decisions(LLR_MVG, LVAL, p, 1.0, 1.0))
            minDCF_MVG.append(compute_minDCF_binary_fast(LLR_MVG, LVAL, p, 1.0, 1.0))

            actDCF_TIED.append(compute_empirical_Bayes_risk_binary_llr_optimal_decisions(LLR_TIED, LVAL, p, 1.0, 1.0))
            minDCF_TIED.append(compute_minDCF_binary_fast(LLR_TIED, LVAL, p, 1.0, 1.0))

            actDCF_NB.append(compute_empirical_Bayes_risk_binary_llr_optimal_decisions(LLR_NB, LVAL, p, 1.0, 1.0))
            minDCF_NB.append(compute_minDCF_binary_fast(LLR_NB, LVAL, p, 1.0, 1.0))

        # Grafico MVG
        plt.figure("Bayes Error Plot - MVG")
        plt.plot(effPriorLogOdds, actDCF_MVG, label='Actual DCF', color='r', linewidth=2)
        plt.plot(effPriorLogOdds, minDCF_MVG, label='Min DCF', color='b', linewidth=2, linestyle='dashed')
        plt.ylim([0, 1.1])
        plt.title("MVG: Bayes Error Plot")
        plt.xlabel("Prior Log-Odds")
        plt.ylabel("DCF")
        plt.legend()
        plt.show()

        # Grafico TIED
        plt.figure("Bayes Error Plot - TIED")
        plt.plot(effPriorLogOdds, actDCF_TIED, label='Actual DCF', color='r', linewidth=2)
        plt.plot(effPriorLogOdds, minDCF_TIED, label='Min DCF', color='b', linewidth=2, linestyle='dashed')
        plt.ylim([0, 1.1])
        plt.title("TIED: Bayes Error Plot")
        plt.xlabel("Prior Log-Odds")
        plt.ylabel("DCF")
        plt.legend()
        plt.show()

        # Grafico NB
        plt.figure("Bayes Error Plot - NB")
        plt.plot(effPriorLogOdds, actDCF_NB, label='Actual DCF', color='r', linewidth=2)
        plt.plot(effPriorLogOdds, minDCF_NB, label='Min DCF', color='b', linewidth=2, linestyle='dashed')
        plt.ylim([0, 1.1])
        plt.title("NB: Bayes Error Plot")
        plt.xlabel("Prior Log-Odds")
        plt.ylabel("DCF")
        plt.legend()
        plt.show()

# Domande
# 1)
# Guardando il Minimum DCF, i modelli che performano meglio sono l'MVG e il Naive Bayes. I due modelli ottenongo risultati quasi identici i tutti gli scenari.
# Il modlelo TIED è nettamente peggiore in ogni situazione

# I risultati sono coerenti, la classifica non cambia mai al variare dell'applicazione. MVG e Naive Bayes si contendono il primo posto,
# TIED resta fisso all'ultimo posto in tutti gli scenari

# 2)
# Nel caso bilanciato (0.5), i modelli sono bel calibrati (0.010). Se ci spostiamo nei casi estemi, la calibrazione peggiora.
# L'MVG a priori 0.9 ha un gap di 0.058, che rappresenta un bel peggioramennto rispetto al minimo teorico

# 3)
# Il modlelo TIED è il meglio calibrato di tutti, anche se fa più errori in assoluto. A 0.5 la sua calibration loss è di 0.005
# Al contrario, l'MVG, pur essendo più potente, soffre di più di miscalibrazione quando i prior diventano estremi

# 4)
# Grafici

# Conclusione
# L'MVG, insieme al Naive Bayes, è il miglior estrattore di informazioni per il nostro dataset.
# Tuttavia, non possiamo fidarci ciecamente delle forule matematiche di bayes nel caso di una prior 0.9 o 0.1. ma dovremmo applicare una calibrazione

