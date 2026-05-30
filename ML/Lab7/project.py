import os

import matplotlib.pyplot as plt
import numpy

from Solution.bayesRisk import compute_minDCF_binary_fast, compute_actDCF_binary_fast
from main import split_db_2to1, trainLogRegBinary, vcol, trainWeigthedLogRegBinary

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

def expand_features(D):

    expanded_D = []
    for i in range (D.shape[1]):
        x = numpy.reshape(D[:, i], (D.shape[0], 1))
        xxT = numpy.dot(x, x.T)

        x_expanded = numpy.vstack([numpy.reshape(xxT, (-1, 1)), x])
        expanded_D.append(x_expanded)

    return numpy.hstack(expanded_D)

if __name__ == "__main__":
    print ("Logistic Regression Dataset Completo")

    D, L = load_data("Project/trainData.txt")
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    pT = 0.1
    pEmp = (LTR == 1).sum() / LTR.size

    lambdas = numpy.logspace(-4, 2, 13)

    '''
    minDCF_list = []
    actDCF_list = []

    for l in lambdas:
        w, b = trainLogRegBinary(DTR, LTR, l)
        sVal = numpy.dot(w.T, DVAL) + b
        sValLLR = sVal - numpy.log(pEmp / (1 - pEmp))

        min_dcf = compute_minDCF_binary_fast(sValLLR, LVAL, pT, 1.0, 1.0)
        act_dcf = compute_actDCF_binary_fast(sValLLR, LVAL, pT, 1.0, 1.0)

        minDCF_list.append(min_dcf)
        actDCF_list.append(act_dcf)

        print(f"Lambda = {l:.4e} | minDCF = {min_dcf:.4f} | actDCF = {act_dcf:.4f}")
        print()

    plt.figure("Logistic Regression - Full Dataset")
    plt.plot(lambdas, minDCF_list, label='minDCF', marker='o', color='b')
    plt.plot(lambdas, actDCF_list, label='actDCF', marker='x', color='r')
    plt.xscale('log', base=10)
    plt.title(f'Logistic Regression Standard (pi = {pT})')
    plt.xlabel('Lambda (Regolarizzazione)')
    plt.ylabel('DCF Value')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.show()
    print()


    print ("Logistic Regression - Dataset Ridotto")

    DTR_reduced = DTR[:, ::50]
    LTR_reduced = LTR[::50]

    minDCF_list_reduced = []
    actDCF_list_reduced = []

    for l in lambdas:
        w, b = trainLogRegBinary(DTR_reduced, LTR_reduced, l)
        sVal = numpy.dot(w.T, DVAL) + b
        sValLLR = sVal - numpy.log(pEmp / (1 - pEmp))

        min_dcf = compute_minDCF_binary_fast(sValLLR, LVAL, pT, 1.0, 1.0)
        act_dcf = compute_actDCF_binary_fast(sValLLR, LVAL, pT, 1.0, 1.0)

        minDCF_list_reduced.append(min_dcf)
        actDCF_list_reduced.append(act_dcf)

        print(f"Lambda = {l:.4e} | minDCF = {min_dcf:.4f} | actDCF = {act_dcf:.4f}")
        print()

    plt.figure("Logistic Regression - Dataset Ridotto")
    plt.plot(lambdas, minDCF_list_reduced, label='minDCF (Reduced)', marker='o', color='b')
    plt.plot(lambdas, actDCF_list_reduced, label='actDCF (Reduced)', marker='x', color='r')
    plt.xscale('log', base=10)
    plt.title(f'Logistic Regression Standard (pi = {pT})')
    plt.xlabel('Lambda (Regolarizzazione)')
    plt.ylabel('DCF Value')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.show()
    print()


    print ("Weigthed Logistic Regression - Dataset Completo")

    actDCF_list_w = []
    minDCF_list_w = []

    for l in lambdas:
        w, b = trainWeigthedLogRegBinary(DTR, LTR, l, pT)
        sVal = numpy.dot(w.T, DVAL) + b
        sValLLR_w = sVal - numpy.log(pT / (1 - pT))

        min_dcf = compute_minDCF_binary_fast(sValLLR_w, LVAL, pT, 1.0, 1.0)
        act_dcf = compute_actDCF_binary_fast(sValLLR_w, LVAL, pT, 1.0, 1.0)

        minDCF_list_w.append(min_dcf)
        actDCF_list_w.append(act_dcf)

        print(f"Weighted LR - Lambda = {l:.4e} | minDCF = {min_dcf:.4f} | actDCF = {act_dcf:.4f}")
        print()

    plt.figure("Weigthed Logistic Regression - Dataset Completo")
    plt.plot(lambdas, minDCF_list_w, label='minDCF', marker='o', color='b')
    plt.plot(lambdas, actDCF_list_w, label='actDCF', marker='x', color='r')
    plt.xscale('log', base=10)
    plt.title(f'Weigthed Logistic Regression Standard (pi = {pT})')
    plt.xlabel('Lambda (Regolarizzazione)')
    plt.ylabel('DCF Value')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.show()
    print()
    '''

    print("Quadratic Logistic Regression")

    DTR_quad = expand_features(DTR)
    DVAL_quad = expand_features(DVAL)

    minDCF_list_quad = []
    actDCF_list_quad = []

    for l in lambdas:
        w, b = trainLogRegBinary(DTR_quad, LTR, l)
        sVal = numpy.dot(w.T, DVAL_quad) + b
        sValLLR_quad = sVal - numpy.log(pEmp / (1 - pEmp))

        min_dcf = compute_minDCF_binary_fast(sValLLR_quad, LVAL, pT, 1.0, 1.0)
        act_dcf = compute_actDCF_binary_fast(sValLLR_quad, LVAL, pT, 1.0, 1.0)

        minDCF_list_quad.append(min_dcf)
        actDCF_list_quad.append(act_dcf)

        print(f"Quadratic LR - Lambda = {l:.4e} | minDCF = {min_dcf:.4f} | actDCF = {act_dcf:.4f}")
        print()

    plt.figure("Quadratic Logistic Regression - Dataset Completo")
    plt.plot(lambdas, minDCF_list_quad, label='minDCF', marker='o', color='b')
    plt.plot(lambdas, actDCF_list_quad, label='actDCF', marker='x', color='r')
    plt.xscale('log', base=10)
    plt.title(f'Quadratic Logistic Regression Standard (pi = {pT})')
    plt.xlabel('Lambda (Regolarizzazione)')
    plt.ylabel('DCF Value')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.show()
    print()

    if not os.path.exists("Scores"):
        os.makedirs("Scores")

    w_best, b_best = trainLogRegBinary(DTR, LTR, 1e-2)
    sVal_best = numpy.dot(w_best.T, DVAL) + b_best
    sValLLR_best = sVal_best - numpy.log(pEmp / (1 - pEmp))

    numpy.save("Scores/LLR_LogReg.npy", sValLLR_best)
   #  numpy.save("Scores/LLR_QuadLogReg.npy", sValLLR_quad_best)
    numpy.save("Scores/LVAL.npy", LVAL)

# Domande
# 1)
# Sul dataset completo, la regolarizzazione risulta inefficace per migliorare la capacità discriminativa.
# Questo accade perchè abbiamo così tanti campioni che il rischio di overfitting è praticamente nullo. AL contrario, aumentare lambda
# a valori alti è dannoso: costirnge i pesi del modello verso lo zero, distruggendo la natura probabilistica degli score e facendo schizzare l'actDCF a 1.0

# 2)
# RIducendo i dati, la regolarizzazione diventa fondamentale. I grafici mostrano una tipica curva ad U: per lambda molto piccoli,
# il modello va in overfitting sui pochi dati disponibili, memorizzandoli ma fallendo sul validation set. Man mano che aumenta lambda,
# la penalità limita la complessità dei pesi e l'errore minDCF scende fino ad un punto di minimo ottimale.
# Aumentando lambda, si cade in underfitting e l'errore torna a salire

# 3)
# Il vantaggio del modello Weigthed non sta nel minDCF, ma nella calibrazione. FOrzando il modello a pesare maggiormente
# gli errori della classe minoritaria fin dalla fase di addestramento, gli score generati sono nativamente più allineati allo scenario operativo.
# IL risultato è un actDCF molto più vicino al minDCF rispetot a quello che si ottiene traslando semplicemente la soglia dle modello standard

#4)
# Utilizzando il modello quadratico, notiamo che minDCF scende fino a 0.2436. In questo caso la regolarizzazione è leggermente efficace:
# passando da 6 a 42 feature, lo spazio delle dimensioni aumenta reintroducendo un lieve rischio di overfitting anche con il dataset completo.
# Un valore intermedio di lambda aiuta a stabilizzare i pesi e atoccare il minimo assoluto dell'errore

#5)
# Il vincitore assoluto è la Quadratic Logistic Regression con un minDCF = 0.2436

