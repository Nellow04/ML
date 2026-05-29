import numpy
from main import split_db_2to1,vrow, vcol, compute_mu_C, logpdf_GAU_ND, GAU_MVG_ML_estimates, GAU_TIED_ML_estimates, \
    GAU_NAIVE_ML_estimates


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

def compute_predictions_from_LLR(LLR, TH):
    PVAL = numpy.zeros(LLR.shape, dtype = numpy.int32)
    PVAL[LLR >= TH] = 1
    PVAL[LLR < TH] = 0

    return PVAL

def compute_error_rate(PVAL, LVAL):
    return ((PVAL != LVAL).sum() / float(LVAL.size) * 100)

def compute_correlation_matrix(C):
     return C / (vcol(C.diagonal() ** 0.5) * vrow(C.diagonal() ** 0.5))

def compute_mu(D):
    mu = vcol(D.mean(1))
    C = ((D - mu) @ (D - mu).T) / float(D.shape[1])
    return mu, C

def compute_pca(D, m):
    mu, C = compute_mu(D)
    U, s, Vh = numpy.linalg.svd(C)
    P = U[:, 0:m]
    return P

def apply_pca(P, D):
    return P.T @ D


if __name__ == "__main__":

    D, L = load_data("Data/trainData.txt")
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    TH = 0

    print ("Valutazione modelli generativi di base")

    # MVG
    hParams_MVG = GAU_MVG_ML_estimates(DTR, LTR)
    LLR_MVG = logpdf_GAU_ND(DVAL, hParams_MVG[1][0], hParams_MVG[1][1]) - logpdf_GAU_ND(DVAL, hParams_MVG[0][0], hParams_MVG[0][1])
    PVAL_MVG = compute_predictions_from_LLR(LLR_MVG, TH)
    err_MVG = compute_error_rate(PVAL_MVG, LVAL)
    print(("MVG erro rate: %.1f" % err_MVG))
    print()

    # TIED
    hParams_TIED = GAU_TIED_ML_estimates(DTR, LTR)
    LLR_TIED = logpdf_GAU_ND(DVAL, hParams_TIED[1][0], hParams_TIED[1][1]) - logpdf_GAU_ND(DVAL, hParams_TIED[0][0], hParams_TIED[0][1])
    PVAL_TIED = compute_predictions_from_LLR(LLR_TIED, TH)
    err_TIED = compute_error_rate(PVAL_TIED, LVAL)
    print(("TIED erro rate: %.1f" % err_TIED))
    print()

    # NB
    hParams_NB = GAU_NAIVE_ML_estimates(DTR, LTR)
    LLR_NB = logpdf_GAU_ND(DVAL, hParams_NB[1][0], hParams_NB[1][1]) - logpdf_GAU_ND(DVAL, hParams_NB[0][0], hParams_NB[0][1])
    PVAL_NB = compute_predictions_from_LLR(LLR_NB, TH)
    err_NB = compute_error_rate(PVAL_NB, LVAL)
    print(("NB erro rate: %.1f" % err_NB))
    print()

    corr_0 = compute_correlation_matrix(hParams_MVG[0][1])
    print("Matrice di correlazione - classe 0")
    print(numpy.round(corr_0, 2))

    corr_1 = compute_correlation_matrix(hParams_MVG[1][1])
    print("Matrice di correlazione - classe 1")
    print(numpy.round(corr_1, 2))
    print()

    print("Rimozione delle feature 5 e 6")

    DTR_4f = DTR[0:4, :]
    DVAL_4f = DVAL[0:4, :]

    # MVG
    hParams_MVG_4f = GAU_MVG_ML_estimates(DTR_4f, LTR)
    LLR_MVG_4f = logpdf_GAU_ND(DVAL_4f, hParams_MVG_4f[1][0], hParams_MVG_4f[1][1]) - logpdf_GAU_ND(DVAL_4f, hParams_MVG_4f[0][0], hParams_MVG_4f[0][1])
    PVAL_MVG_4f = compute_predictions_from_LLR(LLR_MVG_4f, TH)
    err_MVG_4f = compute_error_rate(PVAL_MVG_4f, LVAL)
    print("MVG (1-4) error rate:  %.1f" % err_MVG_4f)

    # TIED
    hParams_TIED_4f = GAU_TIED_ML_estimates(DTR_4f, LTR)
    LLR_TIED_4f = logpdf_GAU_ND(DVAL_4f, hParams_TIED_4f[1][0], hParams_TIED_4f[1][1]) - logpdf_GAU_ND(DVAL_4f, hParams_TIED_4f[0][0], hParams_TIED_4f[0][1])
    PVAL_TIED_4f = compute_predictions_from_LLR(LLR_TIED_4f, TH)
    err_TIED_4f = compute_error_rate(PVAL_TIED_4f, LVAL)
    print("TIED (1-4) error rate: %.1f" % err_TIED_4f)

    # NB
    hParams_NB_4f = GAU_NAIVE_ML_estimates(DTR_4f, LTR)
    LLR_NB_4f = logpdf_GAU_ND(DVAL_4f, hParams_NB_4f[1][0], hParams_NB_4f[1][1]) - logpdf_GAU_ND(DVAL_4f, hParams_NB_4f[0][0],hParams_NB_4f[0][1])
    PVAL_NB_4f = compute_predictions_from_LLR(LLR_NB_4f, TH)
    err_NB_4f = compute_error_rate(PVAL_NB_4f, LVAL)
    print("NB (1-4) error rate:   %.1f" % err_NB_4f)
    print()


    print("Analisi Coppie di Feature (1-2 e 3-4) ")

    DTR_12 = DTR[0:2, :]
    DVAL_12 = DVAL[0:2, :]

    hParams_MVG_12 = GAU_MVG_ML_estimates(DTR_12, LTR)
    LLR_MVG_12 = logpdf_GAU_ND(DVAL_12, hParams_MVG_12[1][0], hParams_MVG_12[1][1]) - logpdf_GAU_ND(DVAL_12, hParams_MVG_12[0][0], hParams_MVG_12[0][1])
    err_MVG_12 = compute_error_rate(compute_predictions_from_LLR(LLR_MVG_12, TH), LVAL)

    hParams_TIED_12 = GAU_TIED_ML_estimates(DTR_12, LTR)
    LLR_TIED_12 = logpdf_GAU_ND(DVAL_12, hParams_TIED_12[1][0], hParams_TIED_12[1][1]) - logpdf_GAU_ND(DVAL_12, hParams_TIED_12[0][0], hParams_TIED_12[0][1])
    err_TIED_12 = compute_error_rate(compute_predictions_from_LLR(LLR_TIED_12, TH), LVAL)

    print("Feature 1-2 - MVG error:  %.1f%%" % err_MVG_12)
    print("Feature 1-2 - TIED error: %.1f%%" % err_TIED_12)
    print()

    DTR_34 = DTR[2:4, :]
    DVAL_34 = DVAL[2:4, :]

    hParams_MVG_34 = GAU_MVG_ML_estimates(DTR_34, LTR)
    LLR_MVG_34 = logpdf_GAU_ND(DVAL_34, hParams_MVG_34[1][0], hParams_MVG_34[1][1]) - logpdf_GAU_ND(DVAL_34, hParams_MVG_34[0][0], hParams_MVG_34[0][1])
    err_MVG_34 = compute_error_rate(compute_predictions_from_LLR(LLR_MVG_34, TH), LVAL)

    hParams_TIED_34 = GAU_TIED_ML_estimates(DTR_34, LTR)
    LLR_TIED_34 = logpdf_GAU_ND(DVAL_34, hParams_TIED_34[1][0], hParams_TIED_34[1][1]) - logpdf_GAU_ND(DVAL_34, hParams_TIED_34[0][0], hParams_TIED_34[0][1])
    err_TIED_34 = compute_error_rate(compute_predictions_from_LLR(LLR_TIED_34, TH), LVAL)

    print("Feature 3-4 - MVG error:  %.1f%%" % err_MVG_34)
    print("Feature 3-4 - TIED error: %.1f%%" % err_TIED_34)
    print()

    print ("PCA")

    for m in range(1, 6):
        print(f"--- PCA con m = {m} ---")

        P_matrix = compute_pca(DTR, m)

        DTR_pca = apply_pca(P_matrix, DTR)
        DVAL_pca = apply_pca(P_matrix, DVAL)

        # MVG + PCA
        hParams_MVG_pca = GAU_MVG_ML_estimates(DTR_pca, LTR)
        LLR_MVG_pca = logpdf_GAU_ND(DVAL_pca, hParams_MVG_pca[1][0], hParams_MVG_pca[1][1]) - logpdf_GAU_ND(DVAL_pca, hParams_MVG_pca[0][0], hParams_MVG_pca[0][1])
        err_MVG_pca = compute_error_rate(compute_predictions_from_LLR(LLR_MVG_pca, TH), LVAL)

        # TIED + PCA
        hParams_TIED_pca = GAU_TIED_ML_estimates(DTR_pca, LTR)
        LLR_TIED_pca = logpdf_GAU_ND(DVAL_pca, hParams_TIED_pca[1][0], hParams_TIED_pca[1][1]) - logpdf_GAU_ND(DVAL_pca, hParams_TIED_pca[0][0], hParams_TIED_pca[0][1])
        err_TIED_pca = compute_error_rate(compute_predictions_from_LLR(LLR_TIED_pca, TH), LVAL)

        # NAIVE BAYES + PCA
        hParams_NB_pca = GAU_NAIVE_ML_estimates(DTR_pca, LTR)
        LLR_NB_pca = logpdf_GAU_ND(DVAL_pca, hParams_NB_pca[1][0], hParams_NB_pca[1][1]) - logpdf_GAU_ND(DVAL_pca, hParams_NB_pca[0][0], hParams_NB_pca[0][1])
        err_NB_pca = compute_error_rate(compute_predictions_from_LLR(LLR_NB_pca, TH), LVAL)

        print("MVG Error rate:         %.1f%%" % err_MVG_pca)
        print("Tied Error rate:        %.1f%%" % err_TIED_pca)
        print("Naive Bayes Error rate: %.1f%%" % err_NB_pca)
        print()

# Domande
# 1)
# MVG è il modello che performa meglio (7.0%). Ha la libertà di adattare forma, dimensione e orientamento della campana per ogni classe
# Naive Bayes è incredibilmente vicino all'MVG (7.2%) nonostante lo abbiamo costretto ad ignorare le correlazioni tra le feature
# (azzerando tutto fuori dalla diagonale). Non peggiorando di molto, implica che le feature sono naturalmente molto indipendenti
# Tied è il peggiore (9.3%), forza le due classi ad avere esattamente la stessa matrice di covarianza

# 2)
# I valori delle feature sono molto vicini allo 0.0, quindi le feature sono debolmente correlate
# Il Naive Bayes assume che la correlazione sia zero. Essendo la correlazione reale dei dati vicina allo zero, l'approssimazione del Naive Bayes è molto buona

# 3)
# Anche se le feature 5 e 6 avevano distribuzioni non ottime e non perfettamente a campana, contenevano comunque dettagli preziosi per la distinzione delle classi.
# Scartare queste informazioni si è rivelato controproducente, essendo il tasso di errore calato da 7.0% a 8.0%

# 4)
# Tra le feature 1 e 2, l'MVG è quello che performa meglio 36.5%. Questo accade perchè le feature hanno varianze molto diverse tra le due classi.
# Il Tied invece, costringendo le varianze ad essere uguali, compie una forzatura abbassando le performance
# Tra le feature 3 e 4, il Tied ha le stesse performance dell'MVG (9.4%). Significa che le feature 3 e 4 hanno varianze quasi identiche.
# Assumerle uguali diventa una mossa vincente che semplifica il modlelo senza perdere precisione

# 5)
# La PCA non ha portato miglioramenti. Infatti il miglior risultato si ha con m =5 (7.1%), che non riesce comunque a battere
# il punteggio di (7.0%) del modello a 6 dimensioni. Questo significa che le 6 feature originali sono già molto compatte e non contengono rumore
# Con la PCA Naive Bayes peggiora (9,2%). Ciò accade perchè la PCA ruota lo spazio combinando le feature lineamente, creando artificialmente delle correlazioni.
# Il Naive Bayes, ignorando queste nuove correlazioni, perde colpi

#Concludendo, sul nostro dataset, il modello migliore risulta l'MVG (7.0%) allenato su ttute e 6 le features originali




