############################################################################################
# Copyright (C) 2024 by Sandro Cumani                                                      #
#                                                                                          #
# This file is provided for didactic purposes only, according to the Politecnico di Torino #
# policies on didactic material.                                                           #
#                                                                                          #
# Any form of re-distribution or online publication is forbidden.                          #
#                                                                                          #
# This file is provided as-is, without any warranty                                        #
############################################################################################
from pyexpat import features

import pca
import lda
import numpy

from lda import vcol, vrow, load_iris


def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1] * 2.0 / 3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]

    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]

    return (DTR, LTR), (DVAL, LVAL)


if __name__ == '__main__':

    DIris, LIris = load_iris()
    D = DIris[:, LIris != 0]
    L = LIris[LIris != 0]
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    # Solution without PCA pre-processing and threshold selection. The threshold is chosen half-way between the two classes
    ULDA = lda.compute_lda_geig(DTR, LTR, m=1)

    DTR_lda = lda.apply_lda(ULDA, DTR)

    # Check if the Virginica class samples are, on average, on the right of the Versicolor samples on the training set. If not, we reverse ULDA and re-apply the transformation.
    if DTR_lda[0, LTR == 1].mean() > DTR_lda[0, LTR == 2].mean():
        ULDA = -ULDA
        DTR_lda = lda.apply_lda(ULDA, DTR)

    DVAL_lda = lda.apply_lda(ULDA, DVAL)

    threshold = (DTR_lda[0, LTR == 1].mean() + DTR_lda[
        0, LTR == 2].mean()) / 2.0  # Estimated only on model training data

    PVAL = numpy.zeros(shape=LVAL.shape, dtype=numpy.int32)
    PVAL[DVAL_lda[0] >= threshold] = 2
    PVAL[DVAL_lda[0] < threshold] = 1
    print('Labels:     ', LVAL)
    print('Predictions:', PVAL)
    print('Number of erros:', (PVAL != LVAL).sum(), '(out of %d samples)' % (LVAL.size))
    print('Error rate: %.1f%%' % ((PVAL != LVAL).sum() / float(LVAL.size) * 100))

    # Solution with PCA pre-processing with dimension m.
    m = 2
    UPCA = pca.compute_pca(DTR, m=m)  # Estimated only on model training data
    DTR_pca = pca.apply_pca(UPCA, DTR)  # Applied to original model training data
    DVAL_pca = pca.apply_pca(UPCA, DVAL)  # Applied to original validation data

    ULDA = lda.compute_lda_geig(DTR_pca, LTR,
                                     m=1)  # Estimated only on model training data, after PCA has been applied

    DTR_lda = lda.apply_lda(ULDA,
                            DTR_pca)  # Applied to PCA-transformed model training data, the projected training samples are required to check the orientation of the direction and to compute the threshold
    # Check if the Virginica class samples are, on average, on the right of the Versicolor samples on the training set. If not, we reverse ULDA and re-apply the transformation
    if DTR_lda[0, LTR == 1].mean() > DTR_lda[0, LTR == 2].mean():
        ULDA = -ULDA
        DTR_lda = lda.apply_lda(ULDA, DTR_pca)

    DVAL_lda = lda.apply_lda(ULDA, DVAL_pca)  # Applied to PCA-transformed validation data

    threshold = (DTR_lda[0, LTR == 1].mean() + DTR_lda[
        0, LTR == 2].mean()) / 2.0  # Estimated only on model training data

    PVAL = numpy.zeros(shape=LVAL.shape, dtype=numpy.int32)
    PVAL[DVAL_lda[0] >= threshold] = 2
    PVAL[DVAL_lda[0] < threshold] = 1
    print('Labels:     ', LVAL)
    print('Predictions:', PVAL)
    print('Number of erros:', (PVAL != LVAL).sum(), '(out of %d samples)' % (LVAL.size))
    print('Error rate: %.1f%%' % ((PVAL != LVAL).sum() / float(LVAL.size) * 100))

    print("\n" + "=" * 50)
    print("ANALISI PERFORMANCE: PCA + LDA AL VARIARE DI m")
    print("=" * 50)

    # Testiamo tutti i possibili valori di m (da 1 a 4 feature)
    for m in range(1, 5):
        # 1. Calcoliamo la PCA SOLO sul training set con dimensione m
        UPCA = pca.compute_pca(DTR, m=m)

        # 2. Applichiamo la trasformazione PCA sia al training che al validation set
        DTR_pca = pca.apply_pca(UPCA, DTR)
        DVAL_pca = pca.apply_pca(UPCA, DVAL)

        # 3. Calcoliamo la LDA sui dati pre-processati con la PCA
        ULDA = lda.compute_lda_geig(DTR_pca, LTR, m=1)

        # 4. Proiettiamo il training set per controllare l'orientamento
        DTR_lda = lda.apply_lda(ULDA, DTR_pca)

        # 5. Controllo Orientamento: assicuriamoci che Virginica sia "a destra" (valori maggiori)
        if DTR_lda[0, LTR == 1].mean() > DTR_lda[0, LTR == 2].mean():
            ULDA = -ULDA
            DTR_lda = lda.apply_lda(ULDA, DTR_pca)

        # 6. Proiettiamo i dati di validazione
        DVAL_lda = lda.apply_lda(ULDA, DVAL_pca)

        # 7. Calcoliamo la soglia usando rigorosamente SOLO i dati di training
        threshold = (DTR_lda[0, LTR == 1].mean() + DTR_lda[0, LTR == 2].mean()) / 2.0

        # 8. Generiamo le predizioni
        PVAL = numpy.zeros(shape=LVAL.shape, dtype=numpy.int32)
        PVAL[DVAL_lda[0] >= threshold] = 2
        PVAL[DVAL_lda[0] < threshold] = 1

        # 9. Calcoliamo e stampiamo le metriche
        errori = (PVAL != LVAL).sum()
        error_rate = (errori / float(LVAL.size)) * 100

        print(f"Dimensioni PCA (m = {m}):")
        print(f"  -> Errori: {errori} su {LVAL.size}")
        print(f"  -> Tasso di errore: {error_rate:.1f}%\n")

# Domande

# 1)
# Osservando gli istogrammi delle 6 direzioni della PCA, notiamo che le distribuzioni per le singole feature
# sembrano diverse rispetto a quelle originali.
# No, i cluster non sono davvero cambiati e le classi non sono diventate più separate
# Applicare una PCA mantenendo tutte e 6 le dimensioni equivale puramente ad una rotazione matematica
# dello spazio delle feature. Le distrnaze relative tra i punti restano esattamente le stesse.
# QUesto dimostra che guardare solo gli istogrammi monodimenisonali può essere fuorviante se non si teien conto della struttura globale dei dati

# 2)
# Proiettando i campioni sull'unica direzione calcolata dalla LDA, noriamo che i due istogrammi delle classi sono molto ben distinti.
# La sovrapposizione tra le due distribuzioni è minima
# Rispetto all'osservazione delle 6 feature originali singolarmente, la LDA ha trovato una direzione di proiezione eccellente.
# La LDA non sta magicamente allontanando i dati nello sapzio, ma sta individuando l'angolazione da cui guardare i dati
# affinchè le due classi risultimno naturalmente il più separate possibile

# 3)
# Modificando il valore della soglia, il tasso di errore calcolato sul validation set oscillerà
# E' possibile trovare valori che migliorano l'accuratezza. La media esatta delle due medie è una soglia ottimale sono in un caso teorico ideale:
# quando entrambe le classi hanno esattamente la stessa identica varianza e sono perfettamente bilanciate come numero.
# Nella realtà, una classe potrebbe essere più stretta o densa dell'altra. Spostare la soglia leggermente in direzione
# della classe più densa solitamente aiuta a ridurre i falsi positivi e negativi

# 4)
# Con m = 1 abbiamo 4 errori su 34 poichè stiamo costringendo la PCA a manetenere una singola componente principale
# e stiamo schiacciando via troppe informazioni vitali
# Con m = 2, 3, 4 le prestazioni si stabilizzano. L'aggiunta della terza o della quarta componente principale non apporta benefici
# La PCA è vantaggiosa quando combinata con la LDA. Applicando la PCA con m = 2 abbiamo compresso le feature del dataset del 50%
# mantenendo però il 100 % del potere predittivo. Le componenti scartate contenevano quindi rumore di fondo o varianza irrilevante.
# In scenari complessi, questo pre-processing diventa vitale poichè riduce drasticamente i costi, previene l'overfitting ed evita errori matematici
