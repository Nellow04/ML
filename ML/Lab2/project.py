import matplotlib.pyplot as plt
import numpy

def mcol(v):
    return v.reshape((v.size, 1))

def load_data(file_name):

    dataset_list = []
    label_list = []

    with open(file_name) as file:
        for line in file:
            try:
                features = line.split(",")[0:-1]
                features = mcol(numpy.array([float(i)for i in features]))
                value = line.split(",")[-1].strip()
                label = int(value)

                dataset_list.append(features)
                label_list.append(label)
            except:
                pass

    return numpy.hstack(dataset_list), numpy.array(label_list, dtype=numpy.int32)

def plot_hist(D, L):

    D0 = D[:, L==0]
    D1 = D[:, L==1]

    hFea = {
        0: "Feature 1",
        1: "Feature 2",
        2: "Feature 3",
        3: "Feature 4",
        4: "Feature 5",
        5: "Feature 6",
    }

    for dIdx in range(6):
        plt.figure()
        plt.xlabel(hFea[dIdx])
        plt.ylabel("Density")

        plt.hist(D0[dIdx, :], bins = 10, density = True, alpha = 0.4, label = "Fake")
        plt.hist(D1[dIdx, :], bins = 10, density = True, alpha = 0.4, label = "Genuine")

        plt.legend()
        plt.tight_layout()
        plt.savefig("hist_%d.pdf" % dIdx)

def plot_scatter(D, L):

    D0 = D[:, L == 0]
    D1 = D[:, L == 1]

    hFea = {
        0: "Feature 1",
        1: "Feature 2",
        2: "Feature 3",
        3: "Feature 4",
        4: "Feature 5",
        5: "Feature 6",
    }

    for dIdx1 in range(6):
        for dIdx2 in range(6):
            if dIdx1 == dIdx2:
                continue

            plt.figure()
            plt.xlabel(hFea[dIdx1])
            plt.ylabel(hFea[dIdx2])

            plt.scatter(D0[dIdx1, :], D0[dIdx2, :], label = "Fake")
            plt.scatter(D1[dIdx1, :], D1[dIdx2, :], label = "Genuine")

            plt.legend()
            plt.tight_layout()
            plt.savefig("scatter_%d_%d.pdf" % (dIdx1, dIdx2))


if __name__ == "__main__":

    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    D, L = load_data("Solution/trainData.txt")

    plot_hist(D, L)
    plot_scatter(D, L)


# Domande

# 1)
# Quanto riguarda la Feature 1, si osserva una forte sovrapposizione tra le classi Fake e Genuine. Questa
# è particolarmente densa nella regione centrale del grafico, concentrandosi nell'intervallo tra -2 e 2
# Per la Feature 2, anche in questo caso le due classi si sovrappongono in modo significativo, con
# la zona di maggiore intersezione sempre localizzata attorno al valore 0

# Le classi mostrano medie molto simili per entrambe le prime due feature. I centri di entrambe le distribuzioni,
# sia per la Feature 1 che per la Feature 2, sono allineati quasi esattamente sullo 0

# Le varianze, al contrario, non sono simili e mostrano un comportamento opposto tra le due feautre.
# Nella Feature 1, la classe Fake ha una varianza molto più piccola (forma stretta e alta),
# mentre la classe Genuine presenta una varianza maggiore, risultando più allargata sui lati
# Nella Feature 2, la situazioni si inverte. La classe Genuine ha una varianza minore ed più concentrata,
# mentre la classe Fake ha una varianza maggiore e sis estende su un intervallo più ampio

# Osservando le distribuzioni, emerge chiaramente un solo picco (una sola moda) per ciascuna classe
# in entrambe le feature. Si tratta, in tutti i casi, di distribuzioni unimodali centrate attorno allo zero

# 2)
# Per la Feature 3, differenza delle prime due feature, qui si nota una separazione molto più marcata tra le due classi.
# Tuttavia, esiste ancora una zona di sovrapposizione che si concentra prevalentemente nell'intervallo tra -1 e 1
# Quanto riguarda la Feature 4, anche qui le classi risultano ben distinte ma presentano una regione di sovrapposizione.
# Questa intersezione aviene nella zona centrale del grafico, all'incirca tra i valori -1 e 1

# Per queste due feature le classi non mostrano una media smile. Il loro comportamento è speculare.
# Nella 3, la classe Fake è traslata verso sinistra con la media intorno a -1, mentre la classe Genuine
# è posizionata a destra con la media intorno a 1. Nella 4, la situazione è l'opposto

# A differenza delle prime due feature, in questo caso le varianze sono molto simili tra le due classi.
# L'ampiezza delle campane è visivamente quasi identica

# Per entrambe le feature e per entrambe le classi si osserva chiaramente un solo picco, quindi una sola moda
# all'interno degli istogrammi. Tutte le distribuzioni si confermano quindi unimodali

# 3)
# A differenza delle feature precedenti, qui notiamo un comportamento diverso. Le classi presentano una
# forte sovrapposizione su quasi tutto il range di valori, in particolar modo nella zona centrale
# dove le distribuzioni si intersecano ripetutamente

# In questi grafici emerge la differenza più significativa.
# La classe Fake mantiene un andamento unimodale in entrambe le feature, con un singolo picco centrale
# La classe Genuine mostra un evidente comportamento bimodale in entrambe le feature. La distribuzione si divide
# chiaramente in due campane con due picchi ben distinti, intorno a -1 e 1

# Poichè la classe Genuine è bimodale su entrambe le feature, vediamo più cluster distinti (4 o 2) per i dati Genuine.
# Al contrario, poichè la classe Fake ha una sola moda centrata sullo zero per entrambe le feature,
# i suoi punti formano un solo grande cluster compatto al centro del grafico

