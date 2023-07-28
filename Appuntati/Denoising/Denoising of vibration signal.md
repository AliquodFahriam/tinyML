# Denoising of vibration signal

Uno dei problemi chiave che bisogna risolvere quando si cerca di analizzare o, comunque di utilizzare, dati provenienti da un accelerometro è la grande quantità di rumore presente al loro interno. In particolare consideriamo il segnale composto nel seguente modo: 

$x(t) = s(t) + n(t)$

In cui $s(t)$ rappresenta il segnale di vibrazione mentre $n(t)$ rappresenta la componente di rumore ambientale. Esistono in particolare due tipi di rumore che ci interessano: 

- Il rumore *impulsivo*

- Il rumore *bianco*

Di solito il primo viene eliminato tramite l'utilizzo di un **filtro mediano**. Il rumore impulsivo possiede infatti le seguenti caratteristiche: 

1. Grande ampiezza 

2. Breve durata

3. Grande intervallo tra un impulso e l'altro

Per quanto riguarda il rumore bianco invece si preferisce utilizzare ***La tecnologia di denoising wavelet***. Di conseguenza per ottenere un effetto di denoising ideale si potrebbe pensare di combinare il filtro mediano con il denoising wavelet. 

Il metodo proposto nel <a href='./Denoising.pdf'> documento di riferimento</a> combina il ***filtro mediano con adattamento dinamico della finestra*** a un approccio di ***wavelet packet denoising con adattamento dinamico della scala di decomposizione e della soglia*** per rimuovere il rumore

## Filtro mediano

Questo filtro risulta essere nonlineare e possiede grandi capacità di sopprimere il rumore impulsivo. L'operazione di filtraggio avviene scansionando i dati campionati facendogli scorrere sopra una finestra di lunghezza fissata e rimpiazzando i dati al centro della finestra con il valore medio dei dati al suo interno. Alla fine del movimento della finestra. 

Quindi tutto dipende dalla finestra: nel caso questa venga posta troppo grande oppure verranno persi dettagli fondamentali del segnale, nel caso invece venga posta troppo piccola non si riuscirà a rimuovere per la maggior parte il rumore. 

## Wavelet Packet Analysis

Questa risulta più precisa della wavelet analysis semplice. 

*Il seguente testo è generato da ChatGPT*

La Wavelet Packet Analysis (analisi dei pacchetti di onde) è una tecnica di analisi dei segnali che estende le capacità della trasformata wavelet discreta (DWT) consentendo una decomposizione più approfondita e versatile del segnale in bande di frequenza e tempo. Questa analisi offre una maggiore flessibilità rispetto alla DWT, poiché permette di esplorare e suddividere ulteriormente le sottobande ottenute durante la trasformazione.

Ecco come funziona la Wavelet Packet Analysis:

1. Decomposizione del segnale: La Wavelet Packet Analysis inizia con una fase di decomposizione simile a quella della DWT. Un segnale viene convoluto con una funzione wavelet madre (spesso chiamata wavelet di base) e un filtro passa-alto e passa-basso. Questa operazione produce due sotto-segnali chiamati approssimazione (coefficients of approximation) e dettagli (coefficients of detail).

2. Iterative Decomposition: A differenza della DWT, che si ferma a un certo livello di decomposizione, la Wavelet Packet Analysis prosegue iterativamente. In ogni iterazione, sia l'approssimazione che i dettagli vengono sottoposti a una nuova decomposizione usando una wavelet diversa o la stessa wavelet di base, ma con parametri differenti. Questo processo di iterazione consente di creare alberi di decomposizione più profondi e complessi.

3. Selezione dei pacchetti di onde (Wavelet Packets): L'utente può scegliere i nodi dell'albero di decomposizione (noti come pacchetti di onde) che rappresentano le sottobande di interesse. I pacchetti di onde corrispondono ai coefficienti di dettaglio e approssimazione a diversi livelli di decomposizione e rappresentano bande di frequenza e tempo specifiche del segnale.

4. Ricostruzione del segnale: Dopo la selezione dei pacchetti di onde, è possibile ricostruire il segnale originale combinando solo quei pacchetti selezionati. Questo processo di ricostruzione permette di concentrarsi sulle caratteristiche di interesse del segnale e di ignorare componenti indesiderate.

L'utilizzo della Wavelet Packet Analysis offre diversi vantaggi, come una migliore risoluzione in frequenza e tempo rispetto alla DWT tradizionale, il che può essere particolarmente utile nell'analisi di segnali complessi o nell'identificazione di eventi transitori. Tuttavia, va notato che una decomposizione troppo profonda può portare a un'eccessiva sensibilità al rumore o all'overfitting, quindi una selezione adeguata dei pacchetti di onde è essenziale per ottenere risultati significativi e utili.

#### Per quanto riguarda la sogliatura

La soglia è una parte importante della Wavelet Packet Analysis (WPA) e di altre tecniche di analisi basate sulle trasformate wavelet. Essa viene utilizzata per eliminare o attenuare i coefficienti di dettaglio che sono al di sotto di una certa soglia. Questo processo, noto come "thresholding" o sogliatura, è un passo comune nella denoising (rimozione del rumore) e nella compressione dei segnali.

Quando si esegue la decomposizione dei pacchetti di onde, si ottengono diversi coefficienti di dettaglio a diversi livelli di decomposizione. Questi coefficienti di dettaglio rappresentano le differenze tra i dettagli dell'immagine originale e l'approssimazione. Nei segnali reali, i coefficienti di dettaglio possono contenere rumore o altre componenti indesiderate.

La sogliatura è un meccanismo per ridurre o eliminare il rumore e altre informazioni meno significative dai coefficienti di dettaglio, concentrandosi solo sulle informazioni più rilevanti. Ciò migliora la qualità della ricostruzione e consente di concentrarsi sulle caratteristiche importanti del segnale.

Esistono diversi metodi per applicare la sogliatura nella Wavelet Packet Analysis, tra cui:

1. Sogliatura Soft: In questo approccio, i coefficienti di dettaglio al di sotto di una certa soglia vengono annullati (posti a zero), mentre quelli al di sopra della soglia rimangono invariati. Questo metodo tende a produrre una maggiore compressione dei dati e può essere utile quando si desidera rimuovere il rumore e mantenere un segnale più compatto.

2. Sogliatura Rigida: In questo caso, i coefficienti di dettaglio al di sotto della soglia vengono annullati, proprio come nella sogliatura soft. Tuttavia, a differenza della sogliatura soft, i coefficienti al di sopra della soglia vengono mantenuti, senza subire alcuna modifica. Questo approccio può essere utile quando si desidera una maggiore conservazione delle informazioni nei coefficienti significativi.

3. Sogliatura Proportionate: Questo metodo adatta la soglia in base a una percentuale o una proporzione dei coefficienti di dettaglio. Ad esempio, si può decidere di eliminare tutti i coefficienti di dettaglio inferiori al 10% del valore massimo tra tutti i coefficienti.

L'efficacia della sogliatura dipende dalla corretta scelta del valore della soglia stessa. Un valore di soglia troppo alto potrebbe eliminare informazioni significative, mentre un valore troppo basso potrebbe non eliminare a sufficienza il rumore. La scelta della soglia può essere guidata da tecniche di stima del rumore o da criteri soggettivi basati sulla natura del segnale e sugli obiettivi dell'analisi. Inoltre, la sogliatura può essere applicata anche in modo adattivo, dove il valore della soglia può variare in base al contenuto locale dei coefficienti di dettaglio.



## Applicazione del denoising migliorato

Passo 1: in base alla frequenza di campionamento Fs del segnale, calcolare la larghezza della finestra Ld del filtro mediano con l'equazione (7).
Passo 2: dopo aver ottenuto la larghezza della finestra Ld, il segnale di rumore viene filtrato con il filtro mediano utilizzando la formula (2).
Passo 3: effettuare la decomposizione dei pacchetti di onde per il segnale filtrato con il filtro mediano, e utilizzare la funzione di costo M(x) espressa nella formula (3) come criterio di giudizio per determinare se la decomposizione prosegue o meno, al fine di determinare la scala di decomposizione ottimale e la base di pacchetti di onde ottimale.
Passo 4: utilizzare il valore di soglia migliorato e l'espressione della funzione di soglia per elaborare il coefficiente wj,k di ciascun pacchetto di onde e ottenere il nuovo coefficiente stimato w⌢j,k.
Passo 5: ricostruire il segnale con il nuovo coefficiente wj,k dopo la contrazione della soglia su ciascuna scala per ottenere il segnale di denoising s⌢(n), che è una stima del segnale di vibrazione reale s(n).



#### Equazioni di riferimento

$L_d = 2L_sF_s$             (7)







$y(n) = Med[x(n-d),..., x(n),...,x(n-d)]$             (2)







$M(x) = -\sum_j p_j log_2(p_j), \space \space \space p_j = \frac {|x_j|^2} {||x||^2}$                            (3)



$\begin{cases} 0, & \text{if} \space \space |w_{j,k}| \lt T \\
sgn(w_{j,k})(|w_{j,k}|³ - T^3)^{\frac {1}{3} }, &\text{otherwise} \space \space   \end{cases}$

```
\begin{cases}
    \frac{x^2-x}{x},& \text{if } x\geq 1\\
    0,              & \text{otherwise}
\end{cases}
```
















