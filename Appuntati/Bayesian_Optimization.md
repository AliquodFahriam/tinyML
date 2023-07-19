# Adaptive Maximum Correlation Kurtosis Deconvolution (AMCKD) 
*Ciò che segue è invenzione di chatGPT* 

<a href="https://chat.openai.com/share/7fb2ce92-1112-4faa-b712-6306edace401">Link alla conversazione</a>

La "kurtosis deconvolution" è un metodo utilizzato per stimare la forma originale di un segnale o una serie temporale a partire dalla sua kurtosi, che è una misura statistica della forma di una distribuzione.

La kurtosis deconvolution si basa sull'idea che la kurtosi di un segnale possa essere influenzata da effetti di convoluzione o distorsioni introdotte da processi o canali di trasmissione. Pertanto, invertendo o deconvolvendo la kurtosi, è possibile stimare la forma originale del segnale prima della convoluzione o distorsione.

Il processo di kurtosis deconvolution coinvolge tipicamente i seguenti passaggi:

1. Calcolo della kurtosi: Viene calcolata la kurtosi della serie temporale osservata o del segnale convoluto.

2. Stima della funzione di risposta: Utilizzando modelli o ipotesi sulla convoluzione o la distorsione, viene stimata la funzione di risposta del processo o del canale che ha generato la kurtosi osservata.

3. Deconvoluzione: La kurtosi osservata viene deconvoluta utilizzando la funzione di risposta stimata per stimare la forma originale del segnale. Questo può essere fatto applicando un filtro inverso o utilizzando algoritmi di deconvoluzione.

La kurtosis deconvolution può essere utile in diversi contesti, come l'analisi di segnali audio o l'identificazione di caratteristiche nascoste in serie temporali. Tuttavia, è importante notare che la kurtosis deconvolution può essere un processo complesso e può richiedere modelli accurati o conoscenze specifiche sulla natura della convoluzione o della distorsione presente nel segnale.

È possibile trovare implementazioni specifiche di kurtosis deconvolution in pacchetti o librerie di segnali e analisi statistica, ma è importante notare che l'efficacia e l'applicabilità del metodo dipenderanno dalla natura del segnale e delle distorsioni coinvolte.

In particolare però la Maximum Correlation Kurtosis Deconvolution è stata ideata da:
McDonald et al. *"Maximum correlated Kurtosis Deconvolution and application on gear tooth chip fault detection"* (Di cui disponiamo una copia nella sezione documenti)

    E la parola adaptive? 
    A quanto pare viene resa adaptive dallo Sparrow Search Algorithm (SSA) 
    che viene utilizzato per selezionare due parametri chiave quali M ed L

# Intinsic Time-Scale Decomposition (ITD)
*Ciò che segue è invenzione di chatGPT* 

*"La decomposizione intrinseca della scala temporale (intrinsic time-scale decomposition) è un metodo utilizzato per analizzare le serie temporali, al fine di identificare e separare le variazioni su diverse scale temporali presenti nei dati. Questa tecnica è particolarmente utile nell'ambito della previsione e analisi delle serie temporali non stazionarie, in cui i modelli tradizionali potrebbero non essere in grado di catturare le variazioni su diverse scale.*

*L'idea principale della decomposizione intrinseca della scala temporale si basa sulla teoria delle onde e sulla trasformata di Fourier.<mark> La serie temporale viene decomposta in una serie di componenti che rappresentano variazioni su diverse scale di tempo, partendo dalle scale più grandi alle scale più piccole.</mark>*

*Il primo passo della decomposizione consiste nel filtrare la serie temporale per estrarre la componente di lungo termine. Questo viene fatto utilizzando una trasformata di Fourier e rimuovendo le frequenze corrispondenti alle scale di tempo più piccole. Ciò consente di isolare la tendenza o la <mark>componente di lungo termine dei dati.*

*Successivamente, <mark>la serie temporale viene sottratta dalla sua componente di lungo termine per ottenere una serie temporale residua</mark>. Questa serie residua contiene principalmente le variazioni a breve termine o le fluttuazioni veloci presenti nei dati.*

*<mark>Il processo viene quindi ripetuto sulla serie residua per estrarre la componente successiva di scala temporale più piccola. Questo passaggio viene iterato fino a quando non rimane una componente residua significativa.*

*L'obiettivo finale della decomposizione intrinseca della scala temporale è ottenere una serie di componenti che rappresentano le variazioni su diverse scale temporali. Queste componenti possono essere analizzate singolarmente o combinate per comprendere meglio i modelli e le tendenze presenti nei dati.*

***Dal portale di MATLAB***: 
Intrinsic Time-Scale Decomposition (ITD) is an adaptive and data-driven method like Empirical Mode Decomposition (EMD). It can decompose a complex signal into several Proper Rotation Components (PRCs) and a residual. 


# The Bayesian Optimization Algorithm (BOA)

Bayesian optimization is a **machine learning** based optimization algorithm <mark>used to find the parameters that globally optimizes a given black box function</mark>. There are 2 important components within this algorithm:

* The black box function to optimize: f(x).<br>
We want to find the value of $x$ which globally optimizes $f(x)$. The $f(x)$ is also sometimes called the objective function, the target function, or the loss function depending on the problem. In general, we only have knowledge about the inputs and outputs of $f(x)$.

* The acquisition function: <br>
$a(x)$, which is used to generate new values of $x$ for evaluation with $f(x)$. $a(x)$ internally relies on a Gaussian process model $m(X, y)$ to generate new values of $x%.

The optimization process itself is as follows:

1. Define the black box function $f(x)$, the acquisition function $a(x)$ and the search space of the parameter x.
2. Generate some initial values of $x$ randomly, and measure the corresponding outputs from $f(x)$.
3. Fit a Gaussian process model $m(X, y)$ onto $X = x$ and $y = f(x)$. In other words,$ m(X, y)$ serves as a surrogate model for $f(x)$!
4. The acquisition function $a(x)$ then uses $m(X, y)$ to generate new values of $x$ as follows. Use $m(X, y)$ to predict how $f(x)$ varies with $x$. The value of $x$ which leads to the largest predicted value in $m(X, y)$ is then suggested as the next sample of $x$ to evaluate with $f(x)$.

Repeat the optimization process in steps 3 and 4 until we finally get a value of x that leads to the global optimum of f(x). Note that all historical values of x and f(x) should be used to train the Gaussian process model m(X, y) in the next iteration — as the number of data points increases, m(X, y) becomes better at predicting the optimum of f(x).

<a href ="https://towardsdatascience.com/bayesian-optimization-with-python-85c66df711ec">Link di riferimento</a>

    Esistono librerie python che ci possono permettere di utilizzare questo algoritmo

