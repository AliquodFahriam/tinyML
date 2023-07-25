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

Permette di ottimizzare funzioni *black-box* che sarebbero difficili da valutare. Ovvero che richiederebbero enormi risorse di calcolo e i cui meccanismi interni non possono essere compresi con precisione. 

Un esempio di ciò è l'ottimizzazione degli **iperparametri** per una rete neurale per cui ogni iterazione potrebbe richiedere alcuni giorni. Un metodo efficiente richiede di trovare il miglior insieme di iperparametri utilizzando il numero minimo di iterazioni. Questo metodo si chiama **Bayesian Optimization**. 

Bayesian optimization is a **machine learning** based optimization algorithm <mark>used to find the parameters that globally optimizes a given black box function</mark>. There are 2 important components within this algorithm:

* The black box function to optimize: $f(x)$.<br>
  We want to find the value of $x$ which globally optimizes $f(x)$. The $f(x)$ is also sometimes called the objective function, the target function, or the loss function depending on the problem. In general, we only have knowledge about the inputs and outputs of $f(x)$.

* The acquisition function: <br>
  $a(x)$, which is used to generate new values of $x$ for evaluation with $f(x)$. $a(x)$ internally relies on a Gaussian process model $m(X, y)$ to generate new values of $x$.

The optimization process itself is as follows:

1. Define the black box function $f(x)$, the acquisition function $a(x)$ and the search space of the parameter x.
2. Generate some initial values of $x$ randomly, and measure the corresponding outputs from $f(x)$.
3. Fit a <a href="">Gaussian process model</a> $m(X, y)$ onto $X = x$ and $y = f(x)$. In other words,$ m(X, y)$ serves as a surrogate model for $f(x)$!
4. The acquisition function $a(x)$ then uses $m(X, y)$ to generate new values of $x$ as follows. Use $m(X, y)$ to predict how $f(x)$ varies with $x$. The value of $x$ which leads to the largest predicted value in $m(X, y)$ is then suggested as the next sample of $x$ to evaluate with $f(x)$.

Repeat the optimization process in steps 3 and 4 until we finally get a value of x that leads to the global optimum of f(x). Note that all historical values of x and f(x) should be used to train the Gaussian process model m(X, y) in the next iteration — as the number of data points increases, m(X, y) becomes better at predicting the optimum of f(x).

<a href ="https://towardsdatascience.com/bayesian-optimization-with-python-85c66df711ec">Link di riferimento</a>

    Esistono librerie python che ci possono permettere di utilizzare questo algoritmo

Codice  python: 

```python
# Prepare the data.
cancer = load_breast_cancer()
X = cancer["data"]
y = cancer["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y,stratify = y,random_state = 42)scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)# Define the black box function to optimize.

def black_box_function(C):
    # C: SVC hyper parameter to optimize for.
    model = SVC(C = C)
    model.fit(X_train_scaled, y_train)
    y_score = model.decision_function(X_test_scaled)
    f = roc_auc_score(y_test, y_score)
    return f# Set range of C to optimize for.

# bayes_opt requires this to be a dictionary.
pbounds = {"C": [0.1, 10]}# Create a BayesianOptimization optimizer,

# and optimize the given black_box_function.
optimizer = BayesianOptimization(f = black_box_function,pbounds = pbounds, verbose = 2,
random_state = 4)
optimizer.maximize(init_points = 5, n_iter = 10)
print("Best result: {}; f(x) = {}.".format(optimizer.max["params"], optimizer.max["target"]))
```

# Regressione logistica

Per regressione logistica si intende l'analisi di regressione ch esi conduce quando la variabile dipendente è *dicotomica*, ovvero binaria. È ovviamente un'analisi predittiva. 

Il modello di regressione logistica, anche chiamato $logit(p)$, si compone dalle seguenti variabili 

* **Y**: variabile dipendente. Assume valore 0 quando l'evento non si verifica e valore 1 quando l'evento si verifica. 
* $X_i, \space i = 1,...,n$: variabili dipendenti o **regressori**. Possono avere qualsiasi natura e rappresentano i fattori di rischio che influenzano la variabile Y. 

Il modello da stimare è dato dall'espressione

$Y = ln(\frac {p}{1-p}) = \beta_0 + \beta_1 *X_1 + \beta_2*X_2 + ...+\beta_n * X_n + \epsilon$

Che può essere anche espressa in termini di probabilità $p$

$p = \frac {1} {\beta_0 + \beta_1 *X_1 + \beta_2*X_2 + ...+\beta_n * X_n} + \epsilon $

### Coefficienti di regressione

Gli esponenziali dei coefficienti $\beta_i$ rappresentano una misura che quantifica quanto più è "alta"la probabilità che si verifichi l'evento. 
In maniera più rigorosa: 

*I coefficienti $\beta_i$ si interpretano come l’**odds ratio (OR)** di accadimento dell'evento per ogni incremento della variabile indipendente, al netto delle altre variabili indipendenti.* 

I coefficienti sono quindi associati ad ogni variabile indipendente $X_i$, e rappresentano il "peso"che la variabile ha nel calcolo della probabilità che l'evento si verifichi. In particolare abbiamo

* Quando $\beta_i$ è positivo allora $OR > 1$ e quindi la variabile associata $X_i$ ha un peso sul verificarsi dell'evento
* Quando il valore di $\beta_i$ è negativo allora $OR < 1$, quindi la variabile associata $X_i$ ha un peso sul **NON** verificarsi dell'evento
* Quando $\beta_i$ è nullo, $X_i$ non influisce sulla variabile dipendente Y. 
