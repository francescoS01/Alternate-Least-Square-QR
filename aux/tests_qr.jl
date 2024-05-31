include("../utils/alternate_LSQR.jl")


# Esempi di test

A = 50*rand(5, 4)
print("\n------- matrice A -------\n")
print_matrix(A)

U, V = LS_QR_alternate(A, 3) 
print("\n------- matrice U*V' -------\n")
print_matrix(U*V')

"""
Un risultato ottenuto è il seguente:

------- matrice A -------
28.82886 37.35867 31.88146 48.53332 
35.61269 30.49915 19.03065 46.60306 
5.40662 5.65864 20.05325 28.00300 
46.61719 0.16429 37.28762 0.75929 
11.31081 9.98594 16.21314 45.82430 

------- matrice U*V' -------
32.95318 33.53614 26.86602 51.02295 
32.28102 33.58704 23.08218 44.59191 
7.38384 3.82610 17.64882 29.19654 
46.19066 0.55962 37.80631 0.50181 
8.60944 12.48964 19.49817 44.19363 

ERRORE 1 : 20.765083062298878
-------
ERRORE 2 : 12.390006630401503
-------
ERRORE 3 : 12.122290528805989
-------
ERRORE 4 : 12.0959007297781
-------
ERRORE 5 : 12.09302562024261
-------
ERRORE 6 : 12.092711862520552
-------
... ... ...
-------
ERRORE 48 : 12.09267343602253
-------
ERRORE 49 : 12.092673436022533
-------
ERRORE 50 : 12.092673436022533
-------

Notiamo che l'errore tende sempre a calare e mai a crescere ma dopo poche iterazioni cala molto lentamente fino a stabilizzarsi su un certe valore
senza riuscire a convergere ad un errore prossimo a zero.
Possiamo infatti notare che la matrice A ha alcuni valori abbastanza diversi da U*V'.

Crediamo che questo possa essere causato dall'inizializzazione random non adatta di V. 

"""
