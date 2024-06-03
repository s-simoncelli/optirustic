# Hyper-volume calculation

This crate contains a Rust wrapper to the C program to calculate the hyper-volume metric
for a multi-objective optimisation in `d` dimensions with `O(n^(d-2) log n)` time and linear space complexity
(in the worst-case). The algorithm was written by Carlos M. Fonseca, Manuel López-Ibáñez, Luís Paquete, and
Andreia P. Guerreiro and is available at:

https://lopez-ibanez.eu/hypervolume#download

Relevant literature:

1. Carlos M. Fonseca, Luís Paquete, and Manuel
   López-Ibáñez. [An improved dimension - sweep algorithm for the hypervolume
   indicator](http://dx.doi.org/10.1109/CEC.2006.1688440). In Proceedings of the 2006 Congress on Evolutionary
   Computation (CEC'06), pages 1157–1163. IEEE Press,
   Piscataway, NJ, July 2006.
   [ [bibtex](https://lopez-ibanez.eu/LopezIbanez_bib.html#FonPaqLop06:hypervolume)
   | [10.1109/CEC.2006.1688440](http://dx.doi.org/10.1109/CEC.2006.1688440) | [PDF](https://lopez-ibanez.eu/doc/FonPaqLop06-hypervolume.pdf) ]

2. Nicola Beume, Carlos M. Fonseca, Manuel López-Ibáñez, Luís Paquete, and J.
   Vahrenhold. [On the complexity of computing
   the hypervolume indicator](http://dx.doi.org/10.1109/TEVC.2009.2015575). IEEE Transactions on Evolutionary
   Computation, 13(5):1075–1082, 2009.