# WFG hyper-volume calculation

This crate contains a Rust wrapper to the C program to calculate the hyper-volume metric using the WFG algorithm
proposed by

> L. While, L. Bradstreet and L. Barone, "A Fast Way of Calculating Exact Hypervolumes," in IEEE Transactions on
> Evolutionary Computation, vol. 16, no. 1, pp. 86-95, Feb. 2012, doi: 10.1109/TEVC.2010.2077298.

The original source code is available at https://github.com/lbradstreet/WFG-hypervolume/tree/master under the `GPL2`
license.

The following changes were applied to the code in the `vendor` folder (from commit `b19f35c`):

- the code is now more flexible and can be used a library by calling the `calculate_hypervolume` function;
- the code was cleanup by removing unnecessary dependencies, formatted and documentation was added;
- the part of the code calculating the hyper-volume from input files was removed.

This code is re-released under the `GPL3` license.