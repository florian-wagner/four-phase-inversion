# Synthetic case

This folder contains the scripts to the figures in section 3 of [the paper](https://academic.oup.com/gji/advance-article-pdf/doi/10.1093/gji/ggz402/29533315/ggz402.pdf).

- Generate the mesh and the synthetic data sets

      make calc

- Perform conventional inversions of both data sets.

      make inv

- Produce the results shown in Fig. 2 (with known porosity)

      make case1

- Produce the results shown in Fig. 3 (without a-priori knowledge of the porosity distribution)

      make case2

- Calculate the model covariance matrix based on grouped parameters (Fig. 4)

      make matrix

- Produce the figures (Fig. 2-4)

      make show

- Remove all files produced by the commands above.

      make clean
