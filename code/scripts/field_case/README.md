# Field case

This folder contains the scripts to the figures in section 4 of [the paper](https://academic.oup.com/gji/advance-article-pdf/doi/10.1093/gji/ggz402/29533315/ggz402.pdf).

- Copy data from the `data` folder to the current working directory

      make prepare

- Build mesh and perform preprocessing

      make mesh

- Produce the results shown in Fig. 5a

      make inv

- Produce the results shown in Fig. 5b

      make joint1

- Produce the results shown in Fig. 5c

      make joint2

- Produce figures (Fig.5-7)

      make show

- Remove all files produced by the commands above.

      make clean
