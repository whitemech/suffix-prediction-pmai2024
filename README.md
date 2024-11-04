# Enhancing Deep Sequence Generation with Logical Temporal Knowledge

*Elena Umili, Gabriel Paludo Licks, and Fabio Patrizi*

Source code for paper submitted to the PMAI@ECAI2024 workshop.

## Dependencies

You can find a `Dockerfile` in this repository. You can either run the code on a Docker container or use the Dockerfile as a reference for the dependencies (mainly CUDA/torch and MONA) that need to be installed. You can also find an `environment.yml` file referring to a Conda virtual environment containing the Python dependencies. 

## Running the code

The file `run_all.py` is the script that runs all experiments shown in the paper. Once the script finishes executing, you can plot the results using the `plot.py` file.

## Citation
```
@inproceedings{UmiliEnhancing,
  author       = {Elena Umili and
                  Gabriel Paludo Licks and
                  Fabio Patrizi},
  editor       = {Giuseppe De Giacomo and
                  Valeria Fionda and
                  Fabiana Fournier and
                  Antonio Ielo and
                  Lior Limonad and
                  Marco Montali},
  title        = {Enhancing Deep Sequence Generation with Logical Temporal Knowledge},
  booktitle    = {Proceedings of the 3rd International Workshop on Process Management
                  in the {AI} Era {(PMAI} 2024) co-located with 27th European Conference
                  on Artificial Intelligence {(ECAI} 2024), Santiago de Compostela,
                  Spain, October 19, 2024},
  series       = {{CEUR} Workshop Proceedings},
  volume       = {3779},
  pages        = {23--34},
  publisher    = {CEUR-WS.org},
  year         = {2024},
  url          = {https://ceur-ws.org/Vol-3779/paper4.pdf},
  timestamp    = {Mon, 28 Oct 2024 16:46:06 +0100},
  biburl       = {https://dblp.org/rec/conf/pmai/UmiliLP24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
