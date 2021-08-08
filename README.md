# RGN2-Replica

To eventually become an unofficial working Pytorch implementation of [RGN2](https://www.biorxiv.org/content/10.1101/2021.08.02.454840v1), an state of the art model for MSA-less Protein Folding for particular use when no evolutionary homologs are available (ie. for protein design). 

## Install

```bash
$ pip install rgn2-replica
```

## TO-DO LIST: 

* [-] Provide basic package and file structure
* [ ] Contribute adaptation of RGN1 for matrix. 
* [ ] Contribute b
* [ ] Contirbute trainer classes / functionality
* [ ] Adapt functionality from [MP-NeRF](https://github.com/EleutherAI/mp_nerf): 
	* [ ] Sidechain building
	* [ ] Full backbone from CA
	* [ ] Fast loss functions and metrics
* [ ] Contribute Rosetta Scripts (contact me by email/discord to get a key for Rosetta if interested in doing this part. )

* **NOTES**: 
* Use functionality provided in [MP-NeRF](https://github.com/EleutherAI/mp_nerf) wherever possible (avoid repetition). 

## Contribute: 

Hey there! New ideas are welcome: open/close issues, fork the repo and share your code with a Pull Request. 

Currently, the main discussions / conversatino about the model development is happening [in this discord server](https://discord.gg/VpPpa9EZ) under the `/self-supervised-learning` channel.  

Clone this project to your computer:

`git clone https://github.com/EricAlcaide/pysimplechain`

Please, follow [this guideline on open source contribtuion](https://numpy.org/devdocs/dev/index.html) 

## Citations:

```bibtex
@article {Chowdhury2021.08.02.454840,
    author = {Chowdhury, Ratul and Bouatta, Nazim and Biswas, Surojit and Rochereau, Charlotte and Church, George M. and Sorger, Peter K. and AlQuraishi, Mohammed},
    title = {Single-sequence protein structure prediction using language models from deep learning},
    elocation-id = {2021.08.02.454840},
    year = {2021},
    doi = {10.1101/2021.08.02.454840},
    publisher = {Cold Spring Harbor Laboratory},
    URL = {https://www.biorxiv.org/content/early/2021/08/04/2021.08.02.454840},
    eprint = {https://www.biorxiv.org/content/early/2021/08/04/2021.08.02.454840.full.pdf},
    journal = {bioRxiv}
}

```

