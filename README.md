# <ins>F</ins>orward-<ins>L</ins>ooking <ins>A</ins>ctive <ins>RE</ins>trieval augmented generation (FLARE)

This repository contains the code and data for the paper
[Active Retrieval Augmented Generation](https://arxiv.org/abs/2305.06983).

## Overview

FLARE is a generic retrieval-augmented generation method that actively decides when and what to retrieve using a prediction of the upcoming sentence to anticipate future content and utilize it as the query to retrieve relevant documents if it contains low-confidence tokens.

<p align="center">
  <img align="middle" src="res/flare.png" height="350" alt="FLARE"/>
</p>

## Install environment with Conda
Create a conda env and follow `setup.sh` to install dependencies.

## Quick start

### Download Wikipedia dump
Download the Wikipedia dump from [the DPR repository](https://github.com/facebookresearch/DPR/blob/main/dpr/data/download_data.py#L32) using the following command:
```shell
wget -O data/dpr/psgs_w100.tsv https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
```

### Build index over the Wikipedia dump using Elasticsearch
```shell
wget -O elasticsearch-7.17.9.tar.gz https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.17.9-linux-x86_64.tar.gz  # download Elasticsearch
tar zxvf elasticsearch-7.17.9.tar.gz
pushd elasticsearch-7.17.9
nohup bin/elasticsearch &  # run Elasticsearch in background
popd
python prep.py --task build_elasticsearch --inp data/dpr/psgs_w100.tsv wikipedia_dpr  # build index
```

### Setup OpenAI keys
Put OpenAI keys in the `keys.sh` file.
Multiple keys can be used to accelerate experiments.
Please avoid uploading your keys to Github by accident!

### Run FLARE
Use the following command to run FLARE on the 2WikiMultihopQA dataset (500 examples) with `text-davinci-003`. Be careful, the experiment is relatively expensive because FLARE iteratively calls OpenAI APIs. To save credits, you can set `debug=true` to active the debugging mode which walks you through the process one example at a time, or you can decrease `max_num_examples` to run small-scale experiments.
```shell
./openai.sh 2wikihop configs/2wikihop_flare_config.json
```

## Citation
```
@article{jiang2023flare,
      title={Active Retrieval Augmented Generation}, 
      author={Zhengbao Jiang and Frank F. Xu and Luyu Gao and Zhiqing Sun and Qian Liu and Jane Dwivedi-Yu and Yiming Yang and Jamie Callan and Graham Neubig},
      year={2023},
      eprint={2305.06983},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
