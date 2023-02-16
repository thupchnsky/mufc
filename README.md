# Provably Accurate Federated Clustering with Unlearning Mechanism
An efficient method for federated (K-means) clustering and its corresponding unlearning procedure, which is introduced in our paper:

- [ICLR 2023] [Machine Unlearning of Federated Clusters](https://openreview.net/pdf?id=VzwfoFyYDga)

# Datasets
`Celltype`, `Gaussian`, `Postures`, `Covtype` can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1LqazOJuH3uOgFxHtBodwon6htEE2Wq13) provided by the authors of [DC-Kmeans](https://arxiv.org/abs/1907.05012). `FEMNIST` can be downloaded from the [Leaf Project](https://leaf.cmu.edu/). `TCGA` and `TMI` may contain potentially sensitive biological data and can be downloaded after logging into the databases ([TCGA](https://www.cancer.gov/about-nci/organization/ccg/research/structural-genomics/tcga), [TMI](https://commonfund.nih.gov/hmp)). We can provide the data processing pipelines upon reasonable requests via emails.

We also provide a utility function `split_data` in `utils.py` to split the data for clients in federated setting. One example of the `Celltype` dataset after data preprocessing is included in this repository.

# Usage
Two other methods, [DC-Kmeans](https://arxiv.org/abs/1907.05012) and [K-FED](https://arxiv.org/abs/2103.00697), are also implemented in this repository for comparison.

To run the methods on the example dataset, you can use the following command
```
python mufc_main.py --num_clusters=4 --num_clients=100 --data_path=celltype_processed.pkl --num_removes=10 \
                    --k_prime=4  --split=non-iid  --compare_kfed --compare_dc --client_kpp_only --verbose --update_centralized_loss
```
or simply run the shell script
```
chmod +x run.sh
./run.sh
```

# Contact
Please contact Chao Pan (chaopan2@illinois.edu), Saurav Prakash (sauravp2@illinois.edu) if you have any question.

# Citation
If you find our code or work useful, please consider citing our paper:
```
@inproceedings{
pan2023machine,
title={Machine Unlearning of Federated Clusters},
author={Chao Pan and Jin Sima and Saurav Prakash and Vishal Rana and Olgica Milenkovic},
booktitle={International Conference on Learning Representations},
year={2023},
url={https://openreview.net/forum?id=VzwfoFyYDga}
}
```
