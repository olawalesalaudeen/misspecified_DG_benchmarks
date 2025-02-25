# Misspecified Domain Generalization Benchmarks

Our experiments are conducted using a modification of the [DomainBed](https://github.com/facebookresearch/DomainBed) library. Notably, this our modification includes additional datasets and model architectures than the original domainbed library. The model architectures also include transfer learning and finetuning.

Datasets:
- WILDSCamelyon (Bandi et al., 2018; Koh et al.,2021)
- CivilComments (Borkan et al., 2019; Koh et al., 2021)
- ColoredMNIST (Arjovsky et al., 2019; Gulrajani & Lopez-Paz, 2020a)
- Covid-CXR (Alzate-Grisales et al., 2022; Cohen et al., 2020b; Tabik et al., 2020; Tahir et al., 2021; Suwalska et al., 2023)
- WILDSFMoW (Christie et al., 2018; Koh et al., 2021)
- PACS (Liet al., 2017; Gulrajani & Lopez-Paz, 2020a)
- Spawrious (Lynch et al., 2023)
- TerraIncognita (Beery et al., 2018; Gulrajani & Lopez-Paz, 2020a)
- Waterbirds (Sagawa et al., 2019)

Model Architectures:
- ResNet-18/50 (He et al., 2016)
- DenseNet-121 (Huang et al., 2017)
- Vision Transformers (Dosovitskiy et al., 2020)
- ConvNeXt-Tiny (Liu et al., 2022)


## Results
Our results only include accuracies for the 'out' split of each domain. Our results include two versions:
- x-axis: source domain accuracies individually, y-axis: target domain accuracy individually
- x-axis: average source domain accuracies, y-axis: target domain accuracy individually

## Running the experiments
To run the experiments, use the following command:
```
python sweep.py --datasets <dataset_names> --algorithms <algorithm_names> --n_hparams <n_hparams> --n_trials <n_trials> --model_arch <model_arch>
```

Example:
```
python sweep.py --datasets TerraIncognita --algorithms ERM --n_hparams 25 --n_trials 1 --model_arch vit_b_16
```

