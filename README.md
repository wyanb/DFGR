# DFGR
We express our gratitude for the great work: [A Unified Model for Multi-class Anomaly Detection](https://arxiv.org/abs/2206.03687), and [Fastflow](https://github.com/gathierry/FastFlow).

## Quick Start

Download the MVTec-AD dataset from [here](https://www.mvtec.com/company/research/datasets/mvtec-ad). 

```
|-- data
    |-- MVTec-AD
        |-- mvtec_anomaly_detection
        |-- train.json
        |-- test.json
```

```
|-- data
    |-- MVTec-DAGM-Rollei
        |-- anomaly_detection
            |-- bottle
            |-- cable
            |-- capsule
            |-- carpet
            |-- grid
            |-- hazelnut
            |-- leather
            |-- metal_nut
            |-- pill
            |-- screw
            |-- tile
            |-- toothbrush
            |-- transistor
            |-- wood
            |-- zipper
            |-- Class1
            |-- Class2
            |-- Class3
            |-- Class4
            |-- Class5
            |-- Class6
            |-- Class7
            |-- Class8
            |-- Class9
            |-- Class10
            |-- Rollei   
        |-- train.json
        |-- test.json
```

- **Train or test** by running: 

python train_val.py

python train_val.py -e


