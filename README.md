## Visible-Light-Guided Infrared Image Super-Resolution with Dual Amplitude-Phase Optimization (vap-SR) 



## Setup

### Environment


* Download the repo and setup the environment with:

```bash
conda env create -f environment.yml
conda activate vapSR
```


### Dataset
 
We reconstructed ODinMJ dataset for vap-SR training and test. The reconstructed ODinMJ datasets can be downloaded from [baidu pan](https://pan.baidu.com/s/1gu7yk4oytT5avXC2taRZsw?pwd=3k8a), password:3k8a.

You can find the original ODinMJ dataset from https://github.com/KustTeamWQW/ODinMJ-RGB-T-Dataset, and original VGTSR dataset from https://github.com/mmic-lcl/Datasets-and-benchmark-code.

## Training

We provide an example for training vap-SR on the ODinMJ dataset. Please modify the paths in `train.sh` and run command:

  ```bash
  sh train.sh
  ```



## Testing

We provide an example for testing vap-SR on the ODinMJ dataset. Please modify the paths in `generate.sh` and run command:

  ```bash
  sh generate.sh
  ```


