# Installation Guide
Clone repository:

```
git clone git@github.com:ahoelzemann/hangtime_har.git
cd hangtime_har
```

Create [Anaconda](https://www.anaconda.com/products/distribution) environment:

```
conda create -n hangtime_har python==3.11.3 
conda activate hangtime_har
```

Install PyTorch distribution:

```
conda install pytorch==1.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

Install other requirements:
```
pip install -r requirements.txt
```
