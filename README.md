# Wavebeat_IDLS24
# Setup TCN

Date: March 10, 2024 → March 12, 2024

# Setup Environment on GCP

1. Instal GPU driver

```jsx
curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py
sudo python3 install_gpu_driver.py
```

1. install anaconda

```jsx
*conda install -c anaconda git
git clone https://github.com/csteinmetz1/wavebeat.git*
```

1. create an new conda environment using python = 3.8
    
    ```jsx
    conda create --name wavebeat python=3.8
    conda activate wavebeat
    ```
    
2. Pre-install

```
sudo apt-get install libsndfile1 # soundfile issue
sudo apt-get install gfortran libopenblas-dev liblapack-dev
sudo pip install scipy==1.2.1
sudo apt-get install sox
```

1. Install torch 

```jsx
python3.8 -m pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f [https://download.pytorch.org/whl/torch_stable.html](https://download.pytorch.org/whl/torch_stable.html)
```

1. Install these packages first

```jsx
pip install numpy cython aiohttp
```

1. Install requirement.txt

# Reproduce

 1. open juputer notebook

```jsx
jupyter notebook --no-browser -port=8888
```

1. Open Reproduce.ipynb
    1. Run Testing Cell

# TCN validation test

## Problems

1. Not complete match of annotation and audiofiles

## Solutions

1. only use the file which is annotated

# Debug

[data.py](http://data.py) recover

# Cite

Steinmetz, C. J., & Reiss, J. D. (2021). "WaveBeat: End-to-end beat and downbeat tracking in the time domain." In 151st AES Convention.