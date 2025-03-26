# Native installation instructions (without the need for docker)

We need older version of detectron2, and older version of pytorch.
Being guided by package selection and versions from Dockerfile.

Pyorch 1.10 requires Python 3.9 or older. Python 3.8 is used in Dockerfile.
Compiling detectron2 from source requires VS2019 or VS2017.

Start VS x64 command prompt, then:
```cmd
set DISTUTILS_USE_SDK=1
"C:\Program Files\Python38\python.exe" -m venv .venv38
.venv38\Scripts\activate
pip install --upgrade setuptools==69.5.1

pip install torch==1.10.1+cu113 -f https://download.pytorch.org/whl/torch/
pip install torchvision==0.11.2+cu113 -f https://download.pytorch.org/whl/torchvision/
pip install torchaudio==0.10.1 -f https://download.pytorch.org/whl/torchaudio/

pip install detectron2@git+https://github.com/facebookresearch/detectron2@v0.6
```
