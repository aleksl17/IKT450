# IKT450

IKT450 group.ai repository

## How to run

### Assumptions

Microsoft Windows 10 Pro or Education

PyCharm version 2020.x or newer.

Pip version 20.x or newer.

Pipenv version 2020.x or newer.

### Requirements

Download and install [Visual Studio 2015, 2017 and 2019 (x64: vc_redist.x64.exe)](https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads).

Download and install [Python 3.7.9 64-bit (Windows x86-64 executable installer)](https://www.python.org/downloads/release/python-379/).

Try to match the images below. If "py launcher" options are grayed out, ignore them.

![](https://i.imgur.com/5JYRAtc.png)
Check "Add Python 3.7 to PATH". Click "Custom Installation".

![](https://i.imgur.com/OIbtaU2.png)

![](https://i.imgur.com/cf1Iytq.png)
Check "Install for all users".

Make sure python, pip and pipenv are added to System Properties -> Advanced -> Environment Varaibles -> System variables -> Path: 
``C:\Program Files\Python37\``
``C:\Program Files\Python37\Scripts\``
``C:\Users\BromTeque\AppData\Roaming\Python\Python37\Scripts``


### Detailed Configuration

Clone the repository:
```
git clone git@github.com:aleksl17/IKT450.git
```

Open PyCharm, and create a new project with the following attributes.

![](https://i.imgur.com/mO8cbaZ.png)
Change "C:\\**BromTeque**\IKT450" to your own **username**.

![](https://i.imgur.com/lIRO417.png)
Click "Create from Existing Sources" if you get the following pop-up.

Open "Terminal" in PyCharm and write:
```
pipenv install
```

![](https://i.imgur.com/O6aEFaC.png)
This might take a while. Be patient.

### (OPTIONAL) GPU Accelleration Installation Guide

Download and install [CUDA 10.1 (Feb 2019)](https://developer.nvidia.com/cuda-10.1-download-archive-base?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal).

Download [Nvidia CuDNN SDK 7.6 (November 5th, 2019), for CUDA 10.1 (Nvidia Developer Account Required)](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/7.6.5.32/Production/10.1_20191031/cudnn-10.1-windows10-x64-v7.6.5.32.zip).

Exctract the unzipped CuDNN files to their repsective locations:
``CuDNN7.zip -> cuda\bin\cudnn64_7.dll`` to ``C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin\``
``CuDNN7.zip -> cuda\include\cudnn.h`` to ``C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\include\``
``CuDNN7.zip -> cuda\lib\x64\cudnn.lib`` to ``C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\lib\x64\\``
Important that these are correct. Please dobble check them.

(OPTIONAL - Not needed) Install [TensorRT 6.0.1](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-601/tensorrt-install-guide/index.html)