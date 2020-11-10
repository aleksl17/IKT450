# IKT450
IKT450 group.ai repository


## How to run
### Assumptions
PyCharm version 2019.x or newer
Pip version 20.x or newer
Pipenv version 2020.x or newer

### Requirements
Download [Visual Studio 2015, 2017 and 2019 (x64: vc_redist.x64.exe)](https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads). \
Download [Python 3.7.9 64-bit (Windows x86-64 executable installer)](https://www.python.org/downloads/release/python-379/). \
Make sure to check "Install for all user" and "Add to system envirnment variables / Path".

### Detailed Configuration
WiP

Installer pipfilen, kjør:
```
pipenv install
```
Når man skal installere pip package, bruk "pipenv install x" istedenfor "pip install x". Da blir packages lakt til i pipfilen, og det blir enklere for andre å manage packages.

### For [TensorFlow GPU](https://www.tensorflow.org/install/gpu) Her står det mye feil, ikke følg det! Må skrives om.
Krever nvidia GPU driver 418.x eller nyere. Kan lastes ned [her](https://www.nvidia.com/drivers) ved behov. \
Last ned [CUDA 10.1 (Feb 2019)](https://developer.nvidia.com/cuda-10.1-download-archive-base?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal). \
Pass på at CUPTI blir installert med CUDA og at CUDA blir lagt til i path/environment variables.

Unzip cudnn zip filen som ligger i prosjekt repositoriet (eller last ned [her](https://developer.nvidia.com/cudnn) (krever nvidia Developer account)) og flytt folderen i zip-filen til "C:/tools/" (lag tools folderen om den ikke eksisterer). Leggg så "C:\tools\cuda" til i system environmental variables PATH.
![add to path](https://i.imgur.com/nZwD49X.png)