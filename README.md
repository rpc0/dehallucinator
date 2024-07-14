# DeHallucinator

Add description information here.

[![python](https://img.shields.io/badge/Python-3.9-3776AB?logo=Python&logoColor=white)](https://python.org/)

## Quickstart

### For Users

You can include this library in your requirements.txt file via:

```sh
git+ssh://git@github.com/rpc0/dehallucinator.git#egg=dehullicinator
```

A specific version of the library can be installed as shown below:

```sh
git+ssh://git@github.com/rpc0/dehallucinator.git@v0.1.2#egg=dehullicinator
```

Both options can be used to simply install this library in your environment
(created with tool like `conda` or `venv`):

```sh
pip install --upgrade pip
pip install git+ssh://git@github.com/rpc0/dehallucinator.git#egg=dehullicinator
```

### For Developers

You need to create a Python virtual environment for this project. We recommend you install and use
[Miniforge](https://github.com/conda-forge/miniforge). Once installed, create the environment:

```sh
conda create -y --name=dehullicinator python=3.9
```

activate it:

```sh
conda activate dehullicinator
```

upgrade your pip installation:

```sh
pip install --upgrade pip
```

Finally, install the project in "development mode" (note, you must be in the project's toplevel directory
for issuing this command):

```sh
pip install -e ".[dev]"
```

The command above works on Windows, macOS and Linux.
