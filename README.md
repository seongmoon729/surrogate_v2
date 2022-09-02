# Filtering Network for VCM

## Requirements
* nvidia GPU
* [nvidia-docker2](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)
* [docker compose](https://docs.docker.com/compose/)

## Installation
1. Download repository.
```bash
$ git clone https://github.com/seongmoon729/surrogate_v2.git
```

2. Go to the `surrogate_v2/` folder.
```bash
$ cd surrogate_v2/
```

3. Build docker image.
```bash
$ docker compose build
```

## Training / Evaluation (`run.py`)
* Run training script.
```bash
$ docker compose run --rm --name WHATEVER_YOU_WANT main \
  python run.py -g 0 train ...
```
* Run evaluation script.
```bash
$ docker compose run --rm --name WHATEVER_YOU_WANT main \
  python run.py -g 0 evaluate ...
```

## `run.py` script help option
You can use `--help` option to view available options.
```bash
$ docker compose run --rm --name WHATEVER_YOU_WANT main \
  python run.py --help
```
```bash
$ docker compose run --rm --name WHATEVER_YOU_WANT main \
  python run.py train --help
```
```bash
$ docker compose run --rm --name WHATEVER_YOU_WANT main \
  python run.py evaluate --help
```

## Visualization.