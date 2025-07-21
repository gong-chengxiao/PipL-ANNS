# PipL-ANNS
We build the code repository from a fork of [SymphonyQG](https://github.com/gouyt13/SymphonyQG). We keep the original directory and code structure.

## Directory structure
```
../
├── data/               # datasets and indices
├── symqglib/           # symphonyqg library
|   ├── qg/             # quantized graph
|   ├── quantization/   # quantization methods (RaBitQ)
|   ├── space/          # distance functions
|   ├── third/          # third party dependency
|   └── utils/          # common utils
├── python/             # python bindings
└── reproduce/          # code for reproduction
```

## Build
We use a python interface to build and run.
```bash
cd python
pip install -r requirements.txt
bash build.sh
```

## Prepare data
Please refer to [data/README.md](./data/README.md) for the preparation of data.

## Run
Please refer to [reproduce/README.md](./reproduce/README.md) for the reproduction of results.