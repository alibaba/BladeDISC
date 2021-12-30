# DISC optimization example for TensorFlow training.

This repository provides a script showing how to optimize a DeePMD training
model with DISC just-in-time.


## Run

Install DeePMD-kit according to
https://github.com/deepmodeling/deepmd-kit/blob/master/doc/install/install-from-source.md.
Then execute the following commands.

```bash
# Download data:
sh download_data.sh

# Run:
sh run.sh
```

Note the dataset downloaded with `download_data.sh` is cloned from
https://github.com/deepmodeling/deepmd-kit.git.


## Performance results.

We evaluate this example on V100 GPU. The execution time of 100 training steps
is as following.

| TensorFlow  |    DISC    |
|-------------|------------|
|   1.89 s    |    1.52s   |

DISC shows a 1.24x speedup over TensorFlow.
