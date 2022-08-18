Generalizing Downsampling from Regular Data to Graphs
-----------------------------------------------------

### Installation ###

You can install the `kmis` environment and the required packages with
Anaconda/Miniconda, by running the following commands in the project
directory:
```shell
conda env create -f conda/environment_cu113.yml
# Alternatively, for CPU-only:
# conda env create -f conda/environment_cpu.yml
conda activate kmis
```

### Available commands ###

We provide the following commands, all of which can be executed by running
```shell
python -m benchmark [COMMAND] [OPTIONS]
``` 
in the project directory:

| Command       | Options (example)                                                 | Description                                                                                                   |
|:--------------|:------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------|
| `info`        | `info --root datasets/`                                           | Prints a description of the datasets.                                                                         |
| `train`       | `train --model KMISPool --dataset DD --config "{'k':1}" --test`   | Train a model on a dataset with a given configuration.                                                        |
| `grid_search` | See `scripts/grid_search.sh`.                                     | Perform a grid search for a model and dataset in a given parameter space.                                     |
| `mnist`       | `mnist --k 1 --scorer lightness --reduction mean`                 | Plot the reduction of a diagonal grid constructed on a digit of the MNIST dataset.                            |
| `dist`        | `dist --k 1 --name minnesota --group Gleich`                      | Plot the distance distortions on a graph from the [Suite Sparse Matrix Collection](https://sparse.tamu.edu/). |
| `profile`     | `profile --k 1 --name europe_osm --group DIMACS10 --device cuda`  | Compute running time and reduction statistics.                                                                |
| `export`      | `export --name luxembourg_osm --group DIMACS10 --max_k 8`         | Export all the power graphs (up to `max_k`) in DIMACS92 format of a given graph.                              |

### Reproducibility ###

The results showed in the paper can be reproduced as following:

 - The reductions of the MNIST digit can be obtained by running the script 
   `./scripts/mnist.sh`
 - To reproduce the length distorsions on the Minnesota road network, run
   `python -m benchmark dist --k "[1,2,4,8]" --reduction None`.
 - To reproduce the classification benchmarks, run the script
   `./scripts/grid_search.sh` (**note:** it may take *long*).
 - To obtain the running time benchmarks, run the script 
   `./scripts/profile.sh` (**note:** the results may vary depending on the
   machine).
 - To reproduce the benchmarks on the total weight, run the script 
   `./scripts/weight.sh`. The results on the *sequential* greedy algorithm
   were computed using 
   [`stablesolver`](https://github.com/fontanf/stablesolver) (with 
   `-a greedy_gwmin` and `-a greedy_gwmin2` options), on power graphs
   previously exported using the `export` command (see above).