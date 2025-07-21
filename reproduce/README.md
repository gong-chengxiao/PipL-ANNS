# Reproduction
Here, we provide the code for reproducing main experimental results reported in the paper. Please make sure datasets have been downloaded and preprocessed.

* Edit variables in `./reproduce/settings.py` to set tested datasets and parameters for indexing and querying
* Run `python ./reproduce/indexing.py` to build and save indices
* Run `python ./reproduce/run.py` to test the query performance (QPS-recall trade-off) of saved indices, results will be saved as csv file
