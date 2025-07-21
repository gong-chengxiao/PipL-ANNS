# Dataset
  * Medium-scale datasets are available at https://www.cse.cuhk.edu.hk/systems/hash/gqr/datasets.html
    * Refer to https://huggingface.co/datasets/Qdrant/arxiv-abstracts-instructorxl-embeddings for the arXiv dataset
  * To download Deep-100M and MSTuring-100M, please refer to https://github.com/harsha-simhadri/big-ann-benchmarks
  * For reading/writing `.fvecs`/`.ivecs` files or reading `.fbin`/`.ibin` files, please refer to `./reproduce/utils/io.py`
    * The information of formats `.fvecs`/`.ivecs` can be found in http://corpus-texmex.irisa.fr/.

### Dataset preprocessing
  * For medium-scale datasets (including sift, gist, msong, deep1M, tiny5m and arXiv), please download and uncompress the package, then move the folder to `./data/`
    * For datasets without groundtruth of KNN, please refer to `./reproduce/compute_gt.py` to get the groundtruth on your own
  * For Deep-100M, after downloading the dataset, please transfer the format from `.fbin`/`.ibin` to `.fvecs`/`.ivecs` by using tools provided in `./reproduce/utils/io.py` for the simplicity of using the test code
  * After preprocessing, each dataset's folder will contain 3 files. For example, `./data/sift/` will contain `sift_base.fvecs` , `sift_query.fvecs` and `sift_groundtruth.ivecs`
