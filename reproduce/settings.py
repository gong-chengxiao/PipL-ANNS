ROUND = 5   # round of testing query
TOPK = 100   # knn
EF = 400    # EF for indexing

datasets_dir = "../data"

datasets = {
    "sift": "l2",
    "deep1M": "l2",
    "gist": "l2",
    "msong": "l2",
    "tiny5m": "l2",
    "imagenet": "l2",
    "deep100M": "l2",
    "msturing100M": "l2"
}


degrees = {
    "sift": [32, 64],
    "deep1M": [32, 64],
    "gist": [32, 64],
    "msong": [32, 64],
    "tiny5m": [32, 64],
    "imagenet": [32, 64],
    "deep100M": [32],
    "msturing100M": [32],
}

iter = {
    "sift": 3,
    "deep1M": 3,
    "gist": 3,
    "msong": 3,
    "tiny5m": 3,
    "imagenet": 3,
    "deep100M": 3,
    "msturing100M": 4
}