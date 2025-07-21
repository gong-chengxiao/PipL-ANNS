import symphonyqg
import os
import gc
import numpy as np
import pandas as pd
from time import time
from utils.io import fvecs_read, ivecs_read
from utils.preprocess import normalize
from utils.memory import get_memory_usage
from settings import ROUND, TOPK, datasets, degrees, datasets_dir

if __name__ == "__main__":
    for DATASET in datasets.keys():
        DISTANCE = datasets[DATASET]

        base = fvecs_read(f"{datasets_dir}/{DATASET}/{DATASET}_base.fvecs")
        query = fvecs_read(f"{datasets_dir}/{DATASET}/{DATASET}_query.fvecs")
        gt = ivecs_read(f"{datasets_dir}/{DATASET}/{DATASET}_groundtruth.ivecs")

        NQ, D = query.shape
        N = base.shape[0]

        if DISTANCE == "angular":
            query = normalize(query)

        for DEGREE in degrees[DATASET]:
            m1 = get_memory_usage()

            index_path = f"{datasets_dir}/{DATASET}/symphonyqg_{DEGREE}.index"
            index = symphonyqg.Index(
                index_type="QG",
                metric="L2",
                num_elements=N,
                dimension=D,
                degree_bound=DEGREE,
            )
            index.load(index_path)

            m2 = get_memory_usage()
            MEMORY = m2 - m1

            EFS = [150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 900, 1000, 1200, 1500, 1800, 2000]

            ALL_QPS = []
            ALL_RECALL = []

            for _ in range(ROUND):
                QPS = []
                RECALL = []
                for EF in EFS:
                    total_time = 0
                    results = []
                    index.set_ef(EF)
                    for i in range(NQ):
                        t1 = time()
                        pred = index.search(query[i], TOPK)
                        t2 = time()
                        results.append(pred)
                        total_time += t2 - t1

                    total_num = NQ * TOPK
                    total_correct = 0
                    for i in range(NQ):
                        res_set = set(results[i])
                        for j in range(TOPK):
                            if gt[i][j] in res_set:
                                total_correct += 1

                    qps = NQ / total_time
                    recall = total_correct / total_num * 100
                    QPS.append(qps)
                    RECALL.append(recall)

                ALL_QPS.append(QPS)
                ALL_RECALL.append(RECALL)

            ALL_QPS = np.median(np.array(ALL_QPS), axis=0)
            ALL_RECALL = np.median(np.array(ALL_RECALL), axis=0)

            df = pd.DataFrame(
                {
                    "QPS": ALL_QPS,
                    "Recall": ALL_RECALL,
                    "EFS": EFS,
                    "Method": f"(PipL)symphonyqg{DEGREE}",
                    "Memory": MEMORY,
                }
            )

            res_dir = f"./results/{DATASET}/PipL_symphonyqg/"
            try:
                os.makedirs(res_dir)
            except OSError as e:
                print(e)
            df.to_csv(res_dir + f"PipL_symphonyqg{DEGREE}_{TOPK}.csv", index=False)

            del index
            del df
            gc.collect()
