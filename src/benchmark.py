import numpy as np
from pyfrechet.metric_spaces import *
from pyfrechet.regression.trees import Tree
from pyfrechet.regression.bagged_regressor import BaggedRegressor
from pyfrechet.metrics import mse
from datetime import datetime
import sklearn
import time
import json

def bench_it(name, est, X_train, y_train, X_test, mx_test) -> dict:
    fitted = sklearn.clone(est)
    
    t0 = time.time()

    print(f'[{str(datetime.now())}] Distances for {name}')

    distances_duration = 0
    if est.precompute_distances:
        tt0 = time.time()
        y_train.compute_distances()
        distances_duration = time.time() - tt0

    print(f'[{str(datetime.now())}] dt = {distances_duration}')
    print(f'[{str(datetime.now())}] Fitting for {name}')

    fitted = fitted.fit(X_train, y_train)
    total_duration = time.time() - t0
    
    print(f'[{str(datetime.now())}] dt = {total_duration - distances_duration}')
    print(f'[{str(datetime.now())}] MSE for {name}')

    e = mse(mx_test, fitted.predict(X_test))
    return dict(total_duration=total_duration, distances_duration=distances_duration, mse=e)


def bench(
        gen_data,
        out_file,
        ps=[2, 5, 10, 20],
        Ns=[50, 100, 200, 400],
        min_split_sizes=[3, 5, 15],
        n_trees=100,
        subsample_fracs=[],
        replicas=50,
    ):
    results = []
    for N in Ns:
        for p in ps:
            for min_split_size in min_split_sizes:
                for subsample_frac in subsample_fracs:
                    for i in range(replicas):
                        print(f'[{str(datetime.now())}] Progress: N={N}\tp={p}\tmin_split_size={min_split_size}\ti={i}')
                        beta = np.random.randn(p)
                        alpha = np.random.randn()
                        X_train, y_train, _ = gen_data(N, p, alpha, beta)
                        X_test, _, mx_test = gen_data(50, p, alpha, beta)


                        params = dict(N=N, p=p, min_split_size=min_split_size, subsample_frac=subsample_frac)

                        params['method']='cart_2means'
                        cart_2means = BaggedRegressor(
                            Tree(impurity_method='cart', split_type='2means', is_honest=True, min_split_size=min_split_size),
                            n_estimators=n_trees,
                            bootstrap_fraction=subsample_frac
                        )
                        params.update(bench_it(params['method'], cart_2means, X_train, y_train, X_test, mx_test))
                        results.append(params.copy())

                        params['method']='medoid_2means'
                        medoid_2means = BaggedRegressor(
                            Tree(impurity_method='medoid', split_type='2means', is_honest=True, min_split_size=min_split_size),
                            n_estimators=n_trees,
                            bootstrap_fraction=subsample_frac
                        )
                        params.update(bench_it(params['method'], medoid_2means, X_train, y_train, X_test, mx_test))
                        results.append(params.copy())

                        params['method']='medoid_greedy'
                        medoid_greedy = BaggedRegressor(
                            Tree(impurity_method='medoid', split_type='greedy', is_honest=True, min_split_size=min_split_size),
                            n_estimators=n_trees,
                            bootstrap_fraction=subsample_frac
                        )
                        params.update(bench_it(params['method'], medoid_greedy, X_train, y_train, X_test, mx_test))
                        results.append(params.copy())

                        with open(out_file, 'w') as f:
                            json.dump(results, f)
