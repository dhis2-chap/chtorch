from chtorch.preferential_optimization import TuneDeepAR
import logging
import os
logger = logging.getLogger(__name__)

from pbmohpo.decision_makers.decision_maker import DecisionMaker
from pbmohpo.optimizers.eubo import EUBO
from pbmohpo.benchmark import Benchmark


if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    prob = TuneDeepAR(data_path=path)

    dm = DecisionMaker(objective_names=prob.get_objective_names(), seed=0)

    print("Decision Maker Preference Scores:")
    print(dm.preferences)

    opt = EUBO(prob.get_config_space())
    bench = Benchmark(
        prob, opt, dm, eval_budget=10, dm_budget=10, eval_batch_size=2, dm_batch_size=1
    )
    print("Running EUBO")
    bench.run()
 
