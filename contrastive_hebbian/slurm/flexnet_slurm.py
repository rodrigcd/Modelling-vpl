import os, sys
# add the path to the directory containing the flexnet module to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scrips.curr_flexnet import main, all_regimes
import argparse
import pickle
from vpl_model.utils import check_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--run-id', type=int, default=0
    )
    parser.add_argument(
        '--save-path', type=str, default="../all_results/neural_data_net_slurm/"
    )
    parser.add_argument(
        '--exp-name', type=str, default="flexnet"
    )

    args = parser.parse_args()
    args = vars(args)
    run = args["run_id"]
    save_path = args["save_path"]
    check_dir(save_path)

    config_list = []
    for key in all_regimes.keys():
        config_list.append({"model_id": key})

    config_id = int(run % len(config_list))
    key = config_list[config_id]["model_id"]
    seed = run // len(config_list)

    print("#####")
    print("seed", seed)
    print("key", key)
    print("#####")

    results, params = main(model_id=key, seed=seed)
    pickle.dump(results, open(save_path + key + "_run_" + str(seed) + ".pkl", "wb"))
    pickle.dump(params, open(save_path + key + "_params_run_" + str(seed) + ".pkl", "wb"))