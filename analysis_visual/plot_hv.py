from pymoo.indicators.hv import HV
import pickle
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def load_pickle(file_name):
    with open(file_name, 'rb') as file:
        loaded_list = pickle.load(file)
    return loaded_list

def get_min_max(scores_dict):
    scores = np.vstack([scores_dict[i] for i in range(len(scores_dict))])
    mins = scores.min(axis=0)
    maxs = scores.max(axis=0)
    return mins, maxs

def compute_hvs(loaded_list, normalize_factor, ref_point=[1,1,1,1]):
    ind_hv = HV(ref_point=ref_point)
    hvs = []
    min_value = 1
    for index, item in enumerate(loaded_list):
        for i in [0, 1, 6, 7, 8, 9]:
            item[:, i] = item[:, i] / normalize_factor[i]
        # for i in [1]:
        #     item[:, i] /= normalize_factor[i]
        for i in [2, 3, 4,5, 6, 7, 9]:
            item[:, i] = 1 - item[:, i]
        temp_hv = 1 - ind_hv(item)
        print(f"[{index}, {temp_hv}],")
        if min_value >= temp_hv:
            hvs.append(temp_hv)
            min_value = temp_hv
        else:
            hvs.append(min_value)
    return hvs

map_obj={0: {'name':'parameter', 'weight': 'False' },
            1: {'name':'inference_speed', 'weight': 'False' },
            2: {'name':'precision', 'weight': 'True' },
            3: {'name':'recall', 'weight': 'True' },
            4: {'name':'mAP50', 'weight': 'True' },
            5: {'name':'mAP50-90', 'weight': 'True' },
            6: {'name':'gflops', 'weight': 'True' },
            7: {'name':'fps', 'weight': 'True' },
            8: {'name':'latency', 'weight': 'False' },
            9: {'name':'AssA', 'weight': 'True' },
    }

if __name__ == '__main__':
    loaded_list_last = load_pickle("pareto_fronts_33_new.pkl")
    total = []

    total.extend(loaded_list_last)
    mins, maxs = get_min_max(total)

    normalize_factor = [maxs[0], maxs[1], 1, 1, 1, 1, maxs[6], maxs[7], maxs[8], maxs[9]]
    hv4 = compute_hvs(loaded_list_last, normalize_factor, ref_point=[1]*10)

    markers = [
    ".", ",", "o", "v", "s", "^", "<", ">", "1", "2", "3", "4", 
    "p", "*", "h", "H", "+", "x", "D", "d", "|", "_"]
    plt.figure()
    count = 2
    for label, item in zip([ 'Llama 3.3 GE + optima'],[hv4]):
        plt.plot(range(len(item)), item, marker=markers[count], markersize=4, label=label)
        count +=1
    plt.legend()
    plt.xlabel("Generation")
    plt.ylabel("Hypervolume")
    plt.grid()
    plt.savefig('HV_all.png')