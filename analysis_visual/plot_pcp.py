import glob
import pandas as pd
import matplotlib.pyplot as plt
from pareto import eps_sort
import numpy as np
from pcp_custom import PCP

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

markers = [
    "v", "s", "D", "*", "h", "^", "<", ">", "1", "2", "3", "4", 
    "p", "*", "h", "H", "+", "x", "D", "d", "|", "_", ".", ","]

map_obj={0: {'name':'parameter', 'weight': 'False' },
         1: {'name':'inference_speed', 'weight': 'False' },
         2: {'name':'precision', 'weight': 'True' },
         3: {'name':'recall', 'weight': 'True' },
}
def stepify_pareto_points_2d(x, y, metric_directions):
    """
    Returns the pareto front points, including the steps, from the x/y points of a 2D pareto front
    Takes in the optimization directions for the x and y parameters as a list of booleans (True=Minimized, False=Maximized)
    """

    is_x_minimized, is_y_minimimized = metric_directions
    is_x_minimized = eval(is_x_minimized)
    is_y_minimimized = eval(is_y_minimimized)
    # sort for pareto steps
    y_argsort = np.argsort(y)
    x = x[y_argsort]
    y = y[y_argsort]
    x_argsort = np.argsort(x)
    x = x[x_argsort]
    y = y[x_argsort]

    last_x, last_y = x[0], y[0]
    x_steps = [last_x]
    y_steps = [last_y]
    # step direction is based on the optimization direction of each axis
    for i, x_val in enumerate(x):
        y_val = y[i]
        if last_x != x_val:
            # add the stair step
            if is_x_minimized:
                y_steps.append(last_y)
                x_steps.append(x_val)
            else:
                y_steps.append(y_val)
                x_steps.append(last_x)
            # add the point
            y_steps.append(y_val)
            x_steps.append(x_val)
        last_x, last_y = x_val, y_val
    return x_steps, y_steps

def plot_2d_pareto_front(df, objectives=[2,3], maximize_ojb=[2,3], model='mixtral', columns='', suffixes='val'):
    
    scores = df[columns].values
    mins = scores.min(axis=0)
    maxs = scores.max(axis=0)

    x_min = mins[objectives[0]] * 0.95
    y_min = mins[objectives[1]] * 0.95
    x_max = maxs[objectives[0]] * 1.1
    y_max = maxs[objectives[1]] * 1.1
    plt.figure()
    for index, model in enumerate(['mixtral', 'Llama31', 'Llama33']):
        #model = "results"
        mask = df.file_path.apply(lambda x: model in x)
        selected_df = df[mask]
        
        scores = selected_df[columns].values
        inds = selected_df['file_path'].values
        ones_column = np.array(range(len(scores))).reshape(-1, 1)
        scores = np.hstack((scores, ones_column))
        nondominated_scores = np.array(eps_sort(scores, objectives=objectives, maximize=maximize_ojb))
        nondominated_df =  pd.DataFrame(nondominated_scores[:,0:-1], columns=columns)
        nondominated_df['file_path'] = inds[nondominated_scores[:, -1].astype('int')]
        nondominated_df['file_path'] = nondominated_df.file_path.apply(lambda x: x.split('/')[-1])
        
        
        x = nondominated_scores[:, objectives[0]]
        y = nondominated_scores[:, objectives[1]]
        metric_directions=[]
        for obj in objectives:
            metric_directions.append(map_obj[obj]['weight'])
        x_steps, y_steps = stepify_pareto_points_2d(x, y, metric_directions)

        #plt.scatter(scores[:, objectives[0]], scores[:, objectives[1]], alpha=0.8, label=model, marker=markers[index])
        #plt.scatter(scores[:, 0], scores[:, 1] *4 / 1024**2)
        if model == 'mixtral':
            model = 'Mixtral'
        elif model == 'Llama31':
            model = 'Llama-3.1'
        elif model == 'Llama33':
            model = 'Llama-3.3'
        plt.plot(x_steps, y_steps, label=f"Pareto Frontier-{model}",linewidth=3, marker=markers[index])

    plt.title(f"Pareto Fronts")
    plt.legend()
    #plt.scatter(scores[:, objectives[0]], scores[:, objectives[1]], alpha=0.5)
    #plt.scatter(scores[:, 0], scores[:, 1] *4 / 1024**2)
    #plt.plot(x_steps, y_steps, label="Pareto Frontier", color="xkcd:blue")
    # for marker_id, item in enumerate(baselines):
    #     values = baselines[item]
    #     plt.plot(values[objectives[0]], values[objectives[1]], marker=markers[marker_id], label=item)
    plt.legend(loc='lower right')
    plt.ylabel(map_obj[objectives[1]]['name'])
    plt.xlabel(map_obj[objectives[0]]['name'])
    plt.tight_layout()
    plt.grid()
    #plt.ylim([y_min, y_max])
    #plt.xlim([x_min, x_max])
    plt.savefig(f"{map_obj[objectives[0]]['name']}_{map_obj[objectives[1]]['name']}_{suffixes}.png")

def get_results(files, cols = ['parameter', 'inference_speed', 'precision', 'recall', 'mAP50(B)', 'mAP50-95(B)', 'file_path']):
    scores = []
    for results_path in files:
        with open(results_path, 'r') as file:
            results = file.read()
        results = results.split(',')
        fitness = [float(r.strip()) for r in results]
        
        fitness.append(results_path)
        if len(fitness) == len(cols):
            scores.append(fitness)
    df = pd.DataFrame(scores, columns=cols)
    return df

if __name__=='__main__':

    cols = ['parameter', 'inference_speed', 'precision', 'recall', 'mAP50(B)', 'mAP50-95(B)', 'file_path']
    objectives = ['parameter', 'inference_speed', 'precision', 'recall','mAP50(B)', 'mAP50-95(B)']
    objectives_optima = ['parameter', 'inference_speed', 'precision', 'recall','mAP50(B)', 'mAP50-95(B)']
    objectives_optima_vis = ['parameter', 'inference_speed', 'precision', 'recall','mAP50(B)', 'mAP50-95(B)']
    
    # load score from txt files
    files = glob.glob('/home/hice1/yzhang3942/scratch/llm-guided-evolution/sota/ultralytics/results/*.txt')
    df1 = get_results(files, cols = ['parameter', 'inference_speed', 'precision', 'recall', 'mAP50(B)', 'mAP50-95(B)'])
    df1['id'] = df1.file_path.apply(lambda x: x.split('/')[-1].split('_')[0])
    df1['file_path'] = df1.file_path.apply(lambda x: x + "_Llama33")

    # compute Pareto fronts
    df = df1
    df = df[df.fps >0]
    scores = df[objectives_optima].values
    inds = df['file_path'].values
    ones_column = np.array(range(len(scores))).reshape(-1, 1)
    scores = np.hstack((scores, ones_column))
    nondominated_scores = np.array(eps_sort(scores, objectives=[0, 1, 2, 3], maximize=[2,3]))
    nondominated_df =  pd.DataFrame(nondominated_scores[:,0:-1], columns=objectives_optima)
    nondominated_df['file_path'] = inds[nondominated_scores[:, -1].astype('int')]
    nondominated_df['file_path'] = nondominated_df.file_path.apply(lambda x: x.split('/')[-1])

    print(nondominated_df.sort_values('AssA'))
    
    # flip objectives
    for  item in ['parameter', 'inference_speed', 'gflops', 'latency']:
        nondominated_df[item] =   - nondominated_df[item] 

    df['id'] = df.file_path.apply(lambda x: x.split('/')[-1].split('_')[0])
    print(df.loc[df['id'] == 'v3',objectives_optima ].values)
    for index, item in enumerate(['Llama33']):
        mask = nondominated_df.file_path.apply(lambda x: (item in x) or ("v3" in x))
        temp1 = nondominated_df.loc[mask, objectives_optima_vis].values
    baseline_models = ['v3', 'v3spp', 'v3tiny', 'v9s', 'v10tiny', 'v11']
    temps = temp1

    ####
    plot = PCP(title=("Pareto Front Individuals of Holdout Data", {'pad': 30}),
           legend=(True, {'loc': "lower right"}),
           normalize_each_axis=True,
           reverse=True,
           figsize = (15, 8),
           reverse_index=[0, 1],
           labels=objectives_optima_vis)
    
    plot.bounds = [temps.min(axis=0), temps.max(axis=0)]


    plot.set_axis_style(color="grey", alpha=0.5)
    # plot pareto front
    for index, item in enumerate([ 'Llama33']):
        mask = nondominated_df.file_path.apply(lambda x: item in x)
        F = nondominated_df.loc[mask, objectives_optima_vis].values
        print(item, len(F))
        #if item == 'Llama3.3':
        plot.add(F, alpha=0.8, linewidth=2, marker=markers[index])

    data_baseline = df.loc[df['id'] == 'v3', objectives_optima_vis].values
    data_baseline[:, 0:2] = - data_baseline[:, 0:2]

    # collection inds better than baseline
    good_inds = []
    for id_index, item in enumerate(df.id.values.tolist()):
        data = df.loc[df['id'] == item, objectives_optima_vis].values
        
        for index in [0, 1]:
            data[:, index] = - data[:, index]
        #data[:, 0:2] = - data[:, 0:2]
        if (data_baseline < data).all():
            good_inds.append(item)

    # plot baseline
    for item in ['v3', 'v11', 'v10tiny', 'v3tiny', 'v9s', 'v3spp']:
        data = df.loc[df['id'] == item, objectives_optima_vis].values
        for index in [0, 1]:
            data[:, index] = - data[:, index]
        #data[:, 0:2] = - data[:, 0:2]
        plot.add(data, linewidth=3, label=f'YOLO_{item}', marker=markers[index+1])

    # plot inds
    for id_index, item in enumerate(['xXx5OgxZlhiOFa82oltrpCF1Vif', 'xXxj1Gr22K2oEXkXtman4NVeEK8']):
            data = df.loc[df['id'] == item, objectives_optima_vis].values
            print(item)
            print(df.loc[df['id'] == item, objectives_optima_vis])
            for index in [0, 1]:
                data[:, index] = - data[:, index]
            #data[:, 0:2] = - data[:, 0:2]
            plot.add(data, linewidth=5, label=f'Example {id_index+1}', marker=markers[index+1])
    plot.show()
    plt.savefig("pcp_holdout_test_0123.png")
    
