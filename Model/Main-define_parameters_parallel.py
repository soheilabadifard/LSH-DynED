from Ensemble import Ensemble
from Model.data_related_functions import *
from Model.classifier_monitor import Metrics, check_true, SlidingWindowForMetrics
from tqdm import tqdm
import pandas as pd
import warnings
import random
import os
from concurrent.futures import ProcessPoolExecutor

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('always')
pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)
pd.set_option("display.max_rows", None)

random.seed(101)


def process_test_sizes(args):
    dataset, num_components, size_to_train, size_to_test, vector, root_path = args
    try:
        stream = read_data_2(root_path + dataset)
        label_list = stream.classes
        model = Ensemble(num_components=num_components, class_list=label_list,
                         num_features=stream.n_features + 1, max_pool_size=500,
                         wind_size=1000, ftr=stream.features, size_to_train=size_to_train,
                         size_to_test=size_to_test, vector=vector)
        prequential_metrics = SlidingWindowForMetrics(label_list)
        stream_true = 0
        pred = []
        true = []
        metrics_results = []
        for counter, (x, y) in enumerate(stream):
            prediction = model.predict(x, dy=y)
            stream_true = stream_true + check_true(y, prediction)
            model.set_values_for_lamda_update(stream_true / (counter + 1))
            prequential_metrics.add_to_window(prediction, y)
            pred.append(prediction)
            true.append(y)
            if (counter + 1) % 500 == 0 or (counter + 1) == stream.n_samples:
                mn = Metrics(actual=true, predicted=pred, cls_list=label_list)
                scores = mn.get_scores()
                scores['counter'] = counter + 1
                metrics_results.append(scores)
                pred = []
                true = []
                del mn
        df1 = pd.DataFrame(metrics_results)
        df1.to_csv(f"{root_path}{dataset}_{num_components}_{size_to_train}_{size_to_test}_{vector}.csv", index=False)
        a = {
            'dataset': dataset, 'num_components': num_components, 'train_size': size_to_train,
            'test_size': size_to_test, 'vector_size': vector, 'avg_geo': df1['avg_geo'].mean(),
            'recall': df1['recall'].mean(), 'precision': df1['precision'].mean(), 'f1': df1['f1'].mean(), 'kappa': df1['kappa'].mean()
        }
        prequential_metrics.result_df.to_csv(f"{root_path}{dataset}_{num_components}_{size_to_train}_{size_to_test}_{vector}_prequential.csv", index=False)
        del metrics_results
        del df1
        del stream
        del model
        del label_list
        del pred
        del true
        del prequential_metrics
        return a
    except Exception as e:
        print(
            f"Error processing {dataset}, Component {num_components}, Train {size_to_train}, Test {size_to_test}, vector {vector}: {e}")
        return None


def main():
    num_cpus = os.cpu_count()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root_path = os.path.join(project_root, "Imbalance Datasets",
                             "arff-multi-class-semi-synthetic", "Alberto_data_new") + os.sep
    # the search reported in Hyperparameter_results/ (135 combinations x 4 datasets)
    dataset_names = ['ACTIVITY-D1.arff', 'DJ30-D1.arff', 'GAS-D1.arff', 'TAGS-D1.arff']
    filtered_list = [file for file in dataset_names if not file.endswith(".csv")]
    components = [5, 10, 15]
    train_sizes = [20, 50, 100]
    test_sizes = [50, 100, 200]
    vector_sizes = [2, 3, 4, 5, 6]
    out_dir = os.path.join(project_root, "Hyperparameter_results")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "hyperparameter_search_results.csv")
    for dataset in filtered_list:
        for num_components in tqdm(components, leave=False):
            for size_to_train in tqdm(train_sizes, leave=False):
                for size_to_test in tqdm(test_sizes, leave=False):
                    #for v in tqdm(vector_sizes, leave=False):
                    tasks = [(dataset, num_components, size_to_train, size_to_test, v, root_path)
                             for v in vector_sizes]
                    parameter_table = pd.DataFrame()
                    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
                        results = list(
                            tqdm(executor.map(process_test_sizes, tasks), total=len(tasks), desc="Test Sizes", leave=False))
                        parameter_table = pd.concat([parameter_table, pd.DataFrame(results)], ignore_index=True)

                    # Append results to the CSV after each train size is processed
                    if os.path.exists(csv_path):
                        parameter_table.to_csv(csv_path, mode='a', header=False, index=False)
                    else:
                        parameter_table.to_csv(csv_path, mode='w', header=True, index=False)

                    print(f"Completed: Component {num_components}, Train Size {size_to_train}, Test Size {size_to_test}")


if __name__ == "__main__":
    main()
