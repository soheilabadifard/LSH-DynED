from Ensemble import Ensemble
from Model.data_related_functions import *
from Model.classifier_monitor import Metrics, check_true, SlidingWindowForMetrics
from tqdm import tqdm
import pandas as pd
import warnings
import random
import os

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)
pd.set_option("display.max_rows", None)

random.seed(101)
num_cpus = os.cpu_count()
print(f"Number of CPUs: {num_cpus}")


def normalize_dataset_names(dataset_names):
    normalized = []
    for name in dataset_names:
        name = name.strip()
        if not name:
            continue
        if not name.lower().endswith('.arff'):
            name = f"{name}.arff"
        normalized.append(name)
    return normalized


def select_requested_datasets(available_datasets, requested_datasets):
    available_map = {dataset.lower(): dataset for dataset in available_datasets}
    selected = []
    missing = []
    for dataset in normalize_dataset_names(requested_datasets):
        matched = available_map.get(dataset.lower())
        if matched:
            selected.append(matched)
        else:
            missing.append(dataset)
    return selected, missing


def main(mode='multi', src='alberto'):

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rt = os.path.join(project_root, "Imbalance Datasets")
    ot = os.path.join(project_root, "new_res")
    if mode == 'multi':
        if src == 'alberto':
            root_path = os.path.join(rt, "arff-datasets-multiclass", "Alberto_data")
            out_root_path = os.path.join(ot, "arff-datasets-multiclass", "Alberto_data")
        elif src == 'keel':
            root_path = os.path.join(rt, "arff-datasets-multiclass", "KEEL_imbalanced_multiclass")
            out_root_path = os.path.join(ot, "arff-datasets-multiclass", "KEEL_imbalanced_multiclass")
    elif mode == 'binary':
        if src == 'alberto':
            root_path = os.path.join(rt, "arff-datasets-binary", "Alberto_data")
            out_root_path = os.path.join(ot, "arff-datasets-binary", "Alberto_data")
        elif src == 'keel':
            root_path = os.path.join(rt, "arff-datasets-binary", "KEEL_imbalanced_binary")
            out_root_path = os.path.join(ot, "arff-datasets-binary", "KEEL_imbalanced_binary")
    elif mode == 'semi-synthetic':
        root_path = os.path.join(rt, "arff-multi-class-semi-synthetic", "Alberto_data_new")
        out_root_path = os.path.join(ot, "arff-multi-class-semi-synthetic", "Alberto_data_new")
    elif mode == 'synthetic':
        root_path = os.path.join(rt, "arff-multi-class-synthetic", "new")
        out_root_path = os.path.join(ot, "arff-multi-class-synthetic", "new")
    else:
        raise ValueError("Mode should be either multi, binary or synthetic")

    components = 10
    train = 20
    test = 50
    vector = 5
    sampling_mode = 'faiss_lsh_majority'

    requested_datasets = []  # empty: run every .arff found in the source folder

    out_root_path = os.path.join(out_root_path, f"config_com{components}_train{train}_test{test}")
    os.makedirs(out_root_path, exist_ok=True)

    available_datasets = [dataset for dataset in os.listdir(root_path) if dataset.lower().endswith('.arff')]
    if requested_datasets:
        selected_datasets, missing_datasets = select_requested_datasets(available_datasets, requested_datasets)
    else:
        selected_datasets, missing_datasets = sorted(available_datasets), []

    if missing_datasets:
        print(f"Not found in this source ({mode}/{src}): {missing_datasets}")

    if not selected_datasets:
        print("No requested datasets were found in the selected source folder.")
        return

    pending_datasets = [
        dataset for dataset in selected_datasets
        if not os.path.exists(os.path.join(out_root_path, f"{dataset}_prequential.csv"))
    ]

    print(f"Requested datasets: {len(requested_datasets)}")
    print(f"Found in source: {len(selected_datasets)} -> {selected_datasets}")
    print(f"Pending (not processed yet): {len(pending_datasets)} -> {pending_datasets}")

    for dataset in pending_datasets:
        print(f"Dataset: {dataset}")
        stream = read_data_2(os.path.join(root_path, dataset))
        label_list = stream.classes
        print(f'Number of classes {len(label_list)}')
        print(f'Number of samples {stream.n_samples}')
        print(f'Number of features {stream.n_features}')

        # dataframe to store the statistics of the model
        clms = ['counter', 'avg_geo', 'recall', 'precision', 'f1', 'kappa']
        for cls1 in label_list:
            for cls2 in label_list:
                clms.append(f'CM[{cls1}][{cls2}]')
        df1 = pd.DataFrame(columns=clms, dtype=float)
        prequential_metrics = SlidingWindowForMetrics(label_list)
        stream_true = 0
        pred = []
        true = []
        model = Ensemble(num_components=components, class_list=label_list,
                         num_features=stream.n_features + 1,
                         max_pool_size=100, wind_size=1000,
                         ftr=stream.features, size_to_train=train,
                         size_to_test=test, vector=vector,
                         sampling_mode=sampling_mode)

        for counter, (x, y) in tqdm(enumerate(stream)):
            prediction = model.predict(x, dy=y)
            pred.append(prediction)
            true.append(y)
            stream_true = stream_true + check_true(y, prediction)
            model.set_values_for_lamda_update(stream_true / (counter + 1))
            prequential_metrics.add_to_window(prediction, y)
            if (counter + 1) % 500 == 0:
                mn = Metrics(actual=true, predicted=pred, cls_list=label_list)
                a = mn.get_scores()
                a['counter'] = counter + 1
                df1 = pd.concat([df1, pd.DataFrame([a])], ignore_index=True)
                pred = []
                true = []
            elif (counter + 1) == stream.n_samples:
                mn = Metrics(actual=true, predicted=pred, cls_list=label_list)
                a = mn.get_scores()
                a['counter'] = counter + 1
                df1 = pd.concat([df1, pd.DataFrame([a])], ignore_index=True)
        df1.to_csv(os.path.join(out_root_path, f"{dataset}.csv"), index=False)
        prequential_metrics.result_df.to_csv(os.path.join(out_root_path, f"{dataset}_prequential.csv"), index=False)
        print(f"number of drifts: {model.drift_detected_count}")
        print("completed")
        print("normal evaluation")
        print("---------------------------------------------------")
        print("prequential evaluation")
        print("---------------------------------------------------")


if __name__ == "__main__":
    main('multi', 'alberto')
    main('semi-synthetic', 'alberto')
