from Ensemble import Ensemble
from Model.data_related_functions import *
from Model.classifier_monitor import Metrics
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


def main():
    root_path = ("/data/Soheil-data/Python Projects/DynED-Imb_v2/Imbalance "
                 "Datasets/arff-multi-class-synthetic/parameter_tuning/")
    dataset_names = ["Imb_RandomRBFD_5cls_10ft_30driftatt_0noise_speed500.arff"]
    print(f"Number of datasets: {len(dataset_names)}")
    print('datasets: ', dataset_names)
    components = [5, 10, 20]
    train_sizes = [50, 100, 200, 500]
    test_sizes = [20, 50, 100, 200, 500]

    parameter_table = pd.DataFrame(columns=['dataset', 'num_components', 'train_size', 'test_size', 'avg_geo', 'recall',
                                            'precision', 'f1', 'kappa'])

    for dataset in dataset_names:
        for num_components in components:
            for size_to_train in train_sizes:
                for size_to_test in test_sizes:
                    print(f"Dataset: {dataset}, num_components: {num_components}, train_size: {size_to_train}",
                          f"test_size: {size_to_test}")
                    stream = read_data_2(root_path + dataset)
                    label_list = stream.classes
                    print(
                        f'Number of classes {len(label_list)}, Number of features {stream.n_features}, Number of samples {stream.n_samples}')
                    # dataframe to store the statistics of the model
                    clms = ['counter', 'avg_geo', 'recall', 'precision', 'f1', 'kappa']
                    for cls1 in label_list:
                        for cls2 in label_list:
                            clms.append(f'CM[{cls1}][{cls2}]')
                    df1 = pd.DataFrame(columns=clms, dtype=float)
                    pred = []
                    true = []
                    model = Ensemble(num_components=num_components, class_list=label_list,
                                     num_features=stream.n_features + 1,
                                     max_pool_size=500, wind_size=1000,
                                     ftr=stream.features, size_to_train=size_to_train,
                                     size_to_test=size_to_test)
                    for counter, (x, y) in tqdm(enumerate(stream)):
                        prediction = model.predict(x, dy=y)
                        pred.append(prediction)
                        true.append(y)
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
                    print(df1.mean())
                    print("---------------------------------------------------")
                    b = {'dataset': dataset, 'num_components': num_components, 'train_size': size_to_train,
                         'test_size': size_to_test, 'avg_geo': df1['avg_geo'].mean(), 'recall': df1['recall'].mean(),
                         'precision': df1['precision'].mean(), 'f1': df1['f1'].mean(), 'kappa': df1['kappa'].mean()}
                    del df1
                    del stream
                    del model
                    del label_list
                    del pred
                    del true
                    parameter_table = pd.concat([parameter_table, pd.DataFrame([b])], ignore_index=True)
                    print("---------------------------------------------------")
    parameter_table.to_csv("/data/Soheil-data/Python Projects/DynED-Imb_v2/Imbalance "
                           "Datasets/arff-multi-class-synthetic/parameter_tuning/" + "Imb_RandomRBFD_5cls_10ft_30driftatt_0noise_speed500_" + "parameter_table.csv",
                           index=False)


if __name__ == "__main__":
    main()
