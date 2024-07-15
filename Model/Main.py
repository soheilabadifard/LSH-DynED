from Ensemble import Ensemble
from data_related_functions import *
from classifier_monitor import Metrics, check_true, SlidingWindowForMetrics
from tqdm import tqdm
import pandas as pd
import os

def main(path_to_dataset):

    components = 10
    train = 20
    test = 50
    vector = 5

    dataset_names = os.listdir(path_to_dataset)
    print(f"Number of datasets: {len(dataset_names)}")
    print('datasets: ', dataset_names)

    for dataset in dataset_names:
        print(f"Dataset: {dataset}")
        stream = read_data(path_to_dataset + dataset)
        label_list = stream.classes
        print(f'Number of classes {len(label_list)}')
        print(f'Number of samples {stream.n_samples}')
        print(f'Number of features {stream.n_features}')

        # dataframe to store the statistics of the model
        clms = ['counter', 'g_mean']
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
                         size_to_test=test, vector=vector)
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
            df1.to_csv(f"{path_to_dataset}{dataset}_mgmean.csv", index=False)
            prequential_metrics.get_results_df().to_csv(f"{path_to_dataset}{dataset}_kappa.csv", index=False)


if __name__ == "__main__":
    main('Path to dataset Directory')
