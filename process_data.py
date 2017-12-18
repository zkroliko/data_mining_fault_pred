from sklearn import decomposition

from data_utils import load_raw_data, save_processed_data

PROCESSED_PS = "_ps"

PCA_TARGET_SIZE = 10


def process_and_save(training_filename="training_data.csv", test_filename="test_data.csv"):
    (x_train, y_train), (x_test, y_test) = load_raw_data(training_filename), load_raw_data(test_filename)
    print("### Loaded {} training rows".format(x_train.shape[0]))
    print("## X_train shape: ", x_train.shape)
    print("## Y_train shape: ", y_train.shape)
    print("### Loaded {} test rows".format(x_test.shape[0]))
    print("## X_test shape: ", x_test.shape)
    print("## Y_test shape: ", y_test.shape)

    # PCA dimensionality reduction

    pca = decomposition.PCA(n_components=PCA_TARGET_SIZE)
    pca.fit(x_train)
    x_train = pca.transform(x_train)
    pca.fit(x_test)
    x_test = pca.transform(x_test)

    print("Reduced data to {} dimensions", PCA_TARGET_SIZE)

    print("Saving the process data")

    save_processed_data(training_filename.replace(".csv", PROCESSED_PS), x_train, y_train)
    save_processed_data(test_filename.replace(".csv", PROCESSED_PS), x_test, y_test)


process_and_save()
