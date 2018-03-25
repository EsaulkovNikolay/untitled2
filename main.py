from word2vec import Word2vec
import os


if __name__ == '__main__':
    data_path = os.path.abspath("data/")
    train_data_filename = "data1.xlsx"
    train_data_path = os.path.join(data_path, train_data_filename)

    if not os.path.exists(train_data_path):
        print("File not found {}".format(train_data_path))
        exit(0)

    word2vec = Word2vec()
    word2vec.load_train_data_excel(train_data_path, columns="E,I")
    word2vec.prepare_train_data()
    word2vec.build_vocab()
    # word2vec.load_model("models/best86")
    test_data_file_names = ["data/BE_4500.xlsx", "data/BE_5600.xlsx", "data/BE_6400.xlsx"]
    word2vec.load_test_data_excel(test_data_file_names, columns="C,E")
    while True:
        print(len(word2vec.model.wv.vocab))
        test_values = word2vec.test_1()
        word2vec.train_data_matrix = word2vec.test_data
        word2vec.prepare_train_data(retrain=True)
        word2vec.train()
