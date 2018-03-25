import gensim
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE

import logging

#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class Word2VecInput:
    def __init__(self):
        self.data = []
        self.data_matrix = []
        pass

    def read_excel(self, filename):
        self.data_matrix = pd.read_excel(filename, usecols='C,G').as_matrix()
        return self.data_matrix

    def read_excel_1(self, filename):
        self.data_matrix = pd.read_excel(filename, usecols='E,I').as_matrix()
        return self.data_matrix

    def prepare_data_0(self):
        sentences = []
        pairs = self.data_matrix
        for pair in pairs:
            if not type(pair[0]) is float:
                words = pair[1].upper().split()
                for word in words.copy():
                    if len(word) > 6:
                        words.append(word[0:2])
                        words.append(word[0:3])
                        words.append(word[0:len(word) - 4])
                        words.append(word[0:len(word) - 2])
                    elif len(word)>4:
                        words.append(word[0:2])
                        words.append(word[0:3])
                        words.append(word[0:len(word) - 2])
                    elif len(word)>3:
                        words.append(word[0:2])
                words.insert(len(words) // 2, pair[1])
                sentences.append(words)
        self.data = sentences

        pass

    def prepare_data_1(self):
        sentences = []
        pairs = self.data_matrix
        for pair in pairs:
            sap_name = pair[1].upper()
            for line in pair:
                if not type(line) is float:
                    words = line.upper().replace('"',' ').replace('-','').split()
                    for word in words.copy():
                        if not word[0].isdigit():
                            if len(word) > 6:
                                words.append(word[0:2])
                                words.append(word[0:3])
                                words.append(word[0:len(word) - 4])
                                words.append(word[0:len(word) - 2])
                            elif len(word) > 4:
                                words.append(word[0:2])
                                words.append(word[0:3])
                                words.append(word[0:len(word) - 2])
                            elif len(word) > 3:
                                words.append(word[0:2])
                        #else:
                            #print(word)
                    words.insert(len(words) // 2, sap_name)
                    sentences.append(words)
        self.data = sentences
        #write_to_excel(sentences,"prepared_data.xlsx")
        pass

def show_word2vec(model):
    vocab = list(model.wv.vocab)
    X = model[vocab]

    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(X)
    df = pd.DataFrame(X_tsne, index=vocab, columns=['x', 'y'])
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(df['x'], df['y'])
    for word, pos in df.iterrows():
        ax.annotate(word, pos)

    plt.show()
    pass


def get_word2vec_model(filename):
    data = Word2VecInput()
    data.read_excel_1(filename)
    data.prepare_data_1()
    # 10 => 59% ; 20 => 64.5% ; 25=>63,1 ; 20+compute_loss => 66%
    model = gensim.models.Word2Vec(data.data, iter=100, workers=4, sg=1, hs=1, min_count=1,
                                   window=20, compute_loss=True, alpha=0.005)
    return model, data


def test_model(model, data, data1, make_excel_output=False):
    values = set([c[1].upper() for c in data.data_matrix])
    test_values = [c for c in data1.data_matrix if c[1] in values]
    passed = 0
    tests_len = len(test_values)
    report = []
    train_data=[]
    errors = 0
    max_res_len=0
    for i in range(0, tests_len):
        try:
            result = []
            input = test_values[i][0].upper().split()
            noinput = []
            real_input = []
            for c in input:
                if c in model.wv:
                    real_input.append(c)
                else:
                    noinput.append(c)
            for word in noinput:
                while len(word)>2:
                    word = word[:-1]
                    if word in model.wv:
                        real_input.append(word)
                        break

            similarities = model.most_similar(positive=real_input, topn=30)
            [result.append(c[0]) for c in similarities if c[0] in values]

            max_res_len += len(result)

            passed_bool = False
            if test_values[i][1] in result or test_values[i][0].upper() in values:
            #if len(result)>0:
                passed += 1
                passed_bool = True
            else:
                report.append([test_values[i][0].upper(), test_values[i][1].upper(), "+" if passed_bool else "-"])
                train_data.append([test_values[i]])
        except ValueError:
            errors+=1
            continue

    print()
    print("total tests = ", tests_len)
    print("tests passed = ", passed)
    print("tests errors = ", errors)
    print(str(round(passed / tests_len * 100, 5)) + "% tests passed")
    print("max_res_len = ", max_res_len / tests_len)

    if make_excel_output:
        write_to_excel(report, 'report2.xlsx', columns=["Запрос", "SAP", "Правильный ответ"])

    return test_values


def write_to_excel(data, filename, columns=None):
    frame = pd.DataFrame(data, columns=columns)
    import xlsxwriter
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    frame.to_excel(writer)


if __name__ == '__main__':
    #model, data = get_word2vec_model('data/data1.xlsx')
    #model.save("word2vec_model_last")
    model = gensim.models.Word2Vec.load("77percent_model")
    data = Word2VecInput()
    data.read_excel_1('data/data1.xlsx')
    data1 = Word2VecInput()
    data1.read_excel('data/data.xlsx')
    test_values=test_model(model, data, data1)

    train_input=[]
    for pair in test_values:
        sap_name = pair[1].upper()

        if not type(sap_name) is float:
            words = sap_name.upper().replace('"', ' ').replace('-', '').replace('.', ' ').split()
            for word in words.copy():
                if not word[0].isdigit():
                    if len(word) > 6:
                        words.append(word[0:2])
                        words.append(word[0:3])
                        words.append(word[0:len(word) - 4])
                        words.append(word[0:len(word) - 2])
                    elif len(word) > 4:
                        words.append(word[0:2])
                        words.append(word[0:3])
                        words.append(word[0:len(word) - 2])
                    elif len(word) > 3:
                        words.append(word[0:2])
                # else:
                #    print(word)
            words.insert(len(words) // 2, sap_name)
            train_input.append(words)

    model.train(train_input,total_examples=len(train_input),epochs=model.epochs)
    test_values=test_model(model,data,data1)
