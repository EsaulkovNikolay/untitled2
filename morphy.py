import pandas as pd
import gensim
# import logging
import pymorphy2


# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def split_by_separators(string, separators: list):
    # split string by spaces and list of separators
    for sep in separators:
        string = string.replace(sep, " ")
    return string.split()


class Word2vec:
    def __init__(self):
        self.model = None
        self.train_data_matrix = None
        self.train_data = None
        self.test_data = None
        self.values = None
        self.text_separators = ['"', '-', '.', '#']

    def load_train_data_excel(self, filename, columns=None):
        try:
            self.train_data_matrix = pd.read_excel(filename, usecols=columns).as_matrix()
            self.values = set([c[1].upper() for c in self.train_data_matrix])
        except FileNotFoundError:
            print('Loading failed. File "{}" not found'.format(filename))
            exit(1)

    def prepare_train_data(self, retrain=False):
        sentences = []
        morph = pymorphy2.MorphAnalyzer()
        for pair in self.train_data_matrix:
            sap_name = pair[1].upper()
            for line in pair:
                if retrain:
                    line = pair[1]
                if not type(line) is float:
                    words = split_by_separators(line.upper(), self.text_separators)
                    for word in words.copy():
                        if not word[0].isdigit():
                            words.append(morph.parse(word)[0].normal_form.upper())

                    words.insert(len(words) // 2, sap_name)
                    sentences.append(words)
                if retrain:
                    break
        self.train_data = sentences

    def build_vocab(self, iterations=100, workers=4, sg=1, hs=1, min_count=0, window=15,
                    compute_loss=True):
        if not self.train_data:
            print("Train data is not prepared")
            return
        self.model = gensim.models.Word2Vec(self.train_data,
                                            iter=iterations,
                                            workers=workers,
                                            sg=sg,
                                            hs=hs,
                                            min_count=min_count,
                                            window=window,
                                            compute_loss=compute_loss,
                                            alpha=0.005
                                            )
        # del self.train_data

    def train(self):
        if self.train_data:
            # self.model.train(self.train_data, total_examples=len(self.train_data), epochs=self.model.epochs,
            #                 start_alpha=0.003, end_alpha=0.001)
            # del self.train_data
            self.model.build_vocab(self.train_data, update=True)
            self.model.train(self.train_data, total_examples=len(self.train_data), epochs=100,
                             start_alpha=0.005, end_alpha=0.001)
        else:
            print("Train data is not prepared")

    def load_test_data_excel(self, file_names, columns=None):
        self.test_data = []
        for filename in file_names:
            try:
                data = pd.read_excel(filename, usecols=columns).as_matrix()
                [self.test_data.append(c) for c in data]
            except FileNotFoundError:
                print('Loading failed. File "{}" not found'.format(FileNotFoundError.filename))
                exit(1)

    def __filter_input(self, test_input: list) -> list:
        no_input = []
        real_input = []
        for c in test_input:
            if c in self.model.wv:
                real_input.append(c)
            else:
                no_input.append(c)
        for word in no_input:
            while len(word) > 2:
                word = word[:-1]
                if word in self.model.wv:
                    real_input.append(word)
                    break
        return real_input

    def test_1(self, report_file_name=None, topn=300):
        morph = pymorphy2.MorphAnalyzer()
        test_values = list(filter(lambda x: x[1] in self.values, self.test_data))
        passed_count, errors_count, result_len, max_result_len, easy_passed_count = 0, 0, 0, 0, 0
        report = []
        tests_len = len(test_values)

        for i, test in enumerate(test_values):
            test_input = split_by_separators(test[0].upper(), self.text_separators)
            test_answer = test[1].upper()

            if not test_input == test_answer:
                real_input = list(filter(lambda x: x in self.model.wv,
                                         list(map(lambda x: x if x[0].isdigit() else
                                                morph.parse(x)[0].normal_form.upper(), test_input))))

                real_input = self.__filter_input(real_input)

                similarities = self.model.most_similar(positive=real_input, topn=topn)

                result = list(map(lambda x: x[0], filter(lambda x: x[0] in self.values, similarities)))

                # result = [c for c in result if any([str(c).find(x) != -1 for x in real_input])]

                result_len += len(result)
                max_result_len = max(max_result_len, len(result))

                if test_answer in result:
                    passed_count += 1
            else:
                passed_count += 1
                easy_passed_count += 1

        print()
        print("Всего тестов: ", tests_len)
        print("Тестов пройдено: ", passed_count)
        print("Количество ошибок: ", errors_count)
        print(str(round(passed_count / tests_len * 100, 5)) + "% тестов пройдено")
        print("Средняя длина результата: ", result_len / tests_len)
        print("Максимальная длина результата: ", max_result_len)
        print(easy_passed_count)

        if report_file_name:
            frame = pd.DataFrame(report, columns=["Запрос", "SAP", "Правильный ответ", "Варианты"])
            import xlsxwriter
            writer = pd.ExcelWriter(report_file_name, engine='xlsxwriter')
            frame.to_excel(writer)

        return self.test_data  # test_values

    def test(self, report_filename=None):
        # test_values = [c for c in self.test_data if c[1] in self.values]
        test_values = list(filter(lambda x: x[1] in self.values, self.test_data))
        passed, errors, result_len, max_result_len = 0, 0, 0, 0
        tests_len = len(test_values)
        report, train_data = [], []
        easy = 0
        for i in range(0, tests_len):
            try:
                result = []
                test_input = split_by_separators(test_values[i][0].upper(), self.text_separators)
                no_input = []
                real_input = []
                for c in test_input:
                    if c in self.model.wv:
                        real_input.append(c)
                    else:
                        no_input.append(c)
                for word in no_input:
                    while len(word) > 2:
                        word = word[:-1]
                        if word in self.model.wv:
                            real_input.append(word)
                            break
                if not test_values[i][0].upper() == test_values[i][1].upper():
                    similarities = self.model.most_similar(positive=real_input, topn=500)
                    # similarities = self.model.predict_output_word(real_input,topn=500)
                    result = list(map(lambda x: x[0], filter(lambda x: x[0] in self.values, similarities)))
                    # [result.append(c[0]) for c in similarities if c[0] in self.values]
                    result = [c for c in result if any([str(c).find(x) != -1 for x in real_input])]
                    result_len += len(result)
                    max_result_len = max(max_result_len, len(result))

                    passed_bool = False
                    if test_values[i][1].upper() in result or test_values[i][0].upper() in self.values \
                            or " ".join(real_input) in self.values:
                        passed += 1
                        passed_bool = True
                    else:
                        train_data.append(test_values[i])
                else:
                    easy += 1
                    result_len += 1
                    passed += 1
                    passed_bool = True
                report.append([test_values[i][0].upper(), test_values[i][1].upper(),
                               "+" if passed_bool else "-", result])
            except ValueError:
                errors += 1
                continue

        print()
        print("Всего тестов: ", tests_len)
        print("Тестов пройдено: ", passed)
        print("Количество ошибок: ", errors)
        print(str(round(passed / tests_len * 100, 5)) + "% тестов пройдено")
        print("Средняя длина результата: ", result_len / tests_len)
        print("Максимальная длина результата: ", max_result_len)
        print(easy)

        if report_filename:
            frame = pd.DataFrame(report, columns=["Запрос", "SAP", "Правильный ответ", "Варианты"])
            import xlsxwriter
            writer = pd.ExcelWriter(report_filename, engine='xlsxwriter')
            frame.to_excel(writer)

        return self.test_data  # test_values

    def save_model(self, filename):
        self.model.save(filename)

    def load_model(self, filename):
        self.model = gensim.models.Word2Vec.load(filename)

    def find(self, search_input):
        test_input = split_by_separators(search_input, self.text_separators)
        no_input = []
        real_input = []
        for c in test_input:
            if c in self.model.wv:
                real_input.append(c)
            else:
                no_input.append(c)
        for word in no_input:
            while len(word) > 2:
                word = word[:-1]
                if word in self.model.wv:
                    real_input.append(word)
                    break

        similarities = self.model.most_similar(real_input, topn=700)
        result = []
        if len(similarities):
            result = list(map(lambda x: x[0], filter(lambda x: x[0] in self.values, similarities)))

        return result
