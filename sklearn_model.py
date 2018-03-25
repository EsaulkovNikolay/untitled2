from sklearn.preprocessing import LabelEncoder
import pandas as pd


def count_column_value(x):
    if x == '-':
        return -1
    elif isinstance(x, str):
        if '/' in x:
            x = x.split('/')
            return int(x[0])/int(x[1])
    else:
        return x


if __name__ == '__main__':
    data_frame = pd.read_excel('data/Podshipniki_sistema_kharakteristik27_02_17.xlsx',
                               sheet_name='справочник подшипников')

    index = -1
    answer_column = 'Краткий текст материала'
    print(data_frame.iloc[[index]][answer_column])
    #
    values = {'Габаритные размеры, мм (d*D*B)': '-',
              'Внутренний диаметр подшипника, мм': -1,
              'Место применения': '-',
              'ПОЛНОЕ ОБОЗНАЧЕНИЕ АНАЛОГА': '-',
              "ТИП СИСТЕМЫ ОБОЗНАЧЕНИЯ АНАЛОГА": '-',
              "ОСНОВНАЯ ЧАСТЬ ОБОЗНАЧЕНИЯ АНАЛОГА": '-',
              "ТИП СИСТЕМЫ ОБОЗНАЧЕНИЯ": '-',
              "ОСНОВНАЯ ЧАСТЬ ОБОЗНАЧЕНИЯ": '-',
              "ПОЛНОЕ ОБОЗНАЧЕНИЕ": '-',
              "Ширина подшипника, мм": '-',
              "ОСНОВНАЯ ЧАСТЬ ОБОЗНАЧЕНИЯ ПО ГОСТ": '-',
              "Производитель": '-',
              "Наружный диаметр подшипника, мм": '-',
              "Количество рядов тел качения": '-',
              "Тип подшипника": '-',
              "Тип воспринимаемой нагрузки": '-',
              "Материал": -1,
              "Краткий текст материала": '-',
              "Базисная ЕИ": '-'
              }
    #
    data_frame = data_frame.fillna(value=values)

    encoders = {c: LabelEncoder() for c in values.keys()}

    for column in data_frame.columns:
        if values[column] == '-':
            data_frame[column] = encoders[column].fit_transform(list(map(lambda x: str(x), data_frame[column])))
        else:
            data_frame[column] = list(map(count_column_value, data_frame[column]))

    print()
    print(data_frame.iloc[[index]][answer_column])

    df = data_frame.pop(answer_column)

    from sklearn import svm
    clf = svm.SVC(gamma=0.001, C=100.)

    print(clf.fit(data_frame, df))

    result = clf.predict(data_frame.iloc[[index]])
    print(result)
    if result.size > 0:
        print(encoders[answer_column].inverse_transform(result))

    passed = 0
    not_passed = 0
    for i in range(0, data_frame.shape[0]):
        result = clf.predict(data_frame.iloc[[i]])
        if result.size > 0:
            if result[0] == df[i]:
                passed += 1
            else:
                not_passed += 1
        else:
            not_passed += 1
    print("#", passed, not_passed)
