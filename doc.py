import textract
import os
from functools import reduce

path = r"C:\Users\mvideo\Desktop\docs"
full_path = os.path.join(path, "okdp.docx")
# print(full_path)

doc = textract.process(full_path).decode('utf-8')

lines = list(map(lambda x: x.replace("#", " "), doc.replace(" ", "#").split()[329:]))
result = []
i = 0
string = []
for line in lines:
    if line[0].isdigit():
        result.append(string)
        string = [line]
    else:
        string.append(line)
import pandas as pd

# frame = pd.DataFrame(list(map(lambda x: "".join(x).split(maxsplit=1),result[1:])),columns=["Код","Название"])
# with pd.ExcelWriter("A.xlsx",) as writer:
#    frame.to_excel(writer)
data = list(map(lambda x: "".join(x).split(maxsplit=1), result[1:]))
del result
classes = pd.read_excel("1_1.xlsx", usecols="B,C,D,E").as_matrix()

classes = {c[2]: c for c in classes}

# i=0
result = []
# print(classes)
# while i<len(data):
#    if data[i][1].isupper():

table = dict()
max_i = 0
data.sort(key=lambda x: x[0][:4] + x[0][5:] + x[0][4])


class Tree:
    def __init__(self):
        self.dict = dict()
        self.name = ""


root = Tree()
data = list(filter(lambda x: len(x) > 1, data))
for elem in data:
    # if not elem[0][:2] in root.dict:
    #    root.dict.update({elem[0][:2]: Tree()})
    #    root.dict[elem[0][:2]].name = elem[1]
    # else:
    prod_class = elem[0][:4]
    kind = elem[0][4]
    sub_class = elem[0][5:]

    tree = root
    try:
        if int(kind) == 0:
            if int(sub_class) == 0:
                tree.dict[prod_class] = Tree()
                tree.dict[prod_class].name = elem[1]
            else:
                if prod_class not in tree.dict:
                    tree.dict[prod_class] = Tree()
                    tree.dict[prod_class].name = "#"
                tree = tree.dict[prod_class]
                tree.dict[sub_class] = Tree()
                tree.dict[sub_class].name = elem[1]
        else:
            if prod_class not in tree.dict:
                tree.dict[prod_class] = Tree()
                tree.dict[prod_class].name = "#"
            tree = tree.dict[prod_class]
            if sub_class not in tree.dict:
                tree.dict[sub_class] = Tree()
                tree.dict[sub_class].name = "#"
            tree = tree.dict[sub_class]
            tree.dict[kind] = Tree()
            tree.dict[kind].name = elem[1]
    except KeyError:
        print(elem)
print()

excel_output = []
for class_code in root.dict.keys():
    class_name = root.dict[class_code].name
    tree = root.dict[class_code]
    if len(tree.dict) == 0:
        excel_output.append([class_code, class_name, "", "", "", "", class_code + "000"])
    for sub_class_code in tree.dict.keys():
        sub_class_name = tree.dict[sub_class_code].name
        sub_tree = tree.dict[sub_class_code]
        if len(sub_tree.dict) == 0:
            excel_output.append([class_code, class_name, sub_class_code, sub_class_name, "", "",
                                 class_code + "0" + sub_class_code])
        for kind_code in sub_tree.dict.keys():
            kind_name = sub_tree.dict[kind_code].name
            excel_output.append([class_code, class_name, sub_class_code, sub_class_name, kind_code,
                                 kind_name, class_code + kind_code + sub_class_code])
# print(excel_output)
columns = ["Запрос", "SAP", "Правильный ответ", "Варианты"]
frame = pd.DataFrame(excel_output, columns=["Код класса", "Название класса", "Код подкласса",
                                            "Название подкласса", "Код вида", "Название вида", "Полный код"])
with pd.ExcelWriter("classification.xlsx", engine="xlsxwriter") as writer:
    import xlsxwriter

    frame.to_excel(writer, index=False, header=True)
