from tkinter import *
from word2vec import Word2vec
import os


def search():
    global word2vec
    text = entry.get()
    listbox1.delete(0, END)
    if len(text):
        result = word2vec.find(text.upper())
        for line in result:
            listbox1.insert(END, line)
    return 0

data_path = os.path.abspath("data/")
train_data_filename = "data1.xlsx"
train_data_path = os.path.join(data_path, train_data_filename)

if not os.path.exists(train_data_path):
    print("File not found {}".format(train_data_path))
    exit(0)
#
word2vec = Word2vec()
word2vec.load_train_data_excel(train_data_path, columns="E,I")
word2vec.prepare_train_data()
word2vec.load_model("models/best86")


root = Tk()
root.geometry('500x400+300+200')  # ширина=500, высота=400, x=300, y=200

entry = Entry(bd=1, width=35, bg='white', font='arial 16')
entry.place(x=-5, y=-5, relwidth=0.84, relheight=0.075, relx=0.02, rely=0.02)

button = Button(command=search, text="Поиск")
button.place(x=-2, y=-5, relwidth=0.085, relheight=0.075, relx=0.9, rely=0.02)

listbox1 = Listbox(root, height=5, width=100, selectmode=SINGLE, font='arial 16')
result = []
for i in result:
    listbox1.insert(END, i)
listbox1.place(x=-5, y=-4, relwidth=0.95, relheight=0.8, relx=0.02, rely=0.1)

scrollbar_h = Scrollbar(root, orient='hor')
scrollbar_v = Scrollbar(root, orient='vert')
scrollbar_h.place(x=0, y=-5, relwidth=0.95, relx=0.01, rely=0.9, bordermode='inside')
scrollbar_v.place(x=-20, y=-3, relheight=0.84, relx=1, rely=0.1, bordermode='inside')
scrollbar_v['command'] = listbox1.yview
scrollbar_h['command'] = listbox1.xview
listbox1['yscrollcommand'] = scrollbar_v.set
listbox1['xscrollcommand'] = scrollbar_h.set

root.mainloop()
