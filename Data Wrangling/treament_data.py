# using for data wrangling and group to generate labels in phase pic
# direct run it and give computer an address is all you need to do ！

import pandas as pd

address = input("please input csv flie address: \n")
file = pd.read_csv(address)
data = pd.DataFrame(file)
temp_data = data.iloc[:,2:len(data.keys())]

def find_label_num(n):
    p = 0; temp_lst = []
    for i in temp_data.iloc[n,:]:
        p += 1
        if i != 0:
            temp_lst.append(p-1)
    return temp_lst
def find_label(temp_lst):
    label_lst = []
    for i in temp_lst:
        label_lst.append(temp_data.keys()[i])
    return label_lst

total_label_lst = []
for i in range(len(temp_data)):
    total_label_lst.append(find_label(find_label_num(i)))
type_label_lst = list(pd.Series(total_label_lst).drop_duplicates())
labels_dict = dict(zip([i + 1 for i in range(len(type_label_lst))], type_label_lst))

data['labels'] = total_label_lst

key_list = []
value_list = []
temp_labels = []
for key, value in labels_dict.items():
    key_list.append(key)
    value_list.append(value)
for i in range(len(data)):
    if data['labels'][i] in value_list:
        temp_labels.append(key_list[(value_list.index(data['labels'][i]))])

data['groups'] = temp_labels
data.to_csv(address, index=False)
