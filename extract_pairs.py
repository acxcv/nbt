import json
from word_vector import save_obj

file = open('ontology/ontology_dstc2_en.json', 'r', encoding='utf8')
data = json.load(file)

list_tuples = [(i, data['informable'][i]) for i in data['informable']]

final_pairs = []

for value in list_tuples[0][1]:
    final_pairs.append(['slot', value])
    final_pairs.append(['request', value])


for tuple in list_tuples[1:]:
    final_pairs.append([tuple[0], 'dontcare'])
    for value in tuple[1]:
        final_pairs.append([tuple[0], value])
        


save_obj(final_pairs, 'pairs_list')

