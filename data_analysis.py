import pandas as pd
import os
import pprint

pp = pprint.PrettyPrinter()

dirname = os.path.dirname(__file__)
woz_data = pd.read_json('multiwoz/data/MultiWOZ_2.2/test/dialogues_001.json')

print(woz_data.columns)

print("\n TURNS \n")

pp.pprint(woz_data['turns'][0])

print("\n SERVICES \n")

pp.pprint(woz_data['services'][0])