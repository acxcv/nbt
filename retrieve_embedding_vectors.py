import json
import pandas as pd
import torch
import numpy as np
from nltk.tokenize import word_tokenize
from word_vector import load_obj, save_obj
from pprint import pprint


def retrieve_word_embeddings(word, embeddings_dictionary):
    try:
        word_embedding = embeddings_dictionary[word.lower()]
        word_embedding = [float(x) for x in word_embedding]
    except KeyError:
        word_embedding = [0] * 300
        print("Word not found in Paragram: ", word)

    return word_embedding


def pad_embeddings_list(list, padding_length=40):
    if len(list) < padding_length:
        pad = padding_length - len(list)
        padding = np.array([0] * 300)
        padding_list = [padding] * pad
        final_list = list + padding_list
    else:
        return list

    return final_list


def create_dataset(json_file, output_name):
    pairs_dict = load_obj('pairs_dict')
    embeddings_dict = load_obj('paragram-300-sl999')
    candidate_pairs = load_obj('pairs_list')

    vector_frame = pd.DataFrame(
        columns=[
            # 'dialogue_id',
            # 'utterance_id',
            # 'utterance',
            # 'tokenized_utterance',
            'vectorized_utterance',
            'previous_system_output',
            # 'positive_pairs'
            'candidate_pair',
            'binary_label',
        ])

    df_id = 0
    counter = 0
    total_num = len(json_file)
    for dialogue in json_file:

        # store id in new df
        dialogue_id = dialogue['dialogue_idx']

        # turns object is a list containing dialogue dict objects with
        # belief state, transcript, turns, etc.
        turns = dialogue['dialogue']

        utt_id = 0
        print(f'processing dialogue {counter} of {total_num}')

        for turn in turns:
            current_turn_state = turn['turn_label']
            # for each turn dict declare value for 'transcript' as utterance
            utterance = turn['transcript']
            # tokenize utterances and append them to tokenized utterance list
            tokenized_utterance = word_tokenize(utterance)
            # Retrieve embedding/vector for each token in utterance
            vectorized_utterance = []
            for token in tokenized_utterance[:39]:  # Truncate to 40 tokens
                token_embedding = retrieve_word_embeddings(token,
                                                           embeddings_dict)
                vectorized_utterance.append(token_embedding)

            # Retrieve previous system output
            system_output = turn['system_acts']
            # If no system output, create a zero vector
            if len(system_output) == 0:
                # TODO: Determine how long should zero vector be
                vectorized_system_output = np.array([0] * 300)
                vectorized_system_output = \
                    [torch.tensor(
                        pad_embeddings_list([vectorized_system_output]))]
            # Retrieve embeddings for slot-value pair of system output
            else:
                vectorized_system_output = []
                for slot_or_pair in system_output:
                    # if slot_or_pair is a list, it is a pair
                    if isinstance(slot_or_pair, list):
                        pair_embedding = pairs_dict[f'{slot_or_pair}']
                        # System output may have more than one pair
                        vectorized_system_output.append(pair_embedding)
                    else:
                        # System output may have more than one slot, so we want
                        # a list of slot vectors
                        if ' ' in slot_or_pair:
                            multiword_slot = slot_or_pair.split()
                            multiword_slot_embedding = \
                                [retrieve_word_embeddings(w, embeddings_dict)
                                 for w in multiword_slot]
                            # Sum multiword slots/values into single vector
                            multiword_slot_embedding = \
                                np.array(multiword_slot_embedding).sum(axis=0)
                            # TODO: CHECK IF NEEDS TO BE PADDED AND TENSORIZED
                            multiword_slot_embedding = torch.tensor(
                                pad_embeddings_list([multiword_slot_embedding])
                            )
                            vectorized_system_output.append(
                                multiword_slot_embedding)
                        else:
                            slot_embeddings = retrieve_word_embeddings(
                                slot_or_pair, embeddings_dict)
                            vectorized_system_output.append(
                                torch.tensor(pad_embeddings_list([slot_embeddings])))

            padded_vectorized_utterance = torch.tensor(np.array(
                pad_embeddings_list(vectorized_utterance)))

            # create new relevant dataframe rows with previous sys-output,
            # vectorized utterance, candidate pair, and binary label
            for pair in candidate_pairs:
                binary_label = 1 if pair in current_turn_state else 0
                pair_embeddings = pairs_dict[f'{pair}']
                current_row = [padded_vectorized_utterance,
                               vectorized_system_output,
                               pair_embeddings,
                               binary_label]
                vector_frame.loc[df_id] = current_row
                # print(vector_frame.head())
                # utt_id += 1
                df_id += 1
        counter += 1
        if counter == 25:
            save_obj(vector_frame, f'{output_name}_dummy')
            print(f'pickle object {output_name}_dummy saved')

    print(vector_frame.head(5))
    save_obj(vector_frame, f'{output_name}')
    print(f'pickle object {output_name} saved')
    print('retrieve_embedding_vectors RUN COMPLETE')

    return vector_frame


test_file = open("woz_test_en.json", 'r', encoding='utf-8')
loaded_test = json.load(test_file)

# train_file = open("woz_train_en.json", 'r', encoding='utf-8')
# loaded_train = json.load(train_file)

validate_file = open("woz_validate_en.json", 'r', encoding='utf-8')
loaded_validate = json.load(validate_file)

# create_dataset(loaded_train, "vector_frame_train")
create_dataset(loaded_validate, "vector_frame_validate")
create_dataset(loaded_test, "vector_frame_test")
