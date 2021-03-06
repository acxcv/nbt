def create_one_hot(in_list, candidate_pairs):
    """
    :param in_list: A list of valid candidate pairs e.g. [['food', 'persian']]
    :return: One-hot encoded vector for candidates in in_list if valid pairs
    were provided, otherwise raise ValueError
    """
    """
    candidate_pairs = [['slot', 'address'], ['slot', 'area'], ['slot', 'food'],
                       ['slot', 'phone'], ['slot', 'price range'],
                       ['slot', 'postcode'], ['slot', 'name'],
                       ['food', 'afghan'],
                       ['food', 'african'], ['food', 'afternoon tea'],
                       ['food', 'asian oriental'], ['food', 'australasian'],
                       ['food', 'australian'], ['food', 'austrian'],
                       ['food', 'barbeque'], ['food', 'basque'],
                       ['food', 'belgian'],
                       ['food', 'bistro'], ['food', 'brazilian'],
                       ['food', 'british'],
                       ['food', 'canapes'], ['food', 'cantonese'],
                       ['food', 'caribbean'], ['food', 'catalan'],
                       ['food', 'chinese'],
                       ['food', 'christmas'], ['food', 'corsica'],
                       ['food', 'creative'], ['food', 'crossover'],
                       ['food', 'cuban'],
                       ['food', 'danish'], ['food', 'eastern european'],
                       ['food', 'english'], ['food', 'eritrean'],
                       ['food', 'european'],
                       ['food', 'french'], ['food', 'fusion'],
                       ['food', 'gastropub'],
                       ['food', 'german'], ['food', 'greek'],
                       ['food', 'halal'],
                       ['food', 'hungarian'], ['food', 'indian'],
                       ['food', 'indonesian'], ['food', 'international'],
                       ['food', 'irish'], ['food', 'italian'],
                       ['food', 'jamaican'],
                       ['food', 'japanese'], ['food', 'korean'],
                       ['food', 'kosher'],
                       ['food', 'latin american'], ['food', 'lebanese'],
                       ['food', 'light bites'], ['food', 'malaysian'],
                       ['food', 'mediterranean'], ['food', 'mexican'],
                       ['food', 'middle eastern'], ['food', 'modern american'],
                       ['food', 'modern eclectic'],
                       ['food', 'modern european'],
                       ['food', 'modern global'],
                       ['food', 'molecular gastronomy'],
                       ['food', 'moroccan'], ['food', 'new zealand'],
                       ['food', 'north african'], ['food', 'north american'],
                       ['food', 'north indian'], ['food', 'northern european'],
                       ['food', 'panasian'], ['food', 'persian'],
                       ['food', 'polish'],
                       ['food', 'polynesian'], ['food', 'portuguese'],
                       ['food', 'romanian'], ['food', 'russian'],
                       ['food', 'scandinavian'], ['food', 'scottish'],
                       ['food', 'seafood'], ['food', 'singaporean'],
                       ['food', 'south african'], ['food', 'south indian'],
                       ['food', 'spanish'], ['food', 'sri lankan'],
                       ['food', 'steakhouse'], ['food', 'swedish'],
                       ['food', 'swiss'],
                       ['food', 'thai'], ['food', 'the americas'],
                       ['food', 'traditional'], ['food', 'turkish'],
                       ['food', 'tuscan'], ['food', 'unusual'],
                       ['food', 'vegetarian'],
                       ['food', 'venetian'], ['food', 'vietnamese'],
                       ['food', 'welsh'],
                       ['food', 'world'], ['price range', 'cheap'],
                       ['price range', 'moderate'],
                       ['price range', 'expensive'],
                       ['area', 'centre'], ['area', 'north'], ['area', 'west'],
                       ['area', 'south'], ['area', 'east']]
    """
    one_hot = [0] * len(candidate_pairs)
    one_hot_indices = []

    for pair in in_list:
        try:
            positive_index = candidate_pairs.index(pair)
            one_hot_indices.append(positive_index)
        except ValueError as e:
            print(e)

    for idx in one_hot_indices:
        one_hot[idx] = 1

    return one_hot
