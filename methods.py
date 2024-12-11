#! usr/bin/python3
import dendropy
from pandas import Series, DataFrame
import numpy as np
from copy import deepcopy

def read_tree(path):
    tree = dendropy.Tree.get(path=path, schema='newick') # or whatever relevant format if not newick
    distances = get_distance_dataframe(tree)
    distances = make_distance_matrix(distances, True)
    return distances


def get_tree2_distances(tree1_distances, changes):
    distances = deepcopy(tree1_distances)
    distances.index = distances.index.map(lambda x : changes.get(x, x))
    distances.columns = distances.columns.map(lambda x : changes.get(x, x))
    return distances


def get_distance_dataframe(tree):
    pdm = tree.phylogenetic_distance_matrix()
    distances = [[*name, distance] for name, distance in zip(pdm.distinct_taxon_pair_iter(), pdm.distances())]
    distances = DataFrame(distances)
    
    def get_label(node):
        return node.label
    
    distances[0] = distances[0].map(get_label)
    distances[1] = distances[1].map(get_label)
    
    distances = distances.set_index([0, 1])
    distances = distances.rename(columns = {2:'distance'})

    return distances


def make_distance_matrix(distances, sort=False):
    distances = distances.reset_index().pivot_table(index=0, columns=1, values='distance')
    distances.index.name=None
    distances.columns.name=None
    distances = distances.fillna(distances.T)

    missing_rows = [i for i in distances.index if i not in distances.columns]
    for row in missing_rows:
        distances[row] = distances.loc[row]

    missing_columns = [c for c in distances.columns if c not in distances.index]
    for column in missing_columns:
        distances.loc[column] = distances[column]
    
    if sort:
        distances = distances.sort_index().sort_index(axis=1)
    return distances.fillna(0)


def noisify_distances(distances, scale):
    noise = np.triu(1 + np.random.random(distances.shape) * scale).round(2)
    noise += noise.T
    return distances * noise


def baseline_method(distances1, distances2, epsilon):
    
    common_leaves = [c for c in distances2.columns if c in distances1.columns]
    
    # step 1 compute ratio    
    R = get_ratio(distances1, distances2, common_leaves)
    print('ratio:', R)
    
    # step 2 compute the sum of distances from each candidate to all common leaves
    candidates2 = [c for c in distances2.columns if c not in common_leaves]
    candidate_distances2 = distances2.loc[candidates2, common_leaves].sum(axis=1)
    
    candidates1 = [c for c in distances1.columns if c not in common_leaves]
    candidate_distances1 = distances1.loc[candidates1, common_leaves].sum(axis=1)
    
    # step 3 compare distances adjusted with the ratio
    candidate_distances2 = candidate_distances2 * R
    
    # step 4 find the closest matches
    matches = {}
    for candidate in candidates2:
        d_candidate = candidate_distances2[candidate]
        sorted_matches = (candidate_distances1 - d_candidate).abs().sort_values()        
        acceptable_matches = sorted_matches <= sorted_matches.min() + epsilon
        matches[candidate] = sorted_matches[acceptable_matches].to_dict()

    return DataFrame(matches).stack().dropna()


def get_ratio(distances1, distances2, common_leaves):
    r1 = distances1.loc[common_leaves, common_leaves].sum().sum() / 2
    r2 = distances2.loc[common_leaves, common_leaves].sum().sum() / 2
    return r1/r2


def get_matches(matches, level=0, changes={}):
    # Best matches according to T1 and T2 respectively
    tree_matches = matches.sort_values().groupby(level=level).head(1).reset_index()
    tree_matches.columns = ['t1_name', 't2_name', 'distance']
    tree_matches['t1_match_in_t2'] = tree_matches.t1_name.map(changes)
    return tree_matches


def get_scores(t1_matches, t2_matches):
    t1_score = (t1_matches.t2_name == t1_matches.t1_match_in_t2).mean()
    t2_score = (t2_matches.t2_name == t2_matches.t1_match_in_t2).mean()
    scores = {
        't1_score': t1_score,
        't2_score': t2_score,
        'mean_score': (t1_score+t2_score)/2
    }
    return scores
