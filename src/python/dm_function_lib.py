import pandas as pd
import csv
from scipy.sparse import coo_matrix, save_npz, load_npz
import os
from implicit.evaluation import train_test_split
from sklearn.model_selection import ParameterGrid
from implicit_extend.evaluation import ranking_metrics_at_k
import numpy as np
from itertools import chain
import random
import warnings
from datetime import datetime

warnings.simplefilter('ignore')


############# DATABASE READING FUNCTIONS #############

def read_table(engine, table_name):
    """
    Reads an SQL table and converts integer columns to int, and string columns to str utf-8.
    :param engine: an olready started MySQL engine to connect to the database
    :param table_name: name of the SQL table to read
    :return: a dataframe equivalent to the original SQL table.
    """
    df = pd.read_sql_table(table_name, engine, parse_dates=True)
    if 'id' in df.columns:
        df['id'] = df['id'].astype('int64')
    str_cols = df.select_dtypes([object]).columns
    for col in str_cols:
        if (df[col].apply(type) == bytes).any():
            df[col] = df[col].str.decode('utf-8')

    return df


def get_not_tagged(engine, path="../../data/not_tagged.csv", proposals_df=None):
    """
    Get a list of the ids of the proposals that have not been tagged with category, topic and location
    tags. This list is read from a previously generated csv, that this function creates when it doesn't
    exist.
    :param engine: engine to connect to the required datasets to create the list of not tagged
    :param path: path to the previously created csv file
    :param proposals_df: only required if
    :return:
    """
    try:
        f = open(path, "r")
        with f:
            not_tagged = list(csv.reader(f, delimiter=";"))
            not_tagged = [int(string) for inner_list in not_tagged for string in inner_list]

    except FileNotFoundError:
        if proposals_df is None:
            proposalIds = sorted(read_table(engine, "proposals").id.unique())
        else:
            proposalIds = sorted(proposals_df.id.unique())
        propcatsIds = read_table(engine, 'proposal_categories').id.unique()
        proplocIds = read_table(engine, 'proposal_locations').id.unique()

        not_tagged = [int(i) for i in proposalIds if i not in propcatsIds or i not in proplocIds]

        pd.DataFrame(data={'id': not_tagged}).to_csv(path, sep=';',
                                                     encoding='utf-8', index_label=False,
                                                     index=False,
                                                     header=False)
    return not_tagged


############# CREATION OF THE USER_ITEM RATINGS MATRIX #############
def get_ratings_matrix(engine=None, divide_campaigns=False):
    """ Creates ratings matrix (or matrices if divide_campaigns=True) and returns index conversion
    of userIds and proposalIds for each matrix:
        - Rows are users and columns are proposals
        - Ratings are number of comments made by user in proposal, 0 if none.
        - The matrix is a csr sparse matrix
        - The matrix contains only users that commented in a campaign.
        - If divide_campaigns=False, only one matrix is returned containing ALL proposals ever recorded on the system,
        even if they were never commented.
        - If divide_campaigns=True, it returns as many matrices as campaigns, each matrix containing ALL users,
        and the proposals BELONGING to that campaign (it can happen that some users who commented on other campaigns,
        never commented on the corresponding campaign and are null on the matrix).
        - IMPORTANT: Proposals missing district or category tags are removed and excluded.
        - Those proposals missing one of both tags, are saved into the not_tagged.csv file
    Parameters
    ----------
    engine: mysql engine
        To access DB
    divide_campaigns: (boolean, default=False)
        Whether to divide proposals in campaigns or not, and to return as
        many rating matrices as the resulting divisions.

    Returns
    -------
    - {'campaign': {'rm': rm_campaign,
                    'userId_cnv': userId_cnv_campaign,
                    'proposalId_cnv': proposalId_cnv}}
    Dictionary with a key for each campaign, (key='all' if divide_campaigns=False), and each value the dictionary
    containing the corresponding user-proposal matrix and the user and proposal id conversions.
    """
    if not divide_campaigns:
        propcom_df = read_table(engine, 'proposal_comments')
        proposals_df = read_table(engine, "proposals")

        # Exclude not_tagged
        not_tagged = get_not_tagged(engine, proposals_df=proposals_df)
        propcom_df = propcom_df[~(propcom_df.proposalId.isin(not_tagged))]

        return {'all': get_ratings_matrix_aux(propcom_df,
                                              proposals_df.id.sort_values().unique(), # Any proposal in the system
                                              propcom_df.userId.sort_values().unique())} # Only users who have commented
    else:
        return get_ratings_matrix_campaigns(engine)


def get_ratings_matrix_campaigns(engine):
    """ Function to divide proposals and comments into campaigns

    Returns:
    - List of lists, each list corresponding to the proposalIds of a campaign

    """
    proposals_df = read_table(engine, "proposals")
    propcom_df = read_table(engine, "proposal_comments")

    # Exclude not_tagged
    not_tagged = get_not_tagged(engine, proposals_df=proposals_df)
    proposals_df = proposals_df[~proposals_df.id.isin(not_tagged)]

    # Get proposalsIds divided by campaigns
    campaign_ids = divide_campaigns(engine, proposals_df, not_tagged=False)

    min_date = proposals_df.date.min()
    max_date = proposals_df.date.max()

    results = {}
    i = 0

    while min_date < max_date:

        end_date = min_date + pd.DateOffset(years=1)
        if end_date == max_date:
            end_date = end_date + pd.Timedelta(days=1)

        propcom_camp_df = propcom_df[(propcom_df.proposalId.isin(campaign_ids[i])) &
                                     (min_date <= propcom_df.date) &
                                     (propcom_df.date < end_date)] \
            .reset_index(drop=True)

        userIds_camp = sorted(propcom_camp_df.userId.unique())# Only users who have commented in the campaign

        campaign_ids[i]
        # Using propcom_camp_df forces the matrix to only have proposals that have been commented at least once
        results[i + 1] = get_ratings_matrix_aux(propcom_camp_df, campaign_ids[i], userIds_camp)
        min_date = end_date
        i += 1

    return results


def divide_campaigns(engine, proposals_df=None, not_tagged=False):
    campaign_ids = []

    # Tries to get it from files
    try:
        for i in range(1, 5):
            f = open(f"../../data/campaigns/campaign_{str(i)}_ids.csv", 'r')
            with f:
                proposalIds = list(csv.reader(f, delimiter=";"))
                proposalIds = [int(string) for inner_list in proposalIds for string in inner_list]
                campaign_ids.append(sorted(proposalIds))

    # If not, it generates and saves the files
    except FileNotFoundError:
        if proposals_df is None:
            proposals_df = read_table(engine, "proposals")

        # Exclude not_tagged if it has not been done before
        if not_tagged is not False:
            if not_tagged is None:
                not_tagged = get_not_tagged(engine, proposals_df=proposals_df)
            proposals_df = proposals_df[~proposals_df.id.isin(not_tagged)].sort_values(by='id')

        min_date = proposals_df.date.min()
        max_date = proposals_df.date.max()
        i = 1

        while min_date < max_date:
            end_date = min_date + pd.DateOffset(years=1)
            if end_date == max_date:
                end_date = end_date + pd.Timedelta(days=1)

            proposalIds = proposals_df[(min_date <= proposals_df.date) &
                                       (proposals_df.date < end_date)][['id']].sort_values(by='id')
            outdir = '../../data/campaigns'
            if not os.path.exists(outdir):
                os.mkdir(outdir)

            proposalIds.to_csv(f"{outdir}/campaign_{str(i)}_ids.csv",
                               sep=';', encoding='utf-8', index_label=False,
                               index=False, header=False)
            campaign_ids.append(proposalIds.id.values)

            min_date = end_date
            i += 1

    return campaign_ids


def get_ratings_matrix_aux(propcom_df, proposalIds, userIds):
    # We create ratings with number of comments
    ratings_df = propcom_df[['proposalId', 'userId']].copy()
    ratings_df['num_comments'] = 1
    ratings_df = ratings_df.groupby(['proposalId', 'userId']).sum().reset_index()
    # We correct user and proposal Ids so that they are consecutive, and we keep the conversion on dataframe
    userId_cnv = pd.DataFrame(data={'userId': userIds,
                                    'finalUserId': range(0, len(userIds))})
    proposalId_cnv = pd.DataFrame(data={'proposalId': proposalIds,
                                        'finalProposalId': range(0, len(proposalIds))})
    ratings_df = ratings_df.merge(userId_cnv, how='left', on='userId').drop(columns='userId')
    ratings_df = ratings_df.merge(proposalId_cnv, how='left', on='proposalId').drop(columns='proposalId')
    ratings_df.sort_values(by=['finalUserId', 'finalProposalId']).inplace=True
    # coo_matrix((data, (row, col)), shape=(nrows, ncols))
    print(len(ratings_df.num_comments))
    print(ratings_df.num_comments.shape[0])

    rm = coo_matrix((ratings_df.num_comments, (ratings_df.finalUserId, ratings_df.finalProposalId)),
                    shape=(len(userIds), len(proposalIds))).tocsr()

    return {'rm': rm, 'userId_cnv': userId_cnv, 'proposalId_cnv': proposalId_cnv}


def get_prop_tag_matrix(campaign=1, prop_tags_df=None, engine=None,
                        prop_tags_df_name='proposal_categories',
                        tag='category', weight='weight'):
    try:
        f = open(f"../../data/campaigns/campaign_{str(campaign)}_ids.csv", 'r')
        with f:
            proposalIds = list(csv.reader(f, delimiter=";"))
            proposalIds = [int(string) for inner_list in proposalIds for string in inner_list]
            proposalIds = sorted(proposalIds)
    except FileNotFoundError:
        print('Campaigns can only be a number in the set {1, 2, 3, 4}.')

    if prop_tags_df is None:
        prop_tags_df = read_table(engine, prop_tags_df_name)

    prop_tags_df = prop_tags_df[prop_tags_df.id.isin(proposalIds)]

    # Location weights are calculated as the number of times a district is assigned to a proposal,
    # divided by the number of locations assigned to that proposal.
    if weight not in prop_tags_df.columns:
        prop_tags_df[weight] = prop_tags_df.groupby(by=['id', tag])['neighborhood'].transform('count') / \
                               prop_tags_df.groupby('id')['neighborhood'].transform('count')

    prop_tags_df = prop_tags_df[['id', tag, weight]].drop_duplicates() \
        .rename(columns={'id': 'proposalId', tag: 'tagId'}).sort_values(by='proposalId')

    proposalId_cnv = pd.DataFrame(data={'proposalId': proposalIds,
                                        'finalProposalId': range(0, len(proposalIds))})

    tagId_cnv = pd.DataFrame(data={'tagId': sorted(prop_tags_df.tagId.unique()),
                                   'finalTagId': range(0, prop_tags_df.tagId.nunique())})
    prop_tags_df = prop_tags_df.merge(proposalId_cnv, how='left', on='proposalId').drop(columns='proposalId')
    prop_tags_df = prop_tags_df.merge(tagId_cnv, how='left', on='tagId').drop(columns='tagId')

    prop_tag_matrix = coo_matrix((prop_tags_df.weight, (prop_tags_df.finalProposalId,
                                                        prop_tags_df.finalTagId)),
                                 shape=(len(proposalIds), prop_tags_df.finalTagId.nunique())).tocsr()

    return {'item_profiles': prop_tag_matrix,
            'tag': tag,
            'proposalId_cnv': proposalId_cnv,
            'tagId_cnv': tagId_cnv}



############# CONVERSION OF REAL IDs TO SPARSE MATRIX INDEXES #############

def get_realUserId(userId_cnv, userId):
    return userId_cnv[userId_cnv.finalUserId == userId].userId.values[0]


def get_finalUserId(userId_cnv, userId):
    return userId_cnv[userId_cnv.userId == userId].finalUserId.values[0]


def get_realProposalId(proposalId_cnv, proposalId):
    return proposalId_cnv[proposalId_cnv.finalProposalId == proposalId].proposalId.values[0]


def get_finalProposalId(proposalId_cnv, proposalId):
    return proposalId_cnv[proposalId_cnv.proposalId == proposalId].finalProposalId.values[0]


def get_realTagId(tagId_cnv, tagId):
    return tagId_cnv[tagId_cnv.finalProposalId == tagId].tagId.values[0]


def get_finalTagId(tagId_cnv, tagId):
    return tagId_cnv[tagId_cnv.tagId == tagId].finalTagId.values[0]


################# SAVING RECOMMENDATIONS AND CALCULATING METRICS #################

def gen_recommendations(rm_info, rm_train=None, rm_test=None, model_name='pop', model=None, params={},
                        c='c1', N=50, save=False):
    m = model(**params)
    if rm_train is None or rm_test is None:
        print("ERROR: must provide train and test")

    m.fit(rm_train)
    ids, scores = m.recommend(userid=range(0, rm_train.shape[0]),
                              user_items=rm_train.astype(float),
                              N=N,
                              filter_already_liked_items=True)

    recs = pd.DataFrame(data={'userId': list(chain(*[[i] * ids.shape[1] for i in range(0, ids.shape[0])])),
                              'proposalId': ids.flatten(),
                              'scores': scores.flatten()}) \
        .sort_values(by=['userId', 'scores'], ascending=[True, False]) \
        .reset_index(drop=True) \
        .replace(to_replace={'userId': dict(zip(rm_info['userId_cnv'].finalUserId,
                                                rm_info['userId_cnv'].userId)),
                             'proposalId': dict(zip(rm_info['proposalId_cnv'].finalProposalId,
                                                    rm_info['proposalId_cnv'].proposalId))})
    recs = recs[recs.proposalId != -1].reset_index(drop=True)
    if save:
        params.pop('prop_tag_matrix', None)
        params.pop('tag', None)

        recs.to_csv(f"../../data/recommendations/{c}/rec_{model_name}.csv",
                    sep=';', encoding='utf-8', index_label=False, index=False)

        try:
            f = f'../../data/recommendations/model_history.csv'
            history = pd.read_csv(f, sep=";", encoding='utf-8', parse_dates=[3])
            history = pd.concat([history,
                                 pd.DataFrame(data={'model_name': [model_name],
                                                    'c':[c],
                                                    'params':
                                                        [params[list(params.keys())[0]]] if len(params)>0 else [''],
                                                    'date': [datetime.now()]})])\
                .sort_values(by=['model_name', 'date', 'c'])
            history.to_csv(f, sep=';', encoding='utf-8', index_label=False, index=False)

        except FileNotFoundError:
            pd.DataFrame(data={'model_name': [model_name],
                               'c':[c],
                               'params': [params[list(params.keys())[0]]] if len(params)>0 else [''],
                               'date': [datetime.now()]})\
                .to_csv(f, sep=';', encoding='utf-8', index_label=False, index=False)

    return recs

def get_ratings_df(rm, rm_info):
    df = pd.DataFrame(data={'userId': list(chain(*[[i] * (j_1 - j_0)
                                                   for j_0, j_1, i in zip(rm.indptr[0:-1],
                                                                          rm.indptr[1:],
                                                                          range(len(rm.indptr)))])),
                            'proposalId': rm.indices,
                            'numComments': rm.data})\
        .replace(to_replace={'userId': dict(zip(rm_info['userId_cnv'].finalUserId,
                                                rm_info['userId_cnv'].userId)),
                             'proposalId': dict(zip(rm_info['proposalId_cnv'].finalProposalId,
                                                    rm_info['proposalId_cnv'].proposalId))})
    df.numComments = df.numComments.astype('int64')
    return df

def save_rm_train_test_info(rm_info, c='c1', train_percentage=0.8, random_state=1):

    rm_train, rm_test = train_test_split(rm_info['rm'], train_percentage=train_percentage,
                                         random_state=random_state)
    rm_train = rm_train.astype(float)
    rm_test = rm_test.astype(float)

    outdir = f'../../data/rm/{c}'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    save_npz(f"{outdir}/rm_{c}.npz", rm_info['rm'])
    save_npz(f"{outdir}/rm_{c}_train.npz", rm_train)
    save_npz(f"{outdir}/rm_{c}_test.npz", rm_test)
    rm_info['userId_cnv'].to_csv(f"{outdir}/rm_{c}_userId_cnv.csv", sep=';',
                                 encoding='utf-8', index_label=False,
                                 index=False)
    rm_info['proposalId_cnv'].to_csv(f"{outdir}/rm_{c}_proposalId_cnv.csv", sep=';',
                                     encoding='utf-8', index_label=False,
                                     index=False)
    return

def get_rm_train_test_info(c='c1'):

    outdir = f'../../data/rm/{c}'

    rm_info = dict()

    rm_info['rm'] = load_npz(f"{outdir}/rm_{c}.npz")
    rm_info['userId_cnv'] = pd.read_csv(f"{outdir}/rm_{c}_userId_cnv.csv",
                                        sep=";", encoding='utf-8')
    rm_info['proposalId_cnv'] = pd.read_csv(f"{outdir}/rm_{c}_proposalId_cnv.csv",
                                            sep=";", encoding='utf-8')
    rm_train = load_npz(f"{outdir}/rm_{c}_train.npz").astype(float)
    rm_test = load_npz(f"{outdir}/rm_{c}_test.npz").astype(float)

    return rm_info, rm_train, rm_test



################# HYPERPARAMETER TUNNING #################


def tunning_and_metrics(rm_train, rm_test, model={}, cvk=5, N=50, Nf=None, check_overfitting=False):

    if Nf is None:
        Nf=N
    model_name = list(model.keys())[0]
    algorithm = list(model.values())[0]
    params = algorithm['params']

    all_ht_metric_results_train = None
    all_ht_metric_results_test = None

    if algorithm['m'].__name__ == "ContentBasedRecommender":
        best_params = dict(zip(params.keys(), [v[0] for v in params.values()]))
        model_name2 = model_name
        model = algorithm['m'](**best_params)

    elif algorithm['m'].__name__ == "PopularityRecommender" \
            or algorithm['m'].__name__ == "PopularityNumCommentsRecommender"\
            or algorithm['m'].__name__ == "RandomRecommender":
        model_name2 = model_name
        model = algorithm['m']()

    else:
        if check_overfitting:
            send_test = rm_test
        else:
            send_test = None
        all_ht_metric_results_train, all_ht_metric_results_test, best_params =\
            hyperparameter_tunning_CV(rm_train, model_name, algorithm['m'],
                                      params=params, cvk=cvk, N=N,
                                      check_overfitting=check_overfitting, rm_test=send_test)
        model_name2 = model_name + ''.join(list(chain.from_iterable([str(v) for v, k in zip(best_params.values(),
                                                                                            best_params.keys())
                                                                     if k != 'prop_tag_matrix' and k != 'tag'])))
        model = algorithm['m'](**best_params)

    model.fit(rm_train)

    final_metrics_users = ranking_metrics_at_k(model, rm_train, rm_test, K=Nf)
    final_metric_results = pd.DataFrame(data=final_metrics_users.mean().to_dict(),
                                 index=[model_name2])

    return final_metric_results, all_ht_metric_results_train, all_ht_metric_results_test, final_metrics_users

def hyperparameter_tunning_CV(rm_train, model_name, algorithm, params=None, cvk=1, N=50,
                              check_overfitting=False, rm_test=None):

    random_seeds = random.sample(range(0, 2 ** 32 - 1), cvk)
    print('Random Seeds', random_seeds)
    metric_results = []
    metrics_test = []

    param_grid = ParameterGrid(params)
    best_ndcg, best_params = -1, {}

    for params in param_grid:
        model_name2 = model_name + ''.join(list(chain.from_iterable([str(v) for v, k in zip(params.values(),
                                                                                            params.keys())
                                                                     if k != 'prop_tag_matrix' and k != 'tag'])))
        model = algorithm(**params)
        metrics = cv_recsys(model, model_name2, rm_train, random_seeds, cvk, N)
        metric_results.append(metrics)
        if best_ndcg < metrics[f'ndcg@{N}'].values[0]:
            best_ndcg = metrics[f'ndcg@{N}'].values[0]
            best_params = params

        if check_overfitting:
            model.fit(rm_train)
            metrics_test.append(pd.DataFrame(data=ranking_metrics_at_k(model, rm_train, rm_test, K=N).mean().to_dict(),
                                             index=[model_name2]))

    metric_results = pd.concat(metric_results)
    if check_overfitting:
        metrics_test_results = pd.concat(metrics_test)
    else:
        metrics_test_results = None

    print("Best params are -> ", best_params)

    return metric_results, metrics_test_results, best_params


def cv_recsys(model, model_name, rm, random_seeds, cvk=5, N=50):
    """ Cross Validation experiment to get mean metrics"""

    metrics = pd.DataFrame(data={'precision': 0.0,
                                 'recall': 0.0,
                                 'f1': 0.0,
                                 'map': 0.0,
                                 f'ndcg@{N}': 0.0,
                                 'auc': 0.0,
                                 'mrr': 0.0}, index=[model_name])

    ndcg = []
    print(f"{model_name}")
    for i in range(0, cvk):
        print(f"Iter {i+1} ", end="")
        rm_train, rm_test = train_test_split(rm, train_percentage=0.8, random_state=random_seeds[i])
        rm_train = rm_train.astype(float)
        rm_test = rm_test.astype(float)

        model.fit(rm_train)

        metrics_small = ranking_metrics_at_k(model, rm_train, rm_test, K=N).mean().to_dict()
        metrics = metrics + pd.DataFrame(data=metrics_small, index=[model_name])

        ndcg.append(metrics_small[f'ndcg@{N}'])
    metrics = metrics / cvk
    metrics[f'std_ndcg@{N}'] = np.std(ndcg)
    if metrics[f'ndcg@{N}'].values[0] != 0:
        metrics[f'var_coef_ndcg@{N}'] = metrics[f'std_ndcg@{N}'] / abs(metrics[f'ndcg@{N}'])
    else:
        metrics[f'var_coef_ndcg@{N}'] = float('inf')

    return metrics

def calculate_num_recommendations(rm, model, N=50):
    ids_list = model.recommend(userid=range(0, rm.shape[0]),
                               user_items=rm.astype(float),
                               N=N,
                               filter_already_liked_items=True)[0]
    ids_list = [remove_nulls(ids) for ids in ids_list]
    num_recommendations = pd.Series(list(chain.from_iterable(ids_list))).value_counts()
    return num_recommendations

def remove_nulls(ids):
    ids = set(ids)
    if -1 in ids:
        ids.remove(-1)
    return list(ids)
