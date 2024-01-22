import pandas as pd
import numpy as np

def estimate_model_distribution(all_proposals, df_rec, df_test, proposal_attributes, attribute_values,
                                attribute_name, h=0.95, pc=0.0001, fun='count'):
    rgis_df = pd.DataFrame(data={'proposalId': all_proposals})

    # Get rank of item for user
    df_rec_r = df_rec.sort_values(by=['userId', 'proposalId', 'scores'], ascending=False)

    df_rec_r = df_rec_r.merge(df_rec_r[['userId']].reset_index().rename(columns={'index': 'r_0'}) \
                              .groupby('userId').min().reset_index()[['userId', 'r_0']],
                              how='left', on=['userId'])

    df_rec_r['r'] = df_rec_r.index - df_rec_r.r_0 + 1
    df_rec_r.drop(columns=['r_0', 'scores'], inplace=True)

    rgis_df = rgis_df.merge(df_rec_r, how='left', on='proposalId')
    rgis_df['r'] = rgis_df['r'].fillna(0)

    if fun == 'count':
        rgis_df = rgis_df.groupby('proposalId').count()[['userId']].reset_index()
        rgis = dict(zip(rgis_df['proposalId'], rgis_df['userId']))
    else:
        rgis_df = rgis_df.merge(df_test, how='left', on=['proposalId', 'userId']).rename(
            columns={'numComments': 'rel_u_i'})
        rgis_df.loc[(rgis_df.rel_u_i.isnull()) & (rgis_df.r == 0), 'rel_u_i'] = 0
        rgis_df.loc[~(rgis_df.rel_u_i.isnull()) | (rgis_df.r > 0), 'rel_u_i'] = 1

        if fun == 'bin':
            rgis_df = rgis_df.groupby('proposalId').sum()[['rel_u_i']].reset_index()
            rgis = dict(zip(rgis_df['proposalId'], rgis_df['rel_u_i']))

        elif fun == 'ndcg':
            rgis_df['ndcg'] = 0
            rgis_df.loc[rgis_df.r > 0, 'ndcg'] = (2 ** rgis_df.loc[rgis_df.r > 0, 'rel_u_i'] - 1) / np.log2(
                rgis_df.loc[rgis_df.r > 0, 'r'] + 1)
            rgis_df = rgis_df.groupby('proposalId').sum()[['ndcg']].reset_index()
            rgis = dict(zip(rgis_df['proposalId'], rgis_df['ndcg']))
        else:
            print("ERROR: fun is not defined.")
            return

    Z = np.sum(list(rgis.values()))

    pm_i = {}
    for a in attribute_values:
        pm_i[a] = 0
        # For each proposal with that value for the atribute a
        for i in proposal_attributes[proposal_attributes[attribute_name] == a].proposalId:
            pm_i[a] += rgis[i]
        pm_i[a] = h * pm_i[a] / Z + (1 - h) * pc

    Z_hat = np.sum(list(pm_i.values()))
    for a in attribute_values:
        pm_i[a] = pm_i[a]/Z_hat

    return pm_i


def GCE(proposalIds, df_rec_attributes, df_test_attributes, proposal_attributes, p_f, fun='count',
        beta=2, h=0.95, pc=0.0001):
    GCE = 0

    attribute_name = [i for i in proposal_attributes.columns if i != 'proposalId'][0]
    attribute_values = proposal_attributes[attribute_name].unique()

    p_m = estimate_model_distribution(all_proposals=proposalIds,
                                      df_rec=df_rec_attributes, df_test=df_test_attributes,
                                      proposal_attributes=proposal_attributes,
                                      attribute_values=attribute_values,
                                      attribute_name=attribute_name,
                                      h=h, pc=pc, fun=fun)
    for a in attribute_values:
        if (p_m[a] == 0):
            continue  # If a topic value has never been assigned to the proposals, it is skipped
        GCE += (p_f[a] ** beta) * (p_m[a] ** (1 - beta))
    GCE -= 1
    GCE *= 1 / (beta * (1 - beta))

    return GCE, p_m