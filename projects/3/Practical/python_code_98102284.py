student_number = 98102284
Name = 'Parsa'
Last_Name = 'Mohammadian'


possible_images_count = 2 ** (28 * 28)
print(f'Possible images count: {possible_images_count}')


parameters_count = 2 ** (28 * 28)  # because it is an arbitrary distribution
print(f'Parameters count: {parameters_count}')


def binarize(x):
    return (x > np.random.rand()).astype(int)


def sample():
    z1 = np.random.choice(disc_z1, p=np.vectorize(get_p_z1)(disc_z1))
    z2 = np.random.choice(disc_z2, p=np.vectorize(get_p_z2)(disc_z2))
    x = get_p_x_cond_z1_z2(z1, z2)
    return binarize(x)


for _ in range(5):
    x = sample()
    plt.imshow(x.reshape(28, 28))
    plt.show()




fig, axs = plt.subplots(25, 25, figsize=(50, 50))
for idx1, z1 in enumerate(disc_z1):
    for idx2, z2 in enumerate(disc_z2):
        x = get_p_x_cond_z1_z2(z1, z2)
        bin_x = binarize(x)
        axs[idx1, idx2].imshow(bin_x.reshape(28, 28))


for ax in axs.flat:
    ax.set(xlabel='x-label', ylabel='y-label')

for ax in axs.flat:
    ax.label_outer()


'''
Think about the problem like we have a latent variable z1 and z2 (which are calculated from the input image x), 
and we want to find the probability of the image x to be one of the known digits.
'''
pass


z_1_z_2 = {}
x_0_k_cond_z1_z2 = {}
x_1_k_cond_z1_z2 = {}

for z1 in disc_z1:
    for z2 in disc_z2:
        z_1_z_2[(z1, z2)] = np.log(get_p_z1(z1)) + np.log(get_p_z2(z2))
        x_1_k_cond_z1_z2[(z1, z2)] = np.log(get_p_x_cond_z1_z2(z1, z2))
        x_0_k_cond_z1_z2[(z1, z2)] = np.log(np.ones(NUM_PIXELS) - get_p_x_cond_z1_z2(z1, z2))



def log_likelihood(x):
    p = float('-inf')
    for z1 in disc_z1:
        for z2 in disc_z2:
            p_x = np.where(x == 1, x_1_k_cond_z1_z2[(z1, z2)], x_0_k_cond_z1_z2[(z1, z2)])
            p = np.logaddexp(p, np.sum(p_x) + z_1_z_2[(z1, z2)])
    return p


def bulk_log_likelihood(xs):
    return np.apply_along_axis(log_likelihood, 1, xs)


val_data_log_likelihood = bulk_log_likelihood(val_data)

avg = np.average(val_data_log_likelihood)

std = np.std(val_data_log_likelihood)


import math

threshold_range = (avg - 3 * std, avg + 3 * std)

real_data_log_likelihood = []
corrupted_log_likelihood = []

test_count = 3000

for data in test_data[0:test_count]:
    data_log_likelihood = log_likelihood(data)
    if threshold_range[0] < data_log_likelihood < threshold_range[1]:
        real_data_log_likelihood.append(data_log_likelihood)
    elif not math.isinf(data_log_likelihood):
        corrupted_log_likelihood.append(data_log_likelihood)
    else:
        raise Exception('Infinite log likelihood')

bins = test_count // 10

plt.figure(figsize=(20, 10))
plt.hist(real_data_log_likelihood, bins=bins, color='b')
plt.show()
plt.figure(figsize=(20, 10))
plt.hist(corrupted_log_likelihood, bins=bins, color='orange')
plt.show()


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
np.random.seed(2820)






class BN(object):
    """
    Bayesian Network implementation with sampling methods as a class

    Attributes
    ----------
    n: int
        number of variables

    G: dict
        Network representation as a dictionary.
        {variable:[[children],[parents]]}
    """

    def __init__(self) -> None:

        self.n = 6
        self.G = {
            'A': [['C', 'D'], []],
            'B': [['E'], []],
            'C': [['D'], ['A', 'E']],
            'D': [['F'], ['A', 'C']],
            'E': [['C'], ['B']],
            'F': [[], ['D']],
            'G': [[], ['C', 'D', 'E']]
        }
        self.topological_order = ['A', 'B', 'E', 'C', 'D', 'F']
        self.tables = {
            'A': [[1, 0.8], [0, 0.2]],
            'B': [[1, 0.55], [0, 0.45]],
            'E': [[1, 1, 0.3], [1, 0, 0.9], [0, 1, 0.7], [0, 0, 0.1]],
            'C': [[0, 0, 0, 0.3], [1, 0, 0, 0.7], [0, 1, 0, 0.5], [1, 1, 0, 0.5], [0, 0, 1, 0.85], [1, 0, 1, 0.15], [0, 1, 1, 0.95], [1, 1, 1, 0.05]],
            'D': [[0, 0, 0, 0.2], [1, 0, 0, 0.8], [0, 1, 0, 0.5], [1, 1, 0, 0.5], [0, 0, 1, 0.35], [1, 0, 1, 0.65], [0, 1, 1, 0.33], [1, 1, 1, 0.67]],
            'F': [[1, 1, 0.2], [1, 0, 0.25], [0, 1, 0.8], [0, 0, 0.75]]
        }
        self.cpt_dataframes = {
            'A': pd.DataFrame(self.tables['A'], columns=['A', 'P']),
            'B': pd.DataFrame(self.tables['B'], columns=['B', 'P']),
            'E': pd.DataFrame(self.tables['E'], columns=['E', 'B', 'P']),
            'C': pd.DataFrame(self.tables['C'], columns=['C', 'A', 'E', 'P']),
            'D': pd.DataFrame(self.tables['D'], columns=['D', 'A', 'C', 'P']),
            'F': pd.DataFrame(self.tables['F'], columns=['F', 'D', 'P'])
        }
        self.joint_distribution_dataframe = self.init_joint_distribution()
        self.node_topol_idx = {node: idx for idx,
                               node in enumerate(self.topological_order)}
        self.idx_topol_node = {idx: node for idx,
                               node in enumerate(self.topological_order)}

    def init_joint_distribution(self):
        joint_dict = {'PFinal': [1 for i in range(2**self.n)]}

        for idx, node in enumerate(self.topological_order):
            joint_dict[node] = [0 if x // 2**idx %
                                2 == 0 else 1 for x in range(0, 2**self.n)]

        joint_df = pd.DataFrame(joint_dict)

        for _, cpt in self.cpt_dataframes.items():
            joint_df = pd.merge(joint_df, cpt, on=list(
                cpt.columns[:-1]), suffixes=('', ''))
            joint_df['PFinal'] = joint_df['PFinal'] * joint_df['P']
            joint_df.drop(columns=['P'], inplace=True)

        joint_df.rename(columns={'PFinal': 'P'}, inplace=True)
        return joint_df

    def cpt(self, node, value=None) -> dict:
        """
        This is a function that returns cpt of the given node

        Parameters
        ----------
        node:
            a variable in the bayes' net

        Returns
        -------
        result: dict
            {value1:{{parent1:p1_value1, parent2:p2_value1, ...}: prob1, ...}, value2: ...}
        """

        return self.cpt_dataframes[node] if value is None else self.cpt_dataframes[node].where(self.cpt_dataframes[node][node] == value).dropna()

    def pmf(self, query, evidence) -> float:
        """
        This function gets a variable and its value as query and a list of evidences and returns probability mass function P(Q=q|E=e)

        Parameters
        ----------
        query:
            a variable and its value
            e.g. ('a', 1)
        evidence:
            list of variables and their values
            e.g. [('b', 0), ('c', 1)]

        Returns
        -------
        PMF: float
            P(query|evidence)
        """
        joint = self.joint_distribution_dataframe
        for e_var, e_val in evidence:
            joint = joint.where(joint[e_var] == e_val)
        denominator = joint['P'].sum()
        for q_var, q_val in query:
            joint = joint.where(joint[q_var] == q_val)
        numerator = joint['P'].sum()
        return numerator / denominator

    def sampling(self, query, evidence, sampling_method, num_iter, num_burnin=1e2) -> float:
        """
        Parameters
        ----------
        query: list
            list of variables an their values
            e.g. [('a', 0), ('e', 1)]
        evidence: list
            list of observed variables and their values
            e.g. [('b', 0), ('c', 1)]
        sampling_method:
            "Prior", "Rejection", "Likelihood Weighting", "Gibbs"
        num_iter:
            number of the generated samples
        num_burnin:
            (used only in gibbs sampling) number of samples that we ignore at the start for gibbs method to converge

        Returns
        -------
        probability: float
            approximate P(query|evidence) calculated by sampling
        """

        if sampling_method == "Prior":
            samples = self.prior_sampling(num_iter)
            for e in evidence:
                samples = samples.where(samples[e[0]] == e[1]).dropna()
            compatible = samples.copy()
            for q_var, q_val in query:
                compatible = compatible.where(
                    compatible[q_var] == q_val).dropna()
            return compatible.shape[0] / samples.shape[0] if samples.shape[0] != 0 else 0

        elif sampling_method == "Rejection":
            samples = self.rejection_sampling(evidence, num_iter)
            compatible = samples.copy()
            for q_var, q_val in query:
                compatible = compatible.where(
                    compatible[q_var] == q_val).dropna()
            return compatible.shape[0] / samples.shape[0] if samples.shape[0] != 0 else 0

        elif sampling_method == "Likelihood Weighting":
            samples = self.likelihood_weighting(evidence, num_iter)
            compatible = samples.copy()
            for q_var, q_val in query:
                compatible = compatible.where(
                    compatible[q_var] == q_val).dropna()
            return compatible['W'].sum() / samples['W'].sum() if samples.shape[0] != 0 else 0

        elif sampling_method == "Gibbs":
            samples = self.gibbs_sampling(evidence, num_iter, int(num_burnin))
            compatible = samples.copy()
            for q_var, q_val in query:
                compatible = compatible.where(
                    compatible[q_var] == q_val).dropna()
            return compatible.shape[0] / samples.shape[0] if samples.shape[0] != 0 else 0

        else:
            raise ValueError("Sampling method is not supported")

    def prior_sampling(self, num_iter):
        samples = []
        for _ in range(num_iter):
            sample = []
            for node in self.topological_order:
                cpt = self.cpt(node)
                for idx, sampled_value in enumerate(sample):
                    sampled_node = self.topological_order[idx]
                    if sampled_node in self.G[node][1]:
                        cpt = cpt.where(cpt[sampled_node] ==
                                        sampled_value).dropna()
                if cpt.index.shape[0] != 0:
                    random_row = np.random.choice(
                        cpt.index, p=cpt['P']/cpt['P'].sum())
                    sample.append(cpt.loc[random_row, node])
            samples.append(sample)
        return pd.DataFrame(samples, columns=self.topological_order)

    def rejection_sampling(self, evidence, num_iter):
        samples = []
        for _ in range(num_iter):
            sample = []
            for node in self.topological_order:
                cpt = self.cpt(node)
                for e_var, e_val in evidence:
                    if e_var in self.G[node][1]:
                        cpt = cpt.where(cpt[e_var] == e_val).dropna()
                if cpt.index.shape[0] != 0:
                    random_row = np.random.choice(
                        cpt.index, p=cpt['P']/cpt['P'].sum())
                    sample.append(cpt.loc[random_row, node])
            samples.append(sample)
        return pd.DataFrame(samples, columns=self.topological_order)

    def likelihood_weighting(self, evidence, num_iter):
        samples = []
        evidence_unzip = list(zip(*evidence))
        for _ in range(num_iter):
            sample = []
            w = 1.0
            for node in self.topological_order:
                if node in evidence_unzip[0]:
                    e_val = evidence_unzip[1][evidence_unzip[0].index(node)]
                    sample.append(e_val)
                    w *= self.cpt(node, e_val)['P'].sum()
                else:
                    cpt = self.cpt(node)
                    for e_var, e_val in evidence:
                        if e_var in self.G[node][1]:
                            cpt = cpt.where(cpt[e_var] == e_val).dropna()
                    if cpt.index.shape[0] != 0:
                        random_row = np.random.choice(
                            cpt.index, p=cpt['P']/cpt['P'].sum())
                        sample.append(cpt.loc[random_row, node])
            samples.append(sample + [w])
        return pd.DataFrame(samples, columns=self.topological_order + ['W'])

    def gibbs_sampling(self, evidence, num_iter, num_burnin):
        evidence_dict = {e[0]: e[1] for e in evidence}
        samples = []

        init_sample = []
        for node in self.topological_order:
            if node in evidence_dict.keys():
                init_sample.append(evidence_dict[node])
            else:
                init_sample.append(np.random.choice([0, 1]))
        samples.append(init_sample)

        for _ in range(num_burnin+num_iter):
            sample = []
            for node in self.topological_order:
                if node in evidence_dict.keys():
                    sample.append(evidence_dict[node])
                else:
                    children = self.G[node][0]
                    parents = self.G[node][1]
                    parents_of_children = []
                    for child in children:
                        parents_of_child = self.G[child][1]
                        parents_of_children.extend(parents_of_child)
                    probs = [1.0, 1.0]
                    values = samples[-1][:]
                    for i in range(2):
                        values[self.node_topol_idx[node]] = i
                        cpt = self.cpt(node, i)
                        for p in parents:
                            cpt = cpt.where(cpt[p] == values[self.node_topol_idx[p]]).dropna()
                        probs[i] *= cpt['P'].sum()
                        for child in children:
                            child_cpt=self.cpt(child)
                            for p in self.G[child][1]:
                                child_cpt=child_cpt.where(
                                    child_cpt[p] == values[self.node_topol_idx[p]]).dropna()
                            probs[i] *= child_cpt['P'].sum()
                    sample.append(np.random.choice([0, 1], p=probs))
            samples.append(sample)

        return pd.DataFrame(samples[num_burnin:], columns=self.topological_order)










SAMPLING_NUMBERS = [100, 500, 1000, 3000, 10000, 50000]
SAMPLING_METHODS = ['Prior', 'Rejection', 'Likelihood Weighting', 'Gibbs']
BURNIN_VALUES = [10, 100, 500, 1000]


def plot_error(sampling_method, num_samples, error_list, axs, idx):
    axs[idx//2, idx % 2].plot(num_samples, error_list, '-o')
    axs[idx//2, idx % 2].set_title(sampling_method)


def set_labels(axs):
    for ax in axs.flat:
        ax.set(xlabel='Number of Samples', ylabel='Error')
        ax.label_outer()


queries = [
    {
        'q': [('F', 1)],
        'e': [('A', 1), ('E', 0)]
    },
    {
        'q': [('C', 0), ('B', 1)],
        'e': [('F', 1), ('D', 0)]
    }
]

bn = BN()
pass


queries_error_lists = []
for query in queries:
    errors_list = []
    pmf = bn.pmf(query['q'], query['e'])
    for idx, sampling_method in enumerate(SAMPLING_METHODS):
        error_list = []
        for num_iter in SAMPLING_NUMBERS:
            approximate = bn.sampling(
                query['q'], query['e'], sampling_method, num_iter)
            error_list.append(abs(pmf - approximate))
        errors_list.append(error_list)
    queries_error_lists.append(errors_list)

%store queries_error_lists


%store -r queries_error_lists

for error_lists in queries_error_lists:
    fig, axs = plt.subplots(2, 2, figsize=(20, 10))
    for idx, error_list in enumerate(error_lists):
        plot_error(SAMPLING_METHODS[idx],
                   SAMPLING_NUMBERS, error_list, axs, idx)
    set_labels(axs)
    plt.show()




queries_error_lists_gibbs = []
for query in queries:
    errors_list = []
    pmf = bn.pmf(query['q'], query['e'])
    for idx, burnin_value in enumerate(BURNIN_VALUES):
        error_list = []
        for num_iter in SAMPLING_NUMBERS:
            approximate = bn.sampling(
                query['q'], query['e'], 'Gibbs', num_iter, burnin_value)
            error_list.append(abs(pmf - approximate))
        errors_list.append(error_list)
    queries_error_lists_gibbs.append(errors_list)

%store queries_error_lists_gibbs


%store -r queries_error_lists_gibbs

for error_lists in queries_error_lists_gibbs:
    fig, axs = plt.subplots(2, 2, figsize=(20, 10))
    for idx, error_list in enumerate(error_lists):
        plot_error(f'Gibbs {BURNIN_VALUES[idx]}',
                   SAMPLING_NUMBERS, error_list, axs, idx)
    set_labels(axs)
    plt.show()








def get_mean_towers_coor(time_step: int, tower_records: list):
    x = np.average([tower_coor[0] for tower_coor in tower_records[time_step]])
    y = np.average([tower_coor[1] for tower_coor in tower_records[time_step]])
    return x, y


def P_coor0(coor0):
    x0, y0 = coor0
    return scipy.stats.multivariate_normal.pdf([x0, y0],
                                               mean=moving_model.get('Peurto_coordinates'), cov=moving_model.get('INIT_COV'))



def P_coor_given_prevCoor(coor, prev_coor):
    x, y = abs(coor[0] - prev_coor[0]), abs(coor[1] - prev_coor[1])
    p_x = scipy.stats.expon.pdf(x, scale=moving_model.get('X_STEP'))
    p_y = scipy.stats.expon.pdf(y, scale=moving_model.get('Y_STEP'))
    return 0.5 * p_x * p_y


def P_towerCoor_given_coor(tower_coor, tower_std, coor):
    tower_x, tower_y = tower_coor
    x, y = coor
    p_x = scipy.stats.norm.pdf(tower_x, loc=x, scale=tower_std)
    p_y = scipy.stats.norm.pdf(tower_y, loc=y, scale=tower_std)
    return p_x, p_y


def P_record_given_coor(rec, coor, towers_info):
    p_x = 1.0
    p_y = 1.0
    for i, r in enumerate(rec, start=1):
        x, y = P_towerCoor_given_coor(r, towers_info[f'{i}']['std'], coor)
        p_x *= x
        p_y *= y
    return p_x, p_y


print(real_coordinates)
print(moving_model)
print(P_coor_given_prevCoor([1087.2334169025748, -57.16536812999969], [1044.9362406077148, 72.34399023381683]))
P_towerCoor_given_coor([10, 10], 1, [10, 10])
print(P_towerCoor_given_coor((1405, 600), 60, (1044.936241, 72.343990)))
towers_info









max_Px, max_Py = 0, 0
interval, step = 20, 5

best_x0, best_y0 = None, None
best_x1, best_y1 = None, None

towers_mean_x1, towers_mean_y1 = get_mean_towers_coor(1, tower_records)

for x0 in range(int(coor0_estimations[-1][0] - interval), int(coor0_estimations[-1][0] + interval), step):
    for y0 in range(int(coor0_estimations[-1][1] - interval), int(coor0_estimations[-1][1] + interval), step):

        for x1 in range(int(towers_mean_x1 - interval), int(towers_mean_x1 + interval), step):
            for y1 in range(int(towers_mean_y1 - interval), int(towers_mean_y1 + interval), step):

                coor0 = (x0, y0)
                coor1 = (x1, y1)

                rec0 = tower_records[0]
                rec1 = tower_records[1]



                P_c0 = P_coor0(coor0)
                P_rec0_given_x0, P_rec0_given_y0 = P_record_given_coor(
                    rec0, coor0, towers_info)
                P_c1_given_c0 = P_coor_given_prevCoor(coor1, coor0)
                P_rec1_given_x1, P_rec1_given_y1 = P_record_given_coor(
                    rec1, coor1, towers_info)

                Px = P_rec0_given_x0 * P_c1_given_c0 * P_rec1_given_x1 * P_c0
                Py = P_rec0_given_y0 * P_c1_given_c0 * P_rec1_given_y1 * P_c0

                if Px > max_Px:
                    best_x0 = x0
                    best_x1 = x1
                    max_Px = Px

                if Py > max_Py:
                    best_y0 = y0
                    best_y1 = y1
                    max_Py = Py


coor0_estimations.append((best_x0, best_y0))
coor1_estimations.append((best_x1, best_y1))






max_Px, max_Py = 0, 0
interval, step = 20, 5

best_x0, best_y0 = None, None
best_x1, best_y1 = None, None
best_x2, best_y2 = None, None

towers_mean_x2, towers_mean_y2 = get_mean_towers_coor(2, tower_records)

for x0 in range(int(coor0_estimations[-1][0] - interval), int(coor0_estimations[-1][0] + interval), step):
    for y0 in range(int(coor0_estimations[-1][1] - interval), int(coor0_estimations[-1][1] + interval), step):

        coor0 = (x0, y0)
        rec0 = tower_records[0]
        P_c0 = P_coor0(coor0)
        P_rec0_given_x0, P_rec0_given_y0 = P_record_given_coor(
            rec0, coor0, towers_info)

        for x1 in range(int(coor1_estimations[-1][0] - interval), int(coor1_estimations[-1][0] + interval), step):
            for y1 in range(int(coor1_estimations[-1][1] - interval), int(coor1_estimations[-1][1] + interval), step):

                coor1 = (x1, y1)
                rec1 = tower_records[1]
                P_c1_given_c0 = P_coor_given_prevCoor(coor1, coor0)
                P_rec1_given_x1, P_rec1_given_y1 = P_record_given_coor(
                    rec1, coor1, towers_info)

                for x2 in range(int(towers_mean_x2 - interval), int(towers_mean_x2 + interval), step):
                    for y2 in range(int(towers_mean_y2 - interval), int(towers_mean_y2 + interval), step):

                        coor2 = (x2, y2)
                        rec2 = tower_records[2]
                        P_c2_given_c1 = P_coor_given_prevCoor(coor2, coor1)
                        P_rec2_given_x2, P_rec2_given_y2 = P_record_given_coor(
                            rec2, coor2, towers_info)

                        Px = P_rec0_given_x0 * P_c1_given_c0 * P_rec1_given_x1 * \
                            P_c2_given_c1 * P_rec2_given_x2 * P_c0
                        Py = P_rec0_given_y0 * P_c1_given_c0 * P_rec1_given_y1 * \
                            P_c2_given_c1 * P_rec2_given_y2 * P_c0

                        if Px > max_Px:
                            best_x0 = x0
                            best_x1 = x1
                            best_x2 = x2
                            max_Px = Px

                        if Py > max_Py:
                            best_y0 = y0
                            best_y1 = y1
                            best_y2 = y2
                            max_Py = Py

coor0_estimations.append((best_x0, best_y0))
coor1_estimations.append((best_x1, best_y1))
coor2_estimations.append((best_x2, best_y2))


print(f'real_coor0: {real_coor(0)} - Estimated_coor0: {best_x0, best_y0}')
print(f'Estimation_error: {dist((best_x0, best_y0), real_coor(0))}')
print()
print(f'real_coor1: {real_coor(1)} - Estimated_coor1: {best_x1, best_y1}')
print(f'Estimation_error: {dist((best_x1, best_y1), real_coor(1))}')
print()
print(f'real_coor2: {real_coor(2)} - Estimated_coor2: {best_x2, best_y2}')
print(f'Estimation_error: {dist((best_x2, best_y2), real_coor(2))}')



max_Px, max_Py = 0, 0
interval, step = 20, 5

best_x0, best_y0 = None, None
best_x1, best_y1 = None, None
best_x2, best_y2 = None, None
best_x3, best_y3 = None, None

towers_mean_x3, towers_mean_y3 = get_mean_towers_coor(3, tower_records)

for x0 in range(int(coor0_estimations[-1][0] - interval), int(coor0_estimations[-1][0] + interval), step):
    for y0 in range(int(coor0_estimations[-1][1] - interval), int(coor0_estimations[-1][1] + interval), step):

        coor0 = (x0, y0)
        rec0 = tower_records[0]
        P_c0 = P_coor0(coor0)
        P_rec0_given_x0, P_rec0_given_y0 = P_record_given_coor(
            rec0, coor0, towers_info)

        for x1 in range(int(coor1_estimations[-1][0] - interval), int(coor1_estimations[-1][0] + interval), step):
            for y1 in range(int(coor1_estimations[-1][1] - interval), int(coor1_estimations[-1][1] + interval), step):

                coor1 = (x1, y1)
                rec1 = tower_records[1]
                P_c1_given_c0 = P_coor_given_prevCoor(coor1, coor0)
                P_rec1_given_x1, P_rec1_given_y1 = P_record_given_coor(
                    rec1, coor1, towers_info)
                    
                for x2 in range(int(coor2_estimations[-1][0] - interval), int(coor2_estimations[-1][0] + interval), step):
                    for y2 in range(int(coor2_estimations[-1][1] - interval), int(coor2_estimations[-1][1] + interval), step):
                        
                        coor2 = (x2, y2)
                        rec2 = tower_records[2]
                        P_c2_given_c1 = P_coor_given_prevCoor(coor2, coor1)
                        P_rec2_given_x2, P_rec2_given_y2 = P_record_given_coor(
                            rec2, coor2, towers_info)

                        for x3 in range(int(towers_mean_x3 - interval), int(towers_mean_x3 + interval), step):
                            for y3 in range(int(towers_mean_y3 - interval), int(towers_mean_y3 + interval), step):

                                coor3 = (x3, y3)
                                rec3 = tower_records[3]
                                P_c3_given_c2 = P_coor_given_prevCoor(
                                    coor3, coor2)
                                P_rec3_given_x3, P_rec3_given_y3 = P_record_given_coor(
                                    rec3, coor3, towers_info)

                                Px = P_rec0_given_x0 * P_c1_given_c0 * P_rec1_given_x1 * \
                                    P_c2_given_c1 * P_rec2_given_x2 * P_c3_given_c2 * P_rec3_given_x3 * P_c0
                                Py = P_rec0_given_y0 * P_c1_given_c0 * P_rec1_given_y1 * \
                                    P_c2_given_c1 * P_rec2_given_y2 * P_c3_given_c2 * P_rec3_given_y3 * P_c0

                                if Px > max_Px:
                                    best_x0 = x0
                                    best_x1 = x1
                                    best_x2 = x2
                                    best_x3 = x3
                                    max_Px = Px

                                if Py > max_Py:
                                    best_y0 = y0
                                    best_y1 = y1
                                    best_y2 = y2
                                    best_y3 = y3
                                    max_Py = Py


coor0_estimations.append((best_x0, best_y0))
coor1_estimations.append((best_x1, best_y1))
coor2_estimations.append((best_x2, best_y2))
coor3_estimations.append((best_x3, best_y3))


print(f'real_coor0: {real_coor(0)} - Estimated_coor0: {best_x0, best_y0}')
print(f'Estimation_error: {dist((best_x0, best_y0), real_coor(0))}')




plt.figure(figsize=(20, 10))
plt.plot(list(real_coordinates['X']), label='real X')
plt.plot(list(real_coordinates['Y']), label='real Y')
plt.plot([coor0_estimations[0][0]], marker='o' , label='estimated X 1 record')
plt.plot([coor0_estimations[0][1]], marker='o' , label='estimated Y 1 record')
plt.plot([coor0_estimations[1][0], coor1_estimations[0][0]], label='estimated X 2 records')
plt.plot([coor0_estimations[1][1], coor1_estimations[0][1]], label='estimated Y 2 records')
plt.plot([coor0_estimations[2][0], coor1_estimations[1][0], coor2_estimations[0][0]], label='estimated X 3 records')
plt.plot([coor0_estimations[2][1], coor1_estimations[1][1], coor2_estimations[0][1]], label='estimated Y 3 records')
plt.plot([coor0_estimations[3][0], coor1_estimations[2][0], coor2_estimations[1][0], coor3_estimations[0][0]], label='estimated X 4 records')
plt.plot([coor0_estimations[3][1], coor1_estimations[2][1], coor2_estimations[1][1], coor3_estimations[0][1]], label='estimated Y 4 records')
plt.legend()
pass





