import time
import numpy as np
import matplotlib.pyplot as plt


def load_permutations(filename="preferences.csv"):
    """
    Load all student permutations from a file

    Parameters
    ----------
    filename: string
        Path to a file
    
    Returns
    -------
    animals: A list of animals in alphabetical order
    raters: dictionary( 
        string (Ranker's name): list (This person's permutation as a list of numbers
                                      corresponding to the indices in animals)
    )
    """
    raters = {}
    fin = open(filename)
    lines = fin.readlines()
    fin.close()
    animals = [s.rstrip().replace("\"", "") for s in lines[0].split(",")[1::]]
    for line in lines[1::]:
        fields = line.split(",")
        rater = fields[0].replace("\"", "")
        fields = [int(f) for f in fields[1::]]
        raters[rater] = [0]*len(fields)
        for i, x in enumerate(fields):
            raters[rater][x-1] = i
    return animals, raters


def plot_mds_distances(raters, random_state=0):
    """
    Compute all pairwise Kendall-Tau distances and plot a dimension 
    reduction from the Kendall-Tau metric space to 2D to visualize how
    similar different raters are

    Parameters
    ----------
    raters: dictionary( 
        string (Ranker's name): list (This person's permutation as a list of numbers
                                      corresponding to the indices in animals)
    random_state: int
        A seed to determine which random isometry to use for MDS
    """
    from sklearn.manifold import MDS
    N = len(raters)
    D = np.zeros((N, N))
    rlist = [r for r in raters]
    for i, rater1 in enumerate(rlist):
        for j in range(i+1, N):
            rater2 = rlist[j]
            D[i, j] = kendall_tau(raters[rater1], raters[rater2])
    D = D+D.T
    embedding = MDS(n_components=2, dissimilarity='precomputed', random_state=random_state)
    X = embedding.fit_transform(D)
    plt.scatter(X[:, 0], X[:, 1])
    for i, r in enumerate(rlist):
        plt.text(X[i, 0], X[i, 1], r)
    plt.title("MDS Projected Kendall-Tau Distances")


def kendall_tau(p1, p2):
    """
    An O(N log N) algorithm for computing the Kendall-Tau Distance

    Parameters
    ----------
    p1: List of N elements
        A permutation of the elements 0, 1, 2, ..., N corresponding 
        to the first rating
    p2: List of N elements
        A permutation of the elements 0, 1, 2, .., N corresponding to 
        the second rating
    
    Returns
    -------
    The Kendall-Tau distance between permutation p1 and p2
    """
    pass ## TODO: Get rid of pass and fill this in


## TODO: Fill everything else in!
