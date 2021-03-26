import time
import numpy as np
import matplotlib.pyplot as plt


def load_rankings(filename="rankings.txt"):
    """
    Load all student rankings from a file

    Parameters
    ----------
    filename: string
        Path to a file
    
    Returns
    -------
    superheros: A list of superheros in alphabetical order
    raters: dictionary( 
        string (Ranker's name): list (This person's ranking as a list of numbers
                                      corresponding to the indices in superheros)
    )
    """
    superhero_to_num = {}
    raters = {}
    superheros = []
    fin = open(filename)
    lines = [L.rstrip() for L in fin.readlines()]
    fin.close()
    i = 0
    N = 9
    while (i+1)*N <= len(lines):
        rater = lines[i*N]
        rankings = lines[i*N+1:(i+1)*N]
        if len(superhero_to_num) == 0:
            superheros = sorted(rankings)
            for k, name in enumerate(superheros):
                superhero_to_num[name] = k
        raters[rater] = [superhero_to_num[superhero] for superhero in rankings]
        i += 1
    return superheros, raters


def plot_mds_distances(raters, random_state=0):
    """
    Compute all pairwise Kendall-Tau distances and plot a dimension 
    reduction from the Kendall-Tau metric space to 2D to visualize how
    similar different raters are

    Parameters
    ----------
    raters: dictionary( 
        string (Ranker's name): list (This person's ranking as a list of numbers
                                      corresponding to the indices in superheros)
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


def kendall_tau(r1, r2):
    """
    An O(N log N) algorithm for computing the Kendall-Tau Distance

    Parameters
    ----------
    r1: List of N elements
        A permutation of the elements 0, 1, 2, ..., N corresponding 
        to the first rating
    r2: List of N elements
        A permutation of the elements 0, 1, 2, .., N corresponding to 
        the second rating
    
    Returns
    -------
    The Kendall-Tau distance between r1 and r2
    """
    pass ## TODO: Get rid of pass and fill this in


## TODO: Fill everything else in!