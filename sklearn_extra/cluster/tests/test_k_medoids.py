"""Testing for K-Medoids"""
import warnings
import numpy as np
from unittest import mock
from scipy.sparse import csc_matrix
import pytest

from sklearn.datasets import load_iris
from sklearn.metrics.pairwise import PAIRWISE_DISTANCE_FUNCTIONS
from sklearn.metrics.pairwise import euclidean_distances
from numpy.testing import assert_allclose, assert_array_equal

from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans

seed = 0
X = np.random.RandomState(seed).rand(100, 5)


def test_kmedoids_input_validation_and_fit_check():
    rng = np.random.RandomState(seed)
    # Invalid parameters
    msg = "n_clusters should be a nonnegative integer. 0 was given"
    with pytest.raises(ValueError, match=msg):
        KMedoids(n_clusters=0).fit(X)

    msg = "n_clusters should be a nonnegative integer. None was given"
    with pytest.raises(ValueError, match=msg):
        KMedoids(n_clusters=None).fit(X)

    msg = "max_iter should be a nonnegative integer. 0 was given"
    with pytest.raises(ValueError, match=msg):
        KMedoids(n_clusters=1, max_iter=0).fit(X)

    msg = "max_iter should be a nonnegative integer. None was given"
    with pytest.raises(ValueError, match=msg):
        KMedoids(n_clusters=1, max_iter=None).fit(X)

    msg = (
        r"init needs to be one of the following: "
        r".*random.*heuristic.*k-medoids\+\+"
    )
    with pytest.raises(ValueError, match=msg):
        KMedoids(init=None).fit(X)

    # Trying to fit 3 samples to 8 clusters
    msg = (
        "The number of medoids \(8\) must be less "
        "than the number of samples 5."
    )
    Xsmall = rng.rand(5, 2)
    with pytest.raises(ValueError, match=msg):
        KMedoids(n_clusters=8).fit(Xsmall)


def test_random_deterministic():
    """Random_state should determine 'random' init output."""
    rng = np.random.RandomState(seed)

    X = load_iris()["data"]
    D = euclidean_distances(X)

    medoids = KMedoids(init="random")._initialize_medoids(D, 4, rng)
    assert_array_equal(medoids, [47, 117, 67, 103])


def test_heuristic_deterministic():
    """Result of heuristic init method should not depend on rnadom state."""
    rng1 = np.random.RandomState(1)
    rng2 = np.random.RandomState(2)
    X = load_iris()["data"]
    D = euclidean_distances(X)

    medoids_1 = KMedoids(init="heuristic")._initialize_medoids(D, 10, rng1)

    medoids_2 = KMedoids(init="heuristic")._initialize_medoids(D, 10, rng2)

    assert_array_equal(medoids_1, medoids_2)


def test_update_medoid_idxs_empty_cluster():
    """Label is unchanged for an empty cluster."""
    D = np.zeros((3, 3))
    labels = np.array([0, 0, 0])
    medoid_idxs = np.array([0, 1])
    kmedoids = KMedoids(n_clusters=2)

    # Swallow empty cluster warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kmedoids._update_medoid_idxs_in_place(D, labels, medoid_idxs)

    assert_array_equal(medoid_idxs, [0, 1])


def test_kmedoids_empty_clusters():
    """When a cluster is empty, it should throw a warning."""
    rng = np.random.RandomState(seed)
    X = [[1], [1], [1]]
    kmedoids = KMedoids(n_clusters=2, random_state=rng)
    with pytest.warns(UserWarning, match="Cluster 1 is empty!"):
        kmedoids.fit(X)


@mock.patch.object(KMedoids, "_kpp_init", return_value=object())
def test_kpp_called(_kpp_init_mocked):
    """KMedoids._kpp_init method should be called by _initialize_medoids"""
    D = np.array([[0, 1], [1, 0]])
    n_clusters = 2
    rng = np.random.RandomState(seed)
    kmedoids = KMedoids()
    kmedoids.init = "k-medoids++"
    # set _kpp_init_mocked.return_value to a singleton
    initial_medoids = kmedoids._initialize_medoids(D, n_clusters, rng)

    # assert that _kpp_init was called and its result was returned.
    _kpp_init_mocked.assert_called_once_with(D, n_clusters, rng)
    assert initial_medoids == _kpp_init_mocked.return_value


def test_kmedoids_pp():
    """Initial clusters should be well-separated for k-medoids++"""
    rng = np.random.RandomState(seed)
    kmedoids = KMedoids()
    X = [
        [10, 0],
        [11, 0],
        [0, 10],
        [0, 11],
        [10, 10],
        [11, 10],
        [12, 10],
        [10, 11],
    ]
    D = euclidean_distances(X)

    centers = kmedoids._kpp_init(D, n_clusters=3, random_state_=rng)

    assert len(centers) == 3

    inter_medoid_distances = D[centers][:, centers]
    assert np.all((inter_medoid_distances > 5) | (inter_medoid_distances == 0))


def test_precomputed():
    """Test the 'precomputed' distance metric."""
    rng = np.random.RandomState(seed)
    X_1 = [[1.0, 0.0], [1.1, 0.0], [0.0, 1.0], [0.0, 1.1]]
    D_1 = euclidean_distances(X_1)
    X_2 = [[1.1, 0.0], [0.0, 0.9]]
    D_2 = euclidean_distances(X_2, X_1)

    kmedoids = KMedoids(metric="precomputed", n_clusters=2, random_state=rng)
    kmedoids.fit(D_1)

    assert_allclose(kmedoids.inertia_, 0.2)
    assert_array_equal(kmedoids.medoid_indices_, [2, 0])
    assert_array_equal(kmedoids.labels_, [1, 1, 0, 0])
    assert kmedoids.cluster_centers_ is None

    med_1, med_2 = tuple(kmedoids.medoid_indices_)
    predictions = kmedoids.predict(D_2)
    assert_array_equal(predictions, [med_1 // 2, med_2 // 2])

    transformed = kmedoids.transform(D_2)
    assert_array_equal(transformed, D_2[:, kmedoids.medoid_indices_])


def test_kmedoids_fit_naive():
    n_clusters = 3
    metric = "euclidean"

    model = KMedoids(n_clusters=n_clusters, metric=metric)
    Xnaive = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    model.fit(Xnaive)

    assert_array_equal(
        model.cluster_centers_, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    )
    assert_array_equal(model.labels_, [0, 1, 2])
    assert model.inertia_ == 0.0

    # diagonal must be zero, off-diagonals must be positive
    X_new = model.transform(Xnaive)
    for c in range(n_clusters):
        assert X_new[c, c] == 0
        for c2 in range(n_clusters):
            if c != c2:
                assert X_new[c, c2] > 0


def test_max_iter():
    """Test that warning message is thrown when max_iter is reached."""
    rng = np.random.RandomState(seed)
    X_iris = load_iris()["data"]

    model = KMedoids(
        n_clusters=10, init="random", random_state=rng, max_iter=1
    )
    msg = "Maximum number of iteration reached before"

    with pytest.warns(UserWarning, match=msg):
        model.fit(X_iris)


def test_kmedoids_iris():
    """Test kmedoids on the Iris dataset"""
    rng = np.random.RandomState(seed)
    X_iris = load_iris()["data"]

    ref_model = KMeans(n_clusters=3).fit(X_iris)

    avg_dist_to_closest_centroid = (
        ref_model.transform(X_iris).min(axis=1).mean()
    )

    for init in ["random", "heuristic", "k-medoids++"]:
        distance_metric = "euclidean"
        model = KMedoids(
            n_clusters=3, metric=distance_metric, init=init, random_state=rng
        )
        model.fit(X_iris)

        # test convergence in reasonable number of steps
        assert model.n_iter_ < (len(X_iris) // 10)

        distances = PAIRWISE_DISTANCE_FUNCTIONS[distance_metric](X_iris)
        avg_dist_to_random_medoid = np.mean(distances.ravel())
        avg_dist_to_closest_medoid = model.inertia_ / X_iris.shape[0]
        # We want distance-to-closest-medoid to be reduced from average
        # distance by more than 50%
        assert avg_dist_to_random_medoid > 2 * avg_dist_to_closest_medoid
        # When K-Medoids is using Euclidean distance,
        # we can compare its performance to
        # K-Means. We want the average distance to cluster centers
        # to be similar between K-Means and K-Medoids
        assert_allclose(
            avg_dist_to_closest_medoid, avg_dist_to_closest_centroid, rtol=0.1
        )


def test_kmedoids_fit_predict_transform():
    rng = np.random.RandomState(seed)
    model = KMedoids(random_state=rng)

    labels1 = model.fit_predict(X)
    assert len(labels1) == 100
    assert_array_equal(labels1, model.labels_)

    labels2 = model.predict(X)
    assert_array_equal(labels1, labels2)

    Xt1 = model.fit_transform(X)
    assert_array_equal(Xt1.shape, (100, model.n_clusters))

    Xt2 = model.transform(X)
    assert_array_equal(Xt1, Xt2)


def test_callable_distance_metric():
    rng = np.random.RandomState(seed)

    def my_metric(a, b):
        return np.sqrt(np.sum(np.power(a - b, 2)))

    model = KMedoids(random_state=rng, metric=my_metric)
    labels1 = model.fit_predict(X)
    assert len(labels1) == 100
    assert_array_equal(labels1, model.labels_)


def test_outlier_robustness():
    rng = np.random.RandomState(seed)
    kmeans = KMeans(n_clusters=2, random_state=rng)
    kmedoids = KMedoids(n_clusters=2, random_state=rng)

    X = [[-11, 0], [-10, 0], [-9, 0], [0, 0], [1, 0], [2, 0], [1000, 0]]

    kmeans.fit(X)
    kmedoids.fit(X)

    assert_array_equal(kmeans.labels_, [0, 0, 0, 0, 0, 0, 1])
    assert_array_equal(kmedoids.labels_, [0, 0, 0, 1, 1, 1, 1])


def test_kmedoids_on_sparse_input():
    rng = np.random.RandomState(seed)
    model = KMedoids(n_clusters=2, random_state=rng)
    row = np.array([1, 0])
    col = np.array([0, 4])
    data = np.array([1, 1])
    X = csc_matrix((data, (row, col)), shape=(2, 5))
    labels = model.fit_predict(X)
    assert len(labels) == 2
    assert_array_equal(labels, model.labels_)


from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.metrics.pairwise import pairwise_distances

digits = load_digits()
data = scale(digits.data)
n_digits = len(np.unique(digits.target))

N = 300
reduced_data = PCA(n_components=2).fit_transform(data)[:N]

metrics = ["manhattan", "euclidean", "cosine"]


@pytest.mark.parametrize("metric", metrics)
def test_methods(metric):
    rng = np.random.RandomState(seed)
    model_pam = KMedoids(
        metric=metric, n_clusters=n_digits, method="pam", random_state=rng
    ).fit(reduced_data)
    model_alt = KMedoids(
        metric=metric,
        n_clusters=n_digits,
        method="alternating",
        random_state=rng,
    ).fit(reduced_data)
    centers_pam = model_pam.cluster_centers_
    centers_alt = model_alt.cluster_centers_
    D = pairwise_distances(centers_pam, centers_alt)
    assert all(np.min(D, axis=0) / np.std(reduced_data) < 2)
