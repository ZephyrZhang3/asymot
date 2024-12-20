from utils import hello, set_random_seed


def main():
    set_random_seed(42)
    hello()
    # setup hyper parameters
    DATASET, DATASET_PATH = "fmnist2mnist", "./data/"
    m = dim = 28 * 28
    n = num_samples = None
    r = num_observed_samples = 1000
    epsilon = 0.2
    p = 1
    distance_type = "euclidean"
    # load dataset(FMNIST->MNIST[√], FACE->COMIC)

    # generate intermediate datasets: ddim[√], flow, schodinger bridge, etc.

    # compute concentration of distance probability

    # plot results


if __name__ == "__main__":
    main()
