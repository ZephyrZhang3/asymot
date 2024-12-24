import random

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils
import torch.utils.data
from PIL import Image
from torch.utils.data import TensorDataset


def hello():
    print("Hello from asymot!")


def set_random_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)


def load_dataset(name):
    pass


@torch.no_grad()
def get_ddim_path(
    x,
    dm_scheduler,
    reverse: bool,
    pivotal_list=None,
):
    ddim_path = [x.clone()]

    if pivotal_list is None:
        loop_range = range(dm_scheduler.config.num_train_timesteps)
    else:
        loop_range = range(
            min(dm_scheduler.config.num_train_timesteps, pivotal_list[-1])
        )
    for t in loop_range:
        x_t = ddim_path[-1]
        noise = torch.randn_like(x_t)
        timestep = torch.Tensor([t]).long()
        x_t_next = dm_scheduler.add_noise(x_t, noise, timestep)

        if pivotal_list is None or (t + 1) in pivotal_list:
            ddim_path.append(x_t_next)

    if reverse:
        ddim_path = ddim_path[::-1]

    return torch.stack(ddim_path, dim=0)


@torch.no_grad()
def get_ddib_path(x, y, dm_scheduler, pivotal_list=None, skip_last=True):
    x2g_path = get_ddim_path(x, dm_scheduler, reverse=False, pivotal_list=pivotal_list)

    g2y_path = get_ddim_path(y, dm_scheduler, reverse=True, pivotal_list=pivotal_list)

    if skip_last:
        g2y_path = g2y_path[1:]  # just using x's last pivotal point

    return torch.cat([x2g_path, g2y_path], dim=0)


@torch.no_grad()
def get_flow_path(x, y, total_step=1000):
    flow_path = []
    for i in range(total_step):
        t = i / total_step
        flow_path.append((1 - t) * x + t * y)

    return flow_path


def get_schodier_path(source, target, total_step=1000):
    pass


def h5py_to_dataset(path, img_size=64):
    with h5py.File(path, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]

        # Get the data
        data = list(f[a_group_key])
    with torch.no_grad():
        dataset = (
            2
            * (torch.tensor(np.array(data), dtype=torch.float32) / 255.0).permute(
                0, 3, 1, 2
            )
            - 1
        )
        dataset = F.interpolate(dataset, img_size, mode="bilinear")

    return TensorDataset(dataset, torch.zeros(len(dataset)))


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 3D numpy array with RGB channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGB buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape((h, w, 3))

    return buf


def fig2tensor(fig):
    rgb_buf = fig2data(fig)
    img_tensor = torch.from_numpy(rgb_buf)
    img_tensor = img_tensor.permute(2, 0, 1)
    img_tensor = img_tensor.float().div(255)
    return img_tensor


def fig2img(fig):
    buf = fig2data(fig)
    w, h, c = buf.shape
    return Image.frombytes("RGB", (w, h), buf.tostring())


def cod_prob_bound(
    dataset,
    epsilon,
    query_point=None,
    distance_type="euclidean",
    n=2,
    p=1,
):
    RV = _reletive_variance(dataset, query_point, distance_type, p)
    RV_weight = np.pow(((2 / (np.pow((1 + epsilon), p) - 1)) + 1), 2)
    prob_bound = np.pow(max(0, 1 - RV_weight * RV), n)
    # print(f"{n=} {p=} {epsilon=} {distance_type=}\n{RV=}\n{prob_bound=}")
    return prob_bound


def estimate_cod_prob_bound(
    dataset,
    epsilon,
    num_observed_samples=1000,
    query_point=None,
    distance_type="euclidean",
    n=2,
    p=1,
):
    r = num_observed_samples

    RV = _estimate_reletive_variance(dataset, r, query_point, distance_type, p)
    RV_weight = np.pow(((2 / (np.pow((1 + epsilon), p) - 1)) + 1), 2) * (1 - 1 / r**2)
    prob_bound = np.pow(max(0, 1 - 1 / r - RV_weight * RV), n)
    # print(f"{n=} {r=} {p=} {epsilon=} {distance_type=}\n{RV=}\n{prob_bound=}")
    return prob_bound


def _reletive_variance(dataset, query_point=None, distance_type="euclidean", p=1):
    distance = []
    if isinstance(dataset, torch.utils.data.Dataset):
        for x, _ in dataset:
            if query_point is None:
                query_point = torch.zeros_like(x)
            distance.append(np.pow(_distance(x, query_point, distance_type), p))
    elif isinstance(dataset, torch.Tensor):
        for x in dataset:
            if query_point is None:
                query_point = torch.zeros_like(x)
            distance.append(np.pow(_distance(x, query_point, distance_type), p))
    else:
        raise ValueError("Invalid dataset type")

    distance = np.array(distance)
    mean = distance.mean()

    variance = distance.var()
    return variance / mean**2


def _estimate_reletive_variance(
    dataset, num_observed_samples, query_point=None, distance_type="euclidean", p=1
):
    indices = np.random.choice(len(dataset), num_observed_samples, replace=False)

    distance = []
    if isinstance(dataset, torch.utils.data.Dataset):
        observed_dataset = torch.utils.data.Subset(dataset, indices)
        for x, _ in observed_dataset:
            if query_point is None:
                query_point = torch.zeros_like(x)
            distance.append(np.pow(_distance(x, query_point, distance_type), p))
    elif isinstance(dataset, torch.Tensor):
        observed_dataset = dataset[indices]
        for x in observed_dataset:
            if query_point is None:
                query_point = torch.zeros_like(x)
            distance.append(np.pow(_distance(x, query_point, distance_type), p))
    else:
        raise ValueError("Invalid dataset type")

    distance = np.array(distance)
    mean = distance.mean()

    variance = distance.var(ddof=1)

    return variance / mean**2


def _distance(x, y, type="euclidean"):
    if type == "euclidean":
        return np.linalg.norm(x - y)
    elif type == "cosine":
        return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    else:
        raise ValueError("Invalid distance type")
