from torch.utils.data import Dataset
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
import numpy as np
import scipy
import json
from sklearn.utils import check_array
import pandas as pd
import warnings

class TorchDataset(Dataset):
    """
    X : 2D array
    y : 2D array. one-hot encoded 형태
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        x, y = self.x[index], self.y[index]
        return x, y
    
class PredictDataset(Dataset):
    """
    numpy 배열형태의 데이터 처리
    """
    def __init__(self, x):
        self.x = x
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        x = self.x[index]
        return x

def create_sampler(weights, y_train):
    """
    가중치가 주어지는 경우 이에 해당하는 sampler 만듬
    -----
    weights : 0, 1, dict or iterable
    0 : 가중치 없음
    1 : classification only, inverse frequecy 형태로 balanced 됨
    dict : key는 클래스, value는 샘플링할 가중치
    iterable : list나 np.array는 길이 같아야 함.
    
    y_train : np.array
    """
    if isinstance(weights, int):
        if weights == 0:
            need_shuffle = True
            sampler = None
        elif weights == 1:
            need_shuffle = False
            class_sample_count = np.array(
                [len(np.where(y_train == t)[0]) for t in np.unique(y_train)]
            )
            # class 빈도 순으로 가중치 부여
            weights = 1.0 / class_sample_count
            
            sample_weight = np.array([weights[t] for t in y_train])
            
            sample_weight = torch.from_numpy(sample_weight)
            sample_weight = sample_weight.double()
            sampler = WeightedRandomSampler(sample_weight, len(sample_weight))
        else:
            raise ValueError("Weights는 0, 1, 또는 dictionary 나 리스트가 되어야 함")
    elif isinstance(weights, dict):
        # class 당 사용자가 정한 가중치
        need_shuffle = False
        sample_weight = np.array([weights[t] for t in y_train])
        sampler = WeightedRandomSampler(sample_weight, len(sample_weight))
    else:
        if len(weights) != len(y_train):
            raise ValueError("커스텀 가중치는 train set의 개수랑 맞아야 한다.")
        need_shuffle = False
        sample_weight = np.array(weights)
        sampler = WeightedRandomSampler(sample_weight, len(sample_weight))
    return need_shuffle, sampler

def create_dataloaders(
    X_train, y_train, eval_set, weights, batch_size, num_workers, drop_last, pin_memory
):
    """
    subsampling을 활용하거나 활용하지 않는 Dataloader를 만든다.
    X_train : np.ndarray
    y_train : np.array
    eval_set : list of tuple.(X, y) 형태
    weights : 0, 1, dict or iterable
        0 : 가중치 없음
        1 : classification only, inverse frequecy 형태로 balanced 됨
        dict : 키들은 클래스, 값들은 샘플링할 가중치인 dictionary
        iterable : list나 np.array는 길이 같아야 함. 
    batch_size : int
    num_workers (int, optional) – how many subprocesses to use for data loading. 
        0 means that the data will be loaded in the main process. (default: 0)
    drop_last : bool
        batch를 구성하는 경우 애매하게 남을때 나머지 남은 것들을 버릴지 말지 결정
    pin_memory : bool
        GPU 메모리에 얼마나 할당할지 결정
    -------
    Returns
    train_dataloader, valid_dataloader : torch.DataLoader, torch.DataLoader
    """
    need_shuffle, sampler = create_sampler(weights, y_train)
    
    train_dataloader = DataLoader(
        TorchDataset(X_train.astype(np.float32), y_train),
        batch_size=batch_size,
        sampler=sampler,
        shuffle=need_shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory
    )
    
    valid_dataloader = []
    for X, y in eval_set:
        valid_dataloader.append(
            DataLoader(
                TorchDataset(X.astype(np.float32), y),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory
            )
        )
    return train_dataloader, valid_dataloader

def create_explain_matrix(input_dim, cat_emb_dim, cat_idxs, post_embed_dim):
    """
    input_dim : input 차원
    cat_emb_dim : int or list of int
        int : size of embedding for all categorical feature
        list of int : size of embedding for each categorical feature
    cat_idxs : list of int
        Initial position of categorical features
    post_embed_dim : int
        Post embedding inputs dimension
    """
    if isinstance(cat_emb_dim, int):
        all_emb_impact = [cat_emb_dim - 1]*len(cat_idxs)
    else:
        all_emb_impact = [emb_dim - 1 for emb_dim in cat_emb_dim]
    acc_emb = 0
    nb_emb = 0
    indices_trick = []
    for i in range(input_dim):
        if i not in cat_idxs:
            indices_trick.append([i+acc_emb])
        else:
            indices_trick.append(
                range(i+acc_emb, i+acc_emb+all_emb_impact[nb_emb] + 1)
            )
            acc_emb += all_emb_impact[nb_emb]
            nb_emb += 1
    
    reducing_matrix = np.zeros((post_embed_dim, input_dim))
    for i, cols in enumerate(indices_trick):
        reducing_matrix[cols, i] = 1
    
    return scipy.sparse.csc_matrix(reducing_matrix)

def filter_weights(weights):
    """
    TabNet에서 regression, multitask 작업을 위해 weights format 맞춰줌
    """
    err_msg = """Please provide a a list or np.array of weights for"""
    err_msg += """regression, multitask or pretraining: """
    if isinstance(weights, int):
        if weights == 1:
            raise ValueError(err_msg + "1 given.")
    if isinstance(weights, dict):
        raise ValueError(err_msg + "Dict given.")
    return

def check_input(X):
    """
    pandas DataFrame 형식이면 이를 고치도록 에러메시지 출력.
    그리고 배열에 대해 scikit 형식의 배열 따르는 지 점검
    """
    if isinstance(X, (pd.DataFrame, pd.Series)):
        err_message = "Pandas DataFrame은 지원되지 않음 : X.values를 적용하여 학습시킬 것"
        raise TypeError(err_message)
    check_array(X)
    
def validate_eval_set(eval_set, eval_name, X_train, y_train):
    """
    eval_set이 (X_train, y_train)과 차원적으로 compatible 한지 확인
    """
    
    eval_name = eval_name or [f"val_{i}" for i in range(len(eval_set))]
    
    assert len(eval_set) == len(
        eval_name
        ), "eval_set and eval_name have not the same length"
    if len(eval_set) > 0:
        assert all(
            len(elem) == 2 for elem in eval_set
        ), "Each tuple of eval_set need to have two elements"
    for name, (X, y) in zip(eval_name, eval_set):
        check_input(X)
        msg = (
            f"Number of columns is different between X_{name} "
            + f"({X.shape[1]}) and X_train ({X_train.shape[1]}"
        )
        assert len(X.shape) == len(X_train.shape), msg

        msg = (
            f"Dimension mismatch between y_{name} "
            + f"{y.shape} and y_train {y_train.shape}"
        )
        assert len(y.shape) == len(y_train.shape), msg

        msg = (
            f"Number of columns is different between X_{name} "
            + f"({X.shape[1]}) and X_train ({X_train.sahpe[1]})"
        )
        assert X.shape[1] == X_train.shape[1], msg

        if len(y_train.shape) == 2:
            msg = (
                f"Number of columns is different between y_{name}"
                + f"({y.shape[1]}) and y_train ({y_train.shape[1]})"
            )
            assert y.shape[1] == y_train.shape[1], msg
        msg = (
            f"You need the same number of rows between X_{name} "
            + f"({X.shape[0]}) and y_{name} ({y.shape[0]})"
        )
        assert X.shape[0] == y.shape[0], msg

    return eval_name, eval_set

def define_device(device_name):
    """
    pytorch 학습을 위한 device 지정
    나는 맥 환경이므로 mps 추가
    Parameters
    ----------
    device_name : str
        Either "auto", "cpu" or "cuda" or "mps
    Returns
    -------
    str
        Either "cpu" or "cuda" or "mps
    """
    if device_name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    elif device_name == "cuda" and not torch.cuda.is_available():
        return "cpu"
    elif device_name == "mps" and not torch.backends.mps.is_available():
        return "cpu"
    else:
        return device_name

class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)

def check_warm_start(warm_start, from_unsupervised):
    if warm_start and from_unsupervised is not None:
        warn_msg = "warm_start=True and from_unsupervised != None: "
        warn_msg += "warm_start will be ignore, training will start from unsupervised weights"
        warnings.warn(warn_msg)
    return