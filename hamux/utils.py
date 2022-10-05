import pickle
from pathlib import Path
from typing import Union, Any
from copy import deepcopy
import flax.traverse_util as ftu
from flax.core import freeze
from loguru import logger
import types

suffixes = ['.pckl', ".pickle", ".pkl"]

def pytree_save(data: Any, path: Union[str, Path], overwrite: bool = False):
    path = Path(path)
    if path.suffix not in suffixes:
        path = path.with_suffix(suffixes[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if overwrite:
            path.unlink()
        else:
            raise RuntimeError(f'File {path} already exists.')
    with open(path, 'wb') as file:
        pickle.dump(data, file)


def pytree_load(path: Union[str, Path]) -> Any:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'Not a file: {path}')
    if path.suffix not in suffixes:
        raise ValueError(f'Not a pickle file: {path}')
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def to_pickleable(x):
    if callable(x):
        return x.__name__
    return x


def align_with_state_dict(list_curr, list_new):
    """Load the saved list into the current list.
    
    Important for loading checkpoints because our HAM is composed of a list of layers and list of synapses
    """
    list_curr = deepcopy(list_curr)
    frozen_list_new = [ftu.flatten_dict(freeze(mod)) for mod in list_new]
    for myobj, newobj in zip(list_curr, frozen_list_new):
        for ktup, value in newobj.items():
            item = myobj
            for k in ktup[:-1]:
                try:
                    item = getattr(item, k)
                except AttributeError as e:
                    logger.debug(f"Intermediate access key `{k}` in state dict but not in model. Skipping")
                
                    
            if "shape" == ktup[-1]:
                # assign to a tuple since saving turns tuples into lists
                value = tuple(value)
            try: 
                oldval = getattr(item, ktup[-1])
                if isinstance(value, str) and isinstance(oldval, types.FunctionType):
                    logger.debug(f"New value is a string, but old value is a callable. Setting {ktup[-1]} in new object to be the callable")
                    value = oldval
                elif isinstance(value, list) and isinstance(oldval, tuple):
                    logger.debug(f"New value is a list, but old value is a tuple. Setting {ktup[-1]} in new object to be a tuple")
                    value = tuple(value)

                setattr(item, ktup[-1], value)
            except AttributeError as e:
                logger.debug(f"Leaf key `{ktup[-1]}` in state dict but not in model. Skipping")
    return list_curr