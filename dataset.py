import os
import glob
import numpy as np
import re
import webdataset as wds

import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms

from utils import instantiate_from_config


def dict_collation_fn(samples, combine_tensors=True, combine_scalars=True):
    """Take a list  of samples (as dictionary) and create a batch, preserving the keys.
    If `tensors` is True, `ndarray` objects are combined into
    tensor batches.
    :param dict samples: list of samples
    :param bool tensors: whether to turn lists of ndarrays into a single ndarray
    :returns: single sample consisting of a batch
    :rtype: dict
    """
    keys = set.intersection(*[set(sample.keys()) for sample in samples])
    batched = {key: [] for key in keys}

    for s in samples:
        [batched[key].append(s[key]) for key in batched]

    result = {}
    for key in batched:
        if isinstance(batched[key][0], (int, float)):
            if combine_scalars:
                result[key] = np.array(list(batched[key]))
        elif isinstance(batched[key][0], torch.Tensor):
            if combine_tensors:
                result[key] = torch.stack(list(batched[key]))
        elif isinstance(batched[key][0], np.ndarray):
            if combine_tensors:
                result[key] = np.array(list(batched[key]))
        else:
            result[key] = list(batched[key])
    return result

def preprocess(data, target_size):
    try:
        data['style'] = data['jpg']        
        data['original_size'] = torch.tensor(data['jpg'].size)

        # short edge resize and center crop
        data['jpg'] = transforms.functional.resize(data['jpg'], target_size)
        # data['jpg'] = transforms.functional.center_crop(data['jpg'], target_size)
        
        delta_h = data['jpg'].size[1] - target_size
        delta_w = data['jpg'].size[0] - target_size

        top = delta_h // 2
        left = delta_w // 2
        data['jpg'] = transforms.functional.crop(data['jpg'], top, left, target_size, target_size)
        data['crop_coords_top_left'] = torch.tensor([top, left])
        data['target_size'] = torch.tensor([target_size, target_size])


        if 'txt_blip' in data:
            caption = data['txt_blip'].decode('utf-8')
            # preprocess for text
            patterns = [
                r'^((a )?(\w+\s+)*(?:painting|drawing|mural|artwork|view|portrait|photo|photograph|watercolor|illustration|picture|digital art) of )+',
                r'in (a )?(\w+\s+)*(?:painting|drawing|mural|artwork|view|portrait|photo|photograph|watercolor|illustration|picture|digital art)\s?',
                r'^(?:painting|photograph|digital art|black and white photograph) - ',
            ]

            for pattern in patterns:
                caption = re.sub(pattern, "", caption)
            data['txt'] = caption

    except:
        raise Exception(f'No jpg in data: {data}')
    
    return data

def crop_and_resize(data, target_size):
    try:
        # print(data.keys())
        # print(data['jpg'].shape)
        # print(data['style'].shape)
        # print(data['txt'])
        delta_h = data['jpg'].shape[1] - target_size[0]
        delta_w = data['jpg'].shape[2] - target_size[1]

        top = delta_h // 2
        left = delta_w // 2
        data['jpg'] = transforms.functional.crop(data['jpg'], top, left, target_size[0], target_size[1])
        data['crop_coords_top_left'] = torch.tensor([top, left])
        data['target_size'] = torch.tensor(target_size)

    except:
        raise Exception(f'No style in data: {data}')


def filter_keys(x):
    try:
        return ("jpg" in x) and ("txt" in x)
    except Exception:
        return False
    
def filter_size(x):
    try:
        min_size = 320
        valid = True
        if x['jpg'].size[0] < min_size or x['jpg'].size[1] < min_size:
            valid = False
        aspect = x['jpg'].size[0] / x['jpg'].size[1]
        if aspect < 0.5 or aspect > 2:
            valid = False

        return valid

    except Exception:
        return False

def make_style_image_dataloader(tar_base, batch_size, num_workers=4, dataset_config=None, multinode=True, **kwargs):
    if "image_transforms_config" in dataset_config:
        image_transforms = [instantiate_from_config(tt) for tt in dataset_config.image_transforms_config]
    else:
        image_transforms = []
    image_transforms.extend([torchvision.transforms.ToTensor(),
                                torchvision.transforms.Lambda(lambda x: x * 2. - 1.)])
    image_transforms = torchvision.transforms.Compose(image_transforms)
    print(image_transforms)

    if "style_transforms_config" in dataset_config:
        style_transforms = [instantiate_from_config(tt) for tt in dataset_config.style_transforms_config]
    else:
        style_transforms = []
    # CLIP norm
    style_transforms.extend([torchvision.transforms.ToTensor(),
                             torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])
    style_transforms = torchvision.transforms.Compose(style_transforms)
    print(style_transforms)

    transform_dict = {
        'jpg': image_transforms,
        'style': style_transforms
    }

    if 'postprocess' in dataset_config:
        postprocess = instantiate_from_config(dataset_config['postprocess'])
    else:
        postprocess = None

    shuffle = dataset_config.get('shuffle', 0)
    shardshuffle = shuffle > 0

    nodesplitter = wds.shardlists.split_by_node if multinode else wds.shardlists.single_node_only
    
    tars = []
    for tar in tar_base:
        tars.extend(sorted(glob.glob(os.path.join(tar, "*.tar"))))
    dset = wds.WebDataset(
            tars,
            nodesplitter=nodesplitter,
            shardshuffle=shardshuffle,
            handler=wds.warn_and_continue).repeat(10000).shuffle(shuffle)
    # avoid "repeat()", which causes unknown error
    print(f'Loading webdataset with {len(dset.pipeline[0].urls)} shards.')

    PREPROCESS = lambda x: preprocess(x, target_size=dataset_config['target_size'])
    # crop and resize with target size
    # CROP_AND_RESIZE = lambda x: crop_and_resize(x, target_size=dataset_config['target_size'])

    dset = (dset
            .select(filter_keys)
            .decode('pil', handler=wds.warn_and_continue)
            # .select(filter_size)
            .map(PREPROCESS)
            .map_dict(**transform_dict, handler=wds.warn_and_continue)
            # .map(CROP_AND_RESIZE)
            )
    if postprocess is not None:
        dset = dset.map(postprocess)
    dset = (dset
            .batched(batch_size, partial=False,
                collation_fn=dict_collation_fn)
            )

    loader = wds.WebLoader(dset, batch_size=None, shuffle=False,
                            num_workers=num_workers)

    return loader

