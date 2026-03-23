from distutils.command.config import config
import json
import os
import random
import re

from torch.utils.data import Dataset
import torch
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

import os
from torchvision.transforms.functional import hflip, resize

import math
import random
from random import random as rand


def pre_caption(caption, max_words):
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        str(caption).lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])

    return caption


def _load_annotations_file(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        content = f.read().strip()

    if not content:
        return []

    # 1) Standard JSON (list or single object)
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = None

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    if data is not None:
        raise ValueError(f"Unsupported annotation JSON type in {path}: {type(data)}")

    # 2) JSONL / NDJSON (one JSON object per line)
    annotations = []
    for line_no, line in enumerate(content.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed parsing annotation line {line_no} in {path}: {e}") from e
        if not isinstance(item, dict):
            raise ValueError(f"Line {line_no} in {path} is not a JSON object.")
        annotations.append(item)
    return annotations


class DGM4_Dataset(Dataset):
    def __init__(self, config, ann_file, transform, max_words=30, is_train=True): 
        
        self.root_dir = config.get('data_root', '../../datasets')
        self.ann = []
        for f in ann_file:
            self.ann += _load_annotations_file(f)
        source_filter_key = 'train_sources' if is_train else 'val_sources'
        source_filter = config.get(source_filter_key, [])
        if source_filter:
            source_filter = set([self._normalize_source_name(x) for x in source_filter])
            self.ann = [ann for ann in self.ann if self._infer_source(ann) in source_filter]
        if 'dataset_division' in config:
            self.ann = self.ann[:int(len(self.ann)/config['dataset_division'])]

        self.transform = transform
        self.max_words = max_words
        self.image_res = config['image_res']

        self.is_train = is_train

    def _normalize_source_name(self, source_name):
        source_name = str(source_name).lower().strip().replace('-', '_').replace(' ', '_')
        alias = {
            'gardian': 'guardian',
            'the_guardian': 'guardian',
            'usatoday': 'usa_today',
            'usa-today': 'usa_today',
            'washingtonpost': 'washington_post',
            'washington-post': 'washington_post',
        }
        return alias.get(source_name, source_name)

    def _infer_source(self, ann):
        source_keys = ['source', 'news_source', 'media_source', 'website', 'domain', 'outlet', 'publisher']
        for key in source_keys:
            if key in ann and ann[key]:
                return self._normalize_source_name(ann[key])

        image_path = str(ann.get('image', '')).lower().replace('\\', '/')
        if 'gardian' in image_path or 'guardian' in image_path:
            return 'guardian'
        if 'usa_today' in image_path or 'usatoday' in image_path:
            return 'usa_today'
        if 'washington_post' in image_path or 'washingtonpost' in image_path:
            return 'washington_post'
        if '/bbc/' in image_path or image_path.startswith('bbc/'):
            return 'bbc'
        return None

    def _resolve_image_path(self, img_dir):
        if os.path.isabs(img_dir):
            return img_dir

        candidates = [
            os.path.join(self.root_dir, img_dir),
            os.path.join(self.root_dir, os.path.basename(img_dir)),
        ]
        if img_dir.startswith('DGM4/'):
            candidates.append(os.path.join(self.root_dir, img_dir[len('DGM4/'):]))

        for p in candidates:
            if os.path.exists(p):
                return p
        return candidates[0]
        
    def __len__(self):
        return len(self.ann)

    def get_bbox(self, bbox):
        xmin, ymin, xmax, ymax = bbox
        w = xmax - xmin
        h = ymax - ymin
        return int(xmin), int(ymin), int(w), int(h)    

    def __getitem__(self, index):    
        
        ann = self.ann[index]
        img_dir = ann['image']    
        image_dir_all = self._resolve_image_path(img_dir)

        try:
            image = Image.open(image_dir_all).convert('RGB')   
        except Warning:
            raise ValueError("### Warning: fakenews_dataset Image.open")   
                         
        W, H = image.size
        has_bbox = False
        try:
            x, y, w, h = self.get_bbox(ann['fake_image_box'])
            has_bbox = True
        except:
            fake_image_box = torch.tensor([0, 0, 0, 0], dtype=torch.float)

        do_hflip = False
        if self.is_train:
            if rand() < 0.5:
                # flipped applied
                image = hflip(image)
                do_hflip = True

            image = resize(image, [self.image_res, self.image_res], interpolation=Image.BICUBIC)
        image = self.transform(image)
            
        if has_bbox:
            # flipped applied
            if do_hflip:  
                x = (W - x) - w  # W is w0

            # resize applied
            x = self.image_res / W * x
            w = self.image_res / W * w
            y = self.image_res / H * y
            h = self.image_res / H * h

            center_x = x + 1 / 2 * w
            center_y = y + 1 / 2 * h

            fake_image_box = torch.tensor([center_x / self.image_res, 
                        center_y / self.image_res,
                        w / self.image_res, 
                        h / self.image_res],
                        dtype=torch.float)

        label = ann['fake_cls']
        caption = pre_caption(ann['text'], self.max_words)
        fake_text_pos = ann['fake_text_pos']

        fake_text_pos_list = torch.zeros(self.max_words)

        for i in fake_text_pos:
            if i<self.max_words:
                fake_text_pos_list[i]=1
        
                
        return image, label, caption, fake_image_box, fake_text_pos_list, W, H
