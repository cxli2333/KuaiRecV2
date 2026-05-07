import os
import re
import json
import gzip
import html
import pickle
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
from logging import getLogger
from collections import defaultdict
from typing import Union, List, Optional, Dict, Any, Tuple

from sentence_transformers import SentenceTransformer

def download_file(url: str, path: str) -> None:
    """
    Downloads a file from the given URL and saves it to the specified path.

    Args:
        url (str): The URL of the file to download.
        path (str): The path where the downloaded file will be saved.
    """
    logger = getLogger()
    response = requests.get(url)
    if response.status_code == 200:
        with open(path, 'wb') as f:
            f.write(response.content)
        logger.info(f"Downloaded {os.path.basename(path)}")
    else:
        logger.error(f"Failed to download {os.path.basename(path)}")
        
def list_to_str(l: Union[list, str], remove_blank=False) -> str:
    """
    Converts a list or a string to a string representation.

    Args:
        l (Union[list, str]): The input list or string.

    Returns:
        str: The string representation of the input.
    """
    ret = ''
    if isinstance(l, list):
        ret = ', '.join(map(str, l))
    else:
        ret = l
    if remove_blank:
        ret = ret.replace(' ', '')
    return ret
        
def clean_text(raw_text: str) -> str:
    """
    Cleans the raw text by removing HTML tags, special characters, and extra spaces.

    Args:
        raw_text (str): The raw text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    text = list_to_str(raw_text)
    text = html.unescape(text)
    text = text.strip()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[\n\t]', ' ', text)
    text = re.sub(r' +', ' ', text)
    text=re.sub(r'[^\x00-\x7F]', ' ', text)
    return text

class Amazon2014PreProcessor(object):
    """
    Amazon 2014 纯预处理器（基于原始物品ID，无稠密映射）：
        - 下载 reviews/meta（SNAP）
        - 解析为 user -> [item]（按时间升序），item 为原始 ASIN
        - 可选：生成 metadata.sentence.json（键为原始 ASIN）
        - SentenceTransformer 将 metadata.sentence.json 编码为 item(ASIN) -> embedding 的 pkl
        - 导出 raw_seq.pkl（两列：user, seq，其中 seq 为原始 ASIN 列表）
        - 如缓存命中（raw_seq.pkl & item2emb_*.pkl 已存在），则直接读取并返回

    必要配置：
        - category: str
        - cache_dir: str
        - metadata: 'none' | 'raw' | 'sentence'（如需编码，必须为 'sentence'）

    可选配置：
        - ddp_accelerator: accelerate.Accelerator（仅用于 main_process_first()）
        - logger: logging.Logger
        - sent_emb_model: str (默认 'Alibaba-NLP/gte-large-en-v1.5')
        - sent_emb_batch_size: int (默认 256)
        - sent_emb_dim:  int（仅日志用）
        - normalize_embeddings: bool (默认 True)
        - min_seq_len_export: int (默认 1)  # 导出 raw_seq 时最小序列长度过滤
        - device: str ('cpu' | 'cuda' | ...，默认 'cpu')
    """

    AVAILABLE_CATEGORIES = [
        'Books', 'Electronics', 'Movies_and_TV', 'CDs_and_Vinyl',
        'Clothing_Shoes_and_Jewelry', 'Home_and_Kitchen', 'Kindle_Store',
        'Sports_and_Outdoors', 'Cell_Phones_and_Accessories',
        'Health_and_Personal_Care', 'Toys_and_Games', 'Video_Games',
        'Tools_and_Home_Improvement', 'Beauty', 'Apps_for_Android',
        'Office_Products', 'Pet_Supplies', 'Automotive',
        'Grocery_and_Gourmet_Food', 'Patio_Lawn_and_Garden', 'Baby',
        'Digital_Music', 'Musical_Instruments', 'Amazon_Instant_Video'
    ]

    def __init__(self, config: dict):
        self.config = config
        self.category = str(config['category'])
        assert self.category in self.AVAILABLE_CATEGORIES, \
            f'Category "{self.category}" not in {self.AVAILABLE_CATEGORIES}'

        # 目录：cache/{dataset_name}
        self.base_cache = os.path.join(config['cache_dir'], 'Amazon2014', self.category)
        self.raw_dir = os.path.join(self.base_cache, 'raw')
        self.proc_dir = os.path.join(self.base_cache, 'processed')
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.proc_dir, exist_ok=True)

        self.metadata_mode = str(config.get('metadata', 'none'))
        self.logger = config.get('logger', getLogger())
        self.accelerator = config.get('ddp_accelerator', None)

        # 输出文件名（原始key版本）
        self.seq_pkl_path = os.path.join(self.proc_dir, 'raw_seq.pkl')  # user, seq(原始 ASIN)
        self.id_mapping_json = os.path.join(self.proc_dir, 'id_mapping.json')  # 兼容保留
        self.meta_sentence_json = os.path.join(self.proc_dir, 'metadata.sentence.json')

        # 根据模型命名 Embedding PKL
        model_id = config.get('sent_emb_model', 'Alibaba-NLP/gte-large-en-v1.5')
        model_tag = os.path.basename(model_id).replace('/', '_')
        self.item2emb_pkl = os.path.join(self.proc_dir, f'item2emb_{model_tag}.pkl')

        # 中间状态（all_item_seqs 用**原始 ASIN**）
        self.all_item_seqs: Dict[Any, List[Any]] = {}

        # 兼容保留的映射结构（但本流程不再使用）
        self.id_mapping = {
            'user2id': {'[PAD]': 0},
            'item2id': {'[PAD]': 0},
            'id2user': ['[PAD]'],
            'id2item': ['[PAD]'],
        }

        self.item2meta: Optional[Dict[str, Any]] = None

    # ====================== 公共入口 ======================
    def run(self) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
        """
        运行全流程；若缓存齐全（raw_seq.pkl & item2emb_*.pkl），则直接加载并返回：
        Returns:
            df_seq:  DataFrame 两列 ['user','seq']，seq 为**原始 ASIN** 列表
            item2emb: dict(str -> np.ndarray(float32)) 的元数据向量（键为 ASIN）
        """
        # 命中缓存：直接读取
        if os.path.exists(self.seq_pkl_path) and os.path.exists(self.item2emb_pkl):
            df_seq = pd.read_pickle(self.seq_pkl_path)
            with open(self.item2emb_pkl, 'rb') as f:
                item2emb = pickle.load(f)
            self._log(f'[CACHE] Hit. Loaded raw_seq & item2emb from {self.proc_dir}')
            return df_seq, item2emb

        # 1) 下载原始 reviews / meta
        reviews_path, meta_path = self._download_raw_pair()

        # 2) 尝试载入 processed 的基础缓存（all_item_seqs / id_mapping）
        seq_json = os.path.join(self.proc_dir, 'all_item_seqs.json')
        if os.path.exists(seq_json) and os.path.exists(self.id_mapping_json):
            self._log('[CACHE] Found processed sequences & id mapping, loading...')
            with open(seq_json, 'r') as f:
                self.all_item_seqs = json.load(f)  # 原始 ASIN 序列
            with open(self.id_mapping_json, 'r') as f:
                self.id_mapping = json.load(f)     # 兼容保留
        else:
            # 2a) 解析 reviews → user->[ASIN]（按时间排序）
            reviews = self._load_reviews(reviews_path)
            item_seqs = self._get_item_seqs(reviews)   # 原始 ASIN 序列
            self.all_item_seqs = item_seqs

            # （不 remap）仅把空壳 id_mapping 落盘，以兼容下游其他组件可能读取该文件
            with open(seq_json, 'w') as f:
                json.dump(self.all_item_seqs, f)
            with open(self.id_mapping_json, 'w') as f:
                json.dump(self.id_mapping, f)
            self._log('[SAVE] Processed sequences & id mapping saved (no remap).')

        # 3) 导出 raw_seq（seq 为原始 ASIN）
        df_seq = self._export_raw_seq(min_len=self.config.get('min_seq_len_export', 1))

        # 4) 处理/加载 metadata.sentence.json
        if self.metadata_mode == 'sentence':
            if os.path.exists(self.meta_sentence_json):
                self._log('[CACHE] Found metadata.sentence.json, loading...')
                with open(self.meta_sentence_json, 'r') as f:
                    self.item2meta = json.load(f)
            else:
                self.item2meta = self._process_meta(meta_path)  # 内部会先 _load_metadata()
                with open(self.meta_sentence_json, 'w') as f:
                    json.dump(self.item2meta, f)
                self._log('[SAVE] metadata.sentence.json saved.')
        elif self.metadata_mode == 'raw':
            # 统一转 sentence 便于编码
            meta_raw = self._load_metadata(meta_path)  # 以原始 ASIN 集过滤
            self.item2meta = self._extract_meta_sentences(meta_raw)
            with open(self.meta_sentence_json, 'w') as f:
                json.dump(self.item2meta, f)
            self._log('[INFO] metadata was "raw"; converted to sentence for encoding.')

        # 5) SentenceTransformer 编码 → item2emb pkl
        item2emb = self._encode_sentence_embeddings()

        return df_seq, item2emb

    # ====================== 下载与解析 ======================
    def _download_raw_pair(self) -> Tuple[str, str]:
        def _mpf_ctx():
            if self.accelerator is None:
                class _Dummy(object):
                    def __enter__(self): return None
                    def __exit__(self, *args): return False
                return _Dummy()
            return self.accelerator.main_process_first()

        reviews_url = f'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_{self.category}_5.json.gz'
        meta_url    = f'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_{self.category}.json.gz'
        r_path = os.path.join(self.raw_dir, os.path.basename(reviews_url))
        m_path = os.path.join(self.raw_dir, os.path.basename(meta_url))
        with _mpf_ctx():
            if not os.path.exists(r_path):
                download_file(reviews_url, r_path)
            if not os.path.exists(m_path):
                download_file(meta_url, m_path)
        return r_path, m_path

    def _parse_gz(self, path: str):
        import ast
        with gzip.open(path, 'rt', encoding='utf-8') as g:
            for line in g:
                s = line.strip()
                if not s:
                    continue
                try:
                    yield json.loads(s)
                except Exception:
                    try:
                        yield ast.literal_eval(s)
                    except Exception:
                        continue

    def _load_reviews(self, path: str) -> List[Tuple[Any, Any, int]]:
        self._log(f'Loading reviews: {path}')
        out = []
        for inter in self._parse_gz(path):
            try:
                u = inter['reviewerID']
                i = inter['asin']
                t = int(inter['unixReviewTime'])
                out.append((u, i, t))
            except Exception:
                continue
        return out

    def _get_item_seqs(self, reviews: List[Tuple[Any, Any, int]]) -> Dict[Any, List[Any]]:
        from collections import defaultdict
        user2items: Dict[Any, List[Tuple[Any, int]]] = defaultdict(list)
        for u, i, t in reviews:
            user2items[u].append((i, t))
        for u, pairs in user2items.items():
            pairs.sort(key=lambda x: x[1])
            user2items[u] = [p[0] for p in pairs]  # 原始 ASIN 顺序序列
        return user2items

    # ====================== metadata → sentence ======================
    def _load_metadata(self, meta_path: str) -> Dict[str, Any]:
        """
        读取并按交互里出现过的原始 ASIN 集过滤 metadata。
        """
        data: Dict[str, Any] = {}
        keep: set[str] = set()
        for seq in self.all_item_seqs.values():
            keep.update(seq)

        self._log('Loading metadata (filtered by raw ASINs in interactions)...')
        for info in tqdm(self._parse_gz(meta_path)):
            asin = info.get('asin', None)
            if asin in keep:
                data[asin] = info
        return data

    def _sent_process(self, raw: Any) -> str:
        s = ""
        if isinstance(raw, float):
            s += str(raw)
        elif isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], list):
            parts = []
            for v1 in raw:
                for v in v1:
                    t = clean_text(v).strip()
                    if t:
                        if t.endswith('.'):
                            t = t[:-1]
                        parts.append(t)
            s = ', '.join(parts)
        elif isinstance(raw, list):
            parts = []
            for v in raw:
                t = clean_text(v).strip()
                if t:
                    if t.endswith('.'):
                        t = t[:-1]
                    parts.append(t)
            s = ', '.join(parts)
        else:
            s = clean_text(str(raw)).strip()
            if s.endswith('.'):
                s = s[:-1]
        return s

    def _extract_meta_sentences(self, meta: Dict[str, Any]) -> Dict[str, str]:
        out: Dict[str, str] = {}
        keys_order = [('title', 'Title'), ('price', 'Price'), ('brand', 'Brand'),
                      ('categories', 'Category'), ('feature', 'Features'), ('description', 'Description')]
        self._log('Converting metadata to sentence form...')
        for item, info in tqdm(meta.items()):
            parts = []
            ks = set(info.keys())
            for k, label in keys_order:
                if k in ks and info[k]:
                    val = self._sent_process(info[k]).strip()
                    if val:
                        if val.endswith('.'):
                            parts.append(f'{label}: {val}')
                        else:
                            parts.append(f'{label}: {val}.')
            out[item] = ' '.join(parts)
        return out

    def _process_meta(self, meta_path: str) -> Dict[str, str]:
        assert self.metadata_mode in ('raw', 'sentence'), \
            'metadata must be one of {"none","raw","sentence"} to process.'
        meta_raw = self._load_metadata(meta_path)  # 按原始 ASIN 过滤
        # 统一转 sentence（即使 metadata_mode == 'sentence'，也复用相同流程便于清洗）
        return self._extract_meta_sentences(meta_raw)

    # ====================== 导出序列（原始ID） ======================
    def _export_raw_seq(self, min_len: int = 1) -> pd.DataFrame:
        """导出 raw_seq.pkl：两列 ['user','seq']，seq 为**原始 ASIN**（过滤长度 < min_len 的用户）。"""
        rows = []
        for u, seq in self.all_item_seqs.items():
            seq_out = list(seq)  # 原始 ASIN
            if len(seq_out) >= int(min_len):
                rows.append({'user': u, 'seq': seq_out})

        df = pd.DataFrame(rows)
        df.to_pickle(self.seq_pkl_path)
        self._log(f"[SAVE] raw_seq (raw-key) saved to {self.seq_pkl_path} (users={len(df)})")
        return df

    # ====================== 编码 ======================
    def _encode_sentence_embeddings(self) -> Dict[str, np.ndarray]:
        """
        将 metadata.sentence.json 编码为 **ASIN -> embedding**，
        并保存到 item2emb_*.pkl；全流程通过 logger 打印进度。
        """
        if self.metadata_mode == 'none':
            raise ValueError('metadata="none": cannot encode embeddings. Set metadata="sentence".')

        # 命中缓存：直接载入
        if os.path.exists(self.item2emb_pkl):
            self._log(f'[CACHE] Found {self.item2emb_pkl}, loading...')
            with open(self.item2emb_pkl, 'rb') as f:
                return pickle.load(f)

        # 读取 sentence 版元数据
        if self.item2meta is None:
            if not os.path.exists(self.meta_sentence_json):
                raise FileNotFoundError('metadata.sentence.json not found for encoding.')
            with open(self.meta_sentence_json, 'r') as f:
                self.item2meta = json.load(f)

        # 交互中出现过的 ASIN ∩ 有有效 sentence 的 ASIN
        items_in_seq = set()
        for seq in self.all_item_seqs.values():
            items_in_seq.update(seq)

        candidates: List[str] = []
        sentences: List[str] = []
        for asin in items_in_seq:
            sent = self.item2meta.get(asin, None)
            if isinstance(sent, str) and sent.strip():
                candidates.append(asin)
                sentences.append(sent.strip())

        total = len(candidates)
        if total == 0:
            raise ValueError('No items with sentence metadata to encode (after intersect with interactions).')

        # 日志：准备阶段
        self._log(f'[ENCODE] items with sentences: {total} / {len(items_in_seq)}')
        model_id   = self.config.get('sent_emb_model', 'Alibaba-NLP/gte-large-en-v1.5')
        batch_size = int(self.config.get('sent_emb_batch_size', 256))
        normalize  = bool(self.config.get('normalize_embeddings', True))
        device     = self.config.get('device', 'cpu')
        self._log(f'[ENCODE] model={model_id} device={device} batch_size={batch_size} normalize={normalize}')

        # 编码
        model = SentenceTransformer(model_id, trust_remote_code=True, device=device)
        self._log('[ENCODE] Start sentence embedding…')
        embs = model.encode(
            sentences,
            convert_to_numpy=True,
            batch_size=batch_size,
            show_progress_bar=True,  # 自带tqdm进度条
            device=device,
            normalize_embeddings=normalize,
        ).astype(np.float32)
        emb_dim = int(embs.shape[1]) if embs.ndim == 2 else 0
        self._log(f'[ENCODE] Finished. emb_shape={embs.shape}, emb_dim={emb_dim}')

        # 组装字典：ASIN -> embedding
        item2emb: Dict[str, np.ndarray] = {asin: embs[i] for i, asin in enumerate(candidates)}

        # 保存
        with open(self.item2emb_pkl, 'wb') as f:
            pickle.dump(item2emb, f, protocol=pickle.HIGHEST_PROTOCOL)
        self._log(f'[SAVE] item2emb pkl saved to {self.item2emb_pkl}')

        return item2emb

    # ====================== 工具 ======================
    def _log(self, msg: str):
        try:
            self.logger.info(msg)
        except Exception:
            print(msg)


# class Amazon2014PreProcessor(object):
#     """
#     Amazon 2014 纯预处理器：
#       - 下载 reviews/meta（SNAP）
#       - 解析为 user -> [item]（按时间升序）
#       - 稠密映射（[PAD]=0，从1开始）
#       - 可选：生成 metadata.sentence.json
#       - 用 SentenceTransformer 将 metadata.sentence.json 编码为 item_id -> embedding 的 pkl
#       - 导出 raw_seq.pkl（两列：user, seq，其中 seq 为稠密 item_id）
#       - 如缓存命中（raw_seq.pkl & item2emb_*.pkl 已存在），则直接读取并返回

#     必要配置：
#       - category: str
#       - cache_dir: str
#       - metadata: 'none' | 'raw' | 'sentence'（如需编码，必须为 'sentence'）

#     可选配置：
#       - ddp_accelerator: accelerate.Accelerator（只用于 main_process_first()）
#       - logger: logging.Logger
#       - sent_emb_model: str (默认 'Alibaba-NLP/gte-large-en-v1.5')
#       - sent_emb_batch_size: int (默认 256)
#       - sent_emb_dim:  int（仅做日志用，可不填）
#       - normalize_embeddings: bool (默认 True)
#       - min_seq_len_export: int (默认 1)  # 导出 raw_seq.pkl 时最小序列长度过滤
#     """

#     AVAILABLE_CATEGORIES = [
#         'Books', 'Electronics', 'Movies_and_TV', 'CDs_and_Vinyl',
#         'Clothing_Shoes_and_Jewelry', 'Home_and_Kitchen', 'Kindle_Store',
#         'Sports_and_Outdoors', 'Cell_Phones_and_Accessories',
#         'Health_and_Personal_Care', 'Toys_and_Games', 'Video_Games',
#         'Tools_and_Home_Improvement', 'Beauty', 'Apps_for_Android',
#         'Office_Products', 'Pet_Supplies', 'Automotive',
#         'Grocery_and_Gourmet_Food', 'Patio_Lawn_and_Garden', 'Baby',
#         'Digital_Music', 'Musical_Instruments', 'Amazon_Instant_Video'
#     ]

#     def __init__(self, config: dict):
#         self.config = config
#         self.category = str(config['category'])
#         assert self.category in self.AVAILABLE_CATEGORIES, \
#             f'Category "{self.category}" not in {self.AVAILABLE_CATEGORIES}'

#         # 目录：cache/{dataset_name}
#         self.base_cache = os.path.join(config['cache_dir'], 'Amazon2014', self.category)
#         self.raw_dir = os.path.join(self.base_cache, 'raw')
#         self.proc_dir = os.path.join(self.base_cache, 'processed')
#         os.makedirs(self.raw_dir, exist_ok=True)
#         os.makedirs(self.proc_dir, exist_ok=True)

#         self.metadata_mode = str(config.get('metadata', 'none'))
#         self.logger = config.get('logger', getLogger())
#         self.accelerator = config.get('ddp_accelerator', None)

#         # 输出文件名（固定）
#         self.seq_pkl_path = os.path.join(self.proc_dir, 'raw_seq.pkl')  # user, seq(item_id)
#         self.id_mapping_json = os.path.join(self.proc_dir, 'id_mapping.json')
#         self.meta_sentence_json = os.path.join(self.proc_dir, 'metadata.sentence.json')

#         # 根据模型命名 Embedding PKL
#         model_id = config.get('sent_emb_model', 'Alibaba-NLP/gte-large-en-v1.5')
#         model_tag = os.path.basename(model_id).replace('/', '_')
#         self.item2emb_pkl = os.path.join(self.proc_dir, f'item2emb_{model_tag}.pkl')

#         # 中间状态
#         self.all_item_seqs: Dict[Any, List[Any]] = {}
#         self.id_mapping = {
#             'user2id': {'[PAD]': 0},
#             'item2id': {'[PAD]': 0},
#             'id2user': ['[PAD]'],
#             'id2item': ['[PAD]'],
#         }
#         self.item2meta: Optional[Dict[str, Any]] = None

#     # ====================== 公共入口 ======================
#     def run(self) -> Tuple[pd.DataFrame, Dict[int, np.ndarray]]:
#         """
#         运行全流程；若缓存齐全（raw_seq.pkl & item2emb_*.pkl），则直接加载并返回：
#         Returns:
#             df_seq:  DataFrame 两列 ['user','seq']，seq 为稠密 item_id 列表
#             item2emb: dict(int -> np.ndarray(float32)) 的元数据向量
#         """
#         # 命中缓存：直接读取
#         if os.path.exists(self.seq_pkl_path) and os.path.exists(self.item2emb_pkl):
#             df_seq = pd.read_pickle(self.seq_pkl_path)
#             with open(self.item2emb_pkl, 'rb') as f:
#                 item2emb = pickle.load(f)
#             self._log(f'[CACHE] Hit. Loaded raw_seq.pkl & item2emb pkl from {self.proc_dir}')
#             return df_seq, item2emb

#         # 1) 下载原始 reviews / meta
#         reviews_path, meta_path = self._download_raw_pair()

#         # 2) 尝试载入 processed 的基础缓存（all_item_seqs / id_mapping）
#         seq_json = os.path.join(self.proc_dir, 'all_item_seqs.json')
#         if os.path.exists(seq_json) and os.path.exists(self.id_mapping_json):
#             self._log('[CACHE] Found processed sequences & id mapping, loading...')
#             with open(seq_json, 'r') as f:
#                 self.all_item_seqs = json.load(f)
#             with open(self.id_mapping_json, 'r') as f:
#                 self.id_mapping = json.load(f)
#         else:
#             # 2a) 解析 reviews → user->[item]（按时间排序）
#             reviews = self._load_reviews(reviews_path)
#             item_seqs = self._get_item_seqs(reviews)
#             # 2b) 稠密映射（不改变 all_item_seqs 的 token 形态）
#             #self._remap_ids(item_seqs)
#             # 2c) 落盘基础 processed
#             with open(seq_json, 'w') as f:
#                 json.dump(self.all_item_seqs, f)
#             with open(self.id_mapping_json, 'w') as f:
#                 json.dump(self.id_mapping, f)
#             self._log('[SAVE] Processed sequences & id mapping saved.')

#         # 3) 导出 raw_seq.pkl（seq 为稠密 item_id）
#         df_seq = self._export_raw_seq_as_item_ids(min_len=self.config.get('min_seq_len_export', 1))
#         # 4) 处理/加载 metadata.sentence.json
#         if self.metadata_mode == 'sentence':
#             if os.path.exists(self.meta_sentence_json):
#                 self._log('[CACHE] Found metadata.sentence.json, loading...')
#                 with open(self.meta_sentence_json, 'r') as f:
#                     self.item2meta = json.load(f)
#             else:
#                 self.item2meta = self._process_meta(meta_path)
#                 with open(self.meta_sentence_json, 'w') as f:
#                     json.dump(self.item2meta, f)
#                 self._log('[SAVE] metadata.sentence.json saved.')
#         elif self.metadata_mode == 'raw':
#             # 仅为了编码方便，强制转 sentence
#             self.item2meta = self._extract_meta_sentences(
#                 self._load_metadata(meta_path, self.id_mapping['item2id'])
#             )
#             with open(self.meta_sentence_json, 'w') as f:
#                 json.dump(self.item2meta, f)
#             self._log('[INFO] metadata was "raw"; converted to sentence for encoding.')

#         # 5) SentenceTransformer 编码 → item2emb pkl
#         item2emb = self._encode_sentence_embeddings()

#         return df_seq, item2emb

#     # ====================== 下载与解析 ======================
#     def _download_raw_pair(self) -> Tuple[str, str]:
#         def _mpf_ctx():
#             if self.accelerator is None:
#                 class _Dummy(object):
#                     def __enter__(self): return None
#                     def __exit__(self, *args): return False
#                 return _Dummy()
#             return self.accelerator.main_process_first()

#         reviews_url = f'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_{self.category}_5.json.gz'
#         meta_url    = f'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_{self.category}.json.gz'
#         r_path = os.path.join(self.raw_dir, os.path.basename(reviews_url))
#         m_path = os.path.join(self.raw_dir, os.path.basename(meta_url))
#         with _mpf_ctx():
#             if not os.path.exists(r_path):
#                 download_file(reviews_url, r_path)
#             if not os.path.exists(m_path):
#                 download_file(meta_url, m_path)
#         return r_path, m_path

#     def _parse_gz(self, path: str):
#         import ast
#         with gzip.open(path, 'rt', encoding='utf-8') as g:
#             for line in g:
#                 s = line.strip()
#                 if not s:
#                     continue
#                 try:
#                     yield json.loads(s)
#                 except Exception:
#                     try:
#                         yield ast.literal_eval(s)
#                     except Exception:
#                         continue

#     def _load_reviews(self, path: str) -> List[Tuple[Any, Any, int]]:
#         self._log(f'Loading reviews: {path}')
#         out = []
#         for inter in self._parse_gz(path):
#             try:
#                 u = inter['reviewerID']
#                 i = inter['asin']
#                 t = int(inter['unixReviewTime'])
#                 out.append((u, i, t))
#             except Exception:
#                 continue
#         return out

#     def _get_item_seqs(self, reviews: List[Tuple[Any, Any, int]]) -> Dict[Any, List[Any]]:
#         user2items: Dict[Any, List[Tuple[Any, int]]] = {}
#         from collections import defaultdict
#         user2items = defaultdict(list)
#         for u, i, t in reviews:
#             user2items[u].append((i, t))
#         for u, pairs in user2items.items():
#             pairs.sort(key=lambda x: x[1])
#             user2items[u] = [p[0] for p in pairs]
#         return user2items

#     # def _remap_ids(self, item_seqs: Dict[Any, List[Any]]) -> None:
#     #     self._log('Remapping user/item to dense ids...')
#     #     # user
#     #     for u in item_seqs.keys():
#     #         if u not in self.id_mapping['user2id']:
#     #             self.id_mapping['user2id'][u] = len(self.id_mapping['id2user'])
#     #             self.id_mapping['id2user'].append(u)
#     #     # item
#     #     for _, seq in item_seqs.items():
#     #         for it in seq:
#     #             if it not in self.id_mapping['item2id']:
#     #                 self.id_mapping['item2id'][it] = len(self.id_mapping['id2item'])
#     #                 self.id_mapping['id2item'].append(it)
#     #     self.all_item_seqs = item_seqs

#     # ====================== metadata → sentence ======================
#     def _load_metadata(self, meta_path: str, item2id: Dict[Any, int]) -> Dict[str, Any]:
#         data = {}
#         keep = set(item2id.keys())
#         self._log('Loading metadata (filtered by items in interactions)...')
#         for info in tqdm(self._parse_gz(meta_path)):
#             asin = info.get('asin', None)
#             if asin in keep:
#                 data[asin] = info
#         return data

#     def _sent_process(self, raw: Any) -> str:
#         s = ""
#         if isinstance(raw, float):
#             s += str(raw)
#         elif isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], list):
#             parts = []
#             for v1 in raw:
#                 for v in v1:
#                     t = clean_text(v).strip()
#                     if t:
#                         if t.endswith('.'):
#                             t = t[:-1]
#                         parts.append(t)
#             s = ', '.join(parts)
#         elif isinstance(raw, list):
#             parts = []
#             for v in raw:
#                 t = clean_text(v).strip()
#                 if t:
#                     if t.endswith('.'):
#                         t = t[:-1]
#                     parts.append(t)
#             s = ', '.join(parts)
#         else:
#             s = clean_text(str(raw)).strip()
#             if s.endswith('.'):
#                 s = s[:-1]
#         return s

#     def _extract_meta_sentences(self, meta: Dict[str, Any]) -> Dict[str, str]:
#         out: Dict[str, str] = {}
#         keys_order = [('title', 'Title'), ('price', 'Price'), ('brand', 'Brand'),
#                       ('categories', 'Category'), ('feature', 'Features'), ('description', 'Description')]
#         self._log('Converting metadata to sentence form...')
#         for item, info in tqdm(meta.items()):
#             parts = []
#             ks = set(info.keys())
#             for k, label in keys_order:
#                 if k in ks and info[k]:
#                     val = self._sent_process(info[k]).strip()
#                     if val:
#                         if val.endswith('.'):
#                             parts.append(f'{label}: {val}')
#                         else:
#                             parts.append(f'{label}: {val}.')
#             out[item] = ' '.join(parts)
#         return out

#     def _process_meta(self, meta_path: str) -> Dict[str, str]:
#         assert self.metadata_mode in ('raw', 'sentence'), \
#             'metadata must be one of {"none","raw","sentence"} to process.'
#         meta_raw = self._load_metadata(meta_path, self.id_mapping['item2id'])
#         if self.metadata_mode == 'raw':
#             # 为了编码，仍然转 sentence
#             return self._extract_meta_sentences(meta_raw)
#         return self._extract_meta_sentences(meta_raw)

#     # ====================== 导出序列 & 编码 ======================
#     def _export_raw_seq_as_item_ids(self, min_len: int = 1) -> pd.DataFrame:
#         """导出 raw_seq.pkl：两列 ['user','seq']，seq 为稠密 item_id（过滤长度 < min_len 的用户）。"""
#         item2id = self.id_mapping['item2id']
#         rows = []
#         for u, seq in self.all_item_seqs.items():
#             ids = [item2id[it] for it in seq if it in item2id]
#             if len(ids) >= int(min_len):
#                 rows.append({'user': u, 'seq': ids})
#         df = pd.DataFrame(rows)
#         df.to_pickle(self.seq_pkl_path)
#         self._log(f'[SAVE] raw_seq.pkl saved to {self.seq_pkl_path} (users={len(df)})')
#         return df
    
#     # ====================== 导出序列 & 编码 ======================
#     def _export_raw_seq(self, min_len: int = 1) -> pd.DataFrame:
#         """导出 raw_seq.pkl：两列 ['user','seq']，seq 为稠密 item_id（过滤长度 < min_len 的用户）。"""
        
#         item2id = self.id_mapping['item2id']
#         rows = []
#         for u, seq in self.all_item_seqs.items():
#             seq_out = list(seq)
#             if len(seq_out) >= int(min_len):
#                 rows.append({'user': u, 'seq': seq_out})

#         df = pd.DataFrame(rows)
#         df.to_pickle(self.seq_pkl_path)

#         self._log(f"[SAVE] raw_seq.pkl saved to {self.seq_pkl_path} (users={len(df)})")
#         return df


#     def _encode_sentence_embeddings(self) -> Dict[str, np.ndarray]:
#         """
#         将 metadata.sentence.json 编码为 **item_token(ASIN) -> embedding**，
#         并保存到 item2emb_*.pkl；全流程通过 logger 打印进度。
#         """
#         if self.metadata_mode == 'none':
#             raise ValueError('metadata="none": cannot encode embeddings. Set metadata="sentence".')

#         # 命中缓存：直接载入
#         if os.path.exists(self.item2emb_pkl):
#             self._log(f'[CACHE] Found {self.item2emb_pkl}, loading...')
#             with open(self.item2emb_pkl, 'rb') as f:
#                 return pickle.load(f)

#         # 读取 sentence 版元数据
#         if self.item2meta is None:
#             if not os.path.exists(self.meta_sentence_json):
#                 raise FileNotFoundError('metadata.sentence.json not found for encoding.')
#             with open(self.meta_sentence_json, 'r') as f:
#                 self.item2meta = json.load(f)

#         id2item: List[str] = self.id_mapping['id2item']  # 0 为 [PAD]
#         n_items = len(id2item) - 1

#         # 组装编码列表（按 item_id 升序），键用 ASIN
#         items: List[str] = []
#         sentences: List[str] = []
#         for iid in range(1, n_items + 1):
#             asin = id2item[iid]
#             sent = self.item2meta.get(asin, None)
#             if isinstance(sent, str) and sent.strip():
#                 items.append(asin)
#                 sentences.append(sent.strip())

#         total = len(items)
#         if total == 0:
#             raise ValueError('No items with sentence metadata to encode.')

#         # 日志：准备阶段
#         self._log(f'[ENCODE] items with sentences: {total} / {n_items}')
#         model_id   = self.config.get('sent_emb_model', 'Alibaba-NLP/gte-large-en-v1.5')
#         batch_size = int(self.config.get('sent_emb_batch_size', 256))
#         normalize  = bool(self.config.get('normalize_embeddings', True))
#         device = self.config.get('device', 'cpu')
#         self._log(f'[ENCODE] model={model_id} device={device} batch_size={batch_size} normalize={normalize}')

#         # 编码
#         model = SentenceTransformer(model_id, trust_remote_code=True, device=device)
#         self._log('[ENCODE] Start sentence embedding…')
#         embs = model.encode(
#             sentences,
#             convert_to_numpy=True,
#             batch_size=batch_size,
#             show_progress_bar=True,  # 自带tqdm进度条
#             device=device,
#             normalize_embeddings=normalize,
#         ).astype(np.float32)
#         emb_dim = int(embs.shape[1]) if embs.ndim == 2 else 0
#         self._log(f'[ENCODE] Finished. emb_shape={embs.shape}, emb_dim={emb_dim}')

#         # 组装字典：ASIN -> embedding
#         item2emb: Dict[str, np.ndarray] = {asin: embs[i] for i, asin in enumerate(items)}

#         # 保存
#         with open(self.item2emb_pkl, 'wb') as f:
#             pickle.dump(item2emb, f, protocol=pickle.HIGHEST_PROTOCOL)
#         self._log(f'[SAVE] item2emb pkl saved to {self.item2emb_pkl}')

#         return item2emb


#     # ====================== 工具 ======================
#     def _log(self, msg: str):
#         try:
#             self.logger.info(msg)
#         except Exception:
#             print(msg)

if __name__ == "__main__":
    
    import argparse
    import logging

    parser = argparse.ArgumentParser(description="Run Amazon2014PreProcessor standalone.")
    parser.add_argument("--category", type=str, default="Sports_and_Outdoors", help="Amazon2014 category")
    parser.add_argument("--cache-dir", type=str, default="../cache", help="Cache root directory")
    parser.add_argument("--metadata", type=str, default="sentence", choices=["none", "raw", "sentence"])
    parser.add_argument("--sent-emb-model", type=str, default="sentence-transformers/sentence-t5-base") # Alibaba-NLP/gte-large-en-v1.5
    parser.add_argument("--sent-emb-batch-size", type=int, default=256)
    parser.add_argument("--normalize-embeddings", action="store_false", default=False)
    parser.add_argument("--no-normalize-embeddings", dest="normalize_embeddings", action="store_false")
    parser.add_argument("--min-seq-len-export", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:2", help="GPU or CPU device id, e.g. cuda:1 or cpu")
    parser.add_argument("--dummy-encode", action="store_true",
                        help="Use dummy random vectors instead of SentenceTransformer encoding (for offline testing)")
    args = parser.parse_args()

    logger = logging.getLogger("Amazon2014PreProcessor")
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

    # 可选：使用 Dummy 编码器以避免联网下载模型
    if args.dummy_encode:
        import numpy as np
        import random

        class _DummySentenceTransformer:
            def __init__(self, *_, **__): pass
            def encode(self, sentences, **kwargs):
                np.random.seed(42)
                X = np.random.randn(len(sentences), 768).astype(np.float32)
                X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
                return X

        from sentence_transformers import SentenceTransformer
        SentenceTransformer = _DummySentenceTransformer  # 覆盖

    # 构造配置
    config = {
        "device": args.device,
        "category": args.category,
        "cache_dir": args.cache_dir,
        "metadata": args.metadata,
        "sent_emb_model": args.sent_emb_model,
        "sent_emb_batch_size": args.sent_emb_batch_size,
        "normalize_embeddings": args.normalize_embeddings,
        "min_seq_len_export": args.min_seq_len_export,
        "logger": logger,
    }

    logger.info(f"Running Amazon2014PreProcessor on device={args.device}")

    # 实例化并执行
    prep = Amazon2014PreProcessor(config)
    df_seq, itemid2emb = prep.run()

    # 输出关键结果
    logger.info(f"✅ Finished category={args.category}, users={len(df_seq)}, "
                f"items_with_emb={len(itemid2emb)}")
    logger.info(f"raw_seq.pkl saved at: {prep.seq_pkl_path}")
    logger.info(f"item2emb.pkl saved at: {prep.item2emb_pkl}")
    
# python preprocess.py --category Toys_and_Games --no-normalize-embeddings
# python preprocess.py --category Sports_and_Outdoors --no-normalize-embeddings
# nohup python -u preprocess.py --category Books --no-normalize-embeddings > preprocess.log 2>&1 &

