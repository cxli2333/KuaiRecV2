import os
import math
import pickle
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from logging import getLogger

class FAISSQuantizer:
    """
    从 {cache_dir}/Amazon2014/{category}/processed/item2emb_<model_tag>.pkl 读取 item->embedding，
    用 FAISS 训练量化器并输出 item->codes 到 item2codes_<model_tag>_<quantizer>.pkl。

    config 关键字段（与 Amazon2014PreProcessor 对齐）：
        # 基础
        - device: 'cpu' | 'cuda'（仅用于日志/可选 GPU）
        - category: str
        - cache_dir: str
        - logger: logging.Logger

        # 嵌入文件
        - sent_emb_model: str               # 用于推导 model_tag（或显式给 model_tag）
        - model_tag: str (可选)             # 默认从 sent_emb_model basename 推导

        # 量化器与超参
        - quantizer: 'opq_pq' | 'rq_kmeans' | 'rq_vae'(占位，未实现；抛异常)
        - n_digit: int                      # 子码本个数 M
        - codebook_size: int                # 每位码本大小 K，2 的幂
        - faiss_omp_num_threads: int = 8
        - disable_opq: bool = False
        - opq_use_gpu: bool = False
        - opq_gpu_id: int = 0
        - rq_kmeans_niters: int = 20
        - rq_kmeans_seed: int = 1234

        # 可选：
        - train_mask: np.ndarray(bool)      # (N,) 或 (N+1,)（含 PAD=0），默认全 True
        - verbose: bool = True
    """

    def __init__(self, config: dict, train_mask: Optional[np.ndarray] = None):
        self.config = dict(config)  # 浅拷贝，避免外部被改动
        self.logger = self.config.get('logger', getLogger(__name__))
        self.device = self.config.get('device', 'cpu')
        self.category = self.config['category']
        self.cache_dir = self.config['cache_dir']
        # model_tag：优先显式字段；否则从 sent_emb_model basename 推导
        if 'model_tag' in self.config and self.config['model_tag']:
            self.model_tag = self.config['model_tag']
        # 量化器与超参（含默认）
        self.quantizer = self.config.get('quantizer', 'opq_pq')
        self.n_digit = int(self.config.get('n_digit', 4))
        self.codebook_size = int(self.config.get('codebook_size', 256))
        self.faiss_omp_num_threads = int(self.config.get('faiss_omp_num_threads', 8))
        self.disable_opq = bool(self.config.get('disable_opq', False))
        self.opq_use_gpu = bool(self.config.get('opq_use_gpu', False))
        self.opq_gpu_id = int(self.config.get('opq_gpu_id', 0))
        self.rq_kmeans_niters = int(self.config.get('rq_kmeans_niters', 20))
        self.rq_kmeans_seed = int(self.config.get('rq_kmeans_seed', 1234))
        self.verbose = bool(self.config.get('verbose', True))

        # train_mask：参数优先，其次 config，最后全 True
        if train_mask is not None:
            self.train_mask = np.asarray(train_mask, dtype=bool)
        else:
            tm = self.config.get('train_mask', None)
            self.train_mask = None if tm is None else np.asarray(tm, dtype=bool)

        # 路径
        self.proc_dir = os.path.join(self.cache_dir, "Amazon2014", self.category, "processed")
        os.makedirs(self.proc_dir, exist_ok=True)
        self.item2emb_pkl = os.path.join(self.proc_dir, f"item2emb_{self.model_tag}.pkl")
        self.item2codes_pkl = os.path.join(self.proc_dir, f"item2codes_{self.model_tag}_{self.quantizer}.pkl")

        # 校验
        self._check_power_of_two(self.codebook_size)

    # ============ 公共入口 ============
    def fit_transform(self) -> str:
        self.logger.info(f"Running FAISSQuantizer on device={self.device}")
        self.logger.info(f"category={self.category}, cache_dir={self.cache_dir}")
        self.logger.info(f"model_tag={self.model_tag}, quantizer={self.quantizer}, "
                    f"M={self.n_digit}, K={self.codebook_size}")

        item2emb = self._load_item2emb()
        embs, items = self._stack_embs(item2emb)  # [N, d]
        N, d = embs.shape
        mask = self._assert_or_default_mask(N, self.train_mask)

        self.logger.info(f"Embeddings loaded: N={N}, d={d}; train_mask true={mask.sum()}")

        if self.quantizer == "rq_kmeans":
            codes = self._rq_kmeans_quantize(
                embs, self.n_digit, self.codebook_size, mask,
                niter=self.rq_kmeans_niters, seed=self.rq_kmeans_seed
            )
        elif self.quantizer == "opq_pq":
            codes = self._opq_pq_quantize(
                embs, self.n_digit, self.codebook_size, mask,
                faiss_omp_num_threads=self.faiss_omp_num_threads,
                disable_opq=self.disable_opq,
                use_gpu=self.opq_use_gpu,
                gpu_id=self.opq_gpu_id
            )
        elif self.quantizer == "base_k":
            self.logger.info(
                f"[BaseK] Compressing ID space with base K={self.codebook_size}"
            )
            codes = self._base_k_quantize(
                N=embs.shape[0],
                K=self.codebook_size
            )
            self.logger.info(
                f"[BaseK] n_digit={codes.shape[1]}, capacity={self.codebook_size ** codes.shape[1]}"
            )

        elif self.quantizer == "rq_vae":
            raise NotImplementedError("quantizer='rq_vae' 尚未实现。请在此处接入你的 RQ-VAE。")
        else:
            raise ValueError(f"Unknown quantizer: {self.quantizer}")

        # 写出 item -> codes(list[int])
        item2codes: Dict[str, List[int]] = {it: [int(x) for x in codes[i].tolist()] for i, it in enumerate(items)}
        with open(self.item2codes_pkl, "wb") as f:
            pickle.dump(item2codes, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.logger.info(f"Saved item2codes -> {self.item2codes_pkl} (items={len(item2codes)})")

        return self.item2codes_pkl

    def set_train_mask(self, mask: np.ndarray):
        self.train_mask = np.asarray(mask, dtype=bool)

    @staticmethod
    def _check_power_of_two(x: int):
        ok = x > 0 and ((x & (x - 1)) == 0)
        if not ok:
            raise ValueError(f"codebook_size must be a power of 2, got {x}")

    def _load_item2emb(self) -> Dict[str, np.ndarray]:
        if not os.path.exists(self.item2emb_pkl):
            raise FileNotFoundError(f"Embeddings file not found: {self.item2emb_pkl}")
        with open(self.item2emb_pkl, "rb") as f:
            item2emb = pickle.load(f)
        if not isinstance(item2emb, dict):
            raise ValueError(f"Expect dict for item2emb, got {type(item2emb)}")
        self.logger.info(f"Loaded item2emb: {len(item2emb)} items from {self.item2emb_pkl}")
        return item2emb

    @staticmethod
    def _stack_embs(item2emb: Dict[str, np.ndarray],
                    item_order: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
        if item_order is None:
            items = sorted(item2emb.keys())
        else:
            items = [it for it in item_order if it in item2emb]
        embs = np.stack(
            [np.asarray(item2emb[it], dtype=np.float32) for it in items],
            axis=0
        ).astype(np.float32, copy=False)
        return embs, items

    @staticmethod
    def _assert_or_default_mask(N: int, train_mask: Optional[np.ndarray]) -> np.ndarray:
        if train_mask is None:
            return np.ones((N,), dtype=bool)
        m = np.asarray(train_mask, dtype=bool)
        if m.shape == (N + 1,):
            m = m[1:]  # 去掉 PAD
        assert m.shape == (N,), f"train_mask shape mismatch: expect ({N},) or ({N+1}, with PAD). Got: {m.shape}"
        return m

    # ============ 量化实现 ============
    @staticmethod
    def _rq_kmeans_quantize(embs: np.ndarray, n_digit: int, K: int, train_mask: np.ndarray, \
                            niter: int = 20, seed: int = 1234) -> np.ndarray:
        import faiss
        N, d = embs.shape
        residuals = embs.copy().astype(np.float32, copy=False)
        codes = np.zeros((N, n_digit), dtype=np.int64)
        for stage in range(n_digit):
            km = faiss.Kmeans(d=d, k=K, niter=int(niter), verbose=False, seed=int(seed) + stage)
            km.train(residuals[train_mask])
            centroids = np.asarray(km.centroids, dtype=np.float32)
            if centroids.ndim == 1:
                centroids = centroids.reshape(K, d)
            elif centroids.shape == (d, K):
                centroids = centroids.T
            index = faiss.IndexFlatL2(d)
            index.add(centroids)
            _, I = index.search(residuals, 1)  # [N,1]
            codes[:, stage] = I[:, 0].astype(np.int64)
            residuals = residuals - centroids[I[:, 0]]
        return codes
    
    def _opq_pq_quantize(self, embs: np.ndarray, n_digit: int, K: int, train_mask: np.ndarray,
                        faiss_omp_num_threads: int = 8, disable_opq: bool = False,
                        use_gpu: bool = False, gpu_id: int = 0) -> np.ndarray:
        import time
        import faiss

        nbits = int(math.log2(K))

        # --- 关键修复：有的 faiss-gpu 构建不导出 omp_set_num_threads ---
        if hasattr(faiss, "omp_set_num_threads"):
            faiss.omp_set_num_threads(int(faiss_omp_num_threads))
            self.logger.info(f"[FAISS] omp_set_num_threads({faiss_omp_num_threads})")
        else:
            # 备选：通过环境变量提示 OpenMP 线程数（许多 BLAS/OMP 实现会尊重它）
            if "OMP_NUM_THREADS" not in os.environ:
                os.environ["OMP_NUM_THREADS"] = str(faiss_omp_num_threads)
            self.logger.info(
                f"[FAISS] faiss.omp_set_num_threads 不可用，使用 OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']}"
            )
        # -------------------------------------------------------------

        idx_factory = (f"IVF1,PQ{n_digit}x{nbits}"
                    if disable_opq
                    else f"OPQ{n_digit},IVF1,PQ{n_digit}x{nbits}")
        metric = faiss.METRIC_INNER_PRODUCT
        index_cpu = faiss.index_factory(embs.shape[1], idx_factory, metric)

        index = index_cpu
        used_gpu = False
        if use_gpu and hasattr(faiss, "StandardGpuResources"):
            try:
                self.logger.info(f"[FAISS] Using GPU (id={gpu_id}) with index='{idx_factory}'")
                res = faiss.StandardGpuResources()
                co = faiss.GpuClonerOptions()
                co.useFloat16 = (n_digit >= 56)
                index = faiss.index_cpu_to_gpu(res, int(gpu_id), index_cpu, co)
                used_gpu = True
            except Exception as e:
                self.logger.warning(f"[FAISS] GPU path failed -> fallback to CPU. Reason: {e}")
                index = index_cpu
                used_gpu = False
        elif use_gpu:
            self.logger.warning("[FAISS] faiss-gpu not available -> fallback to CPU.")

        n_train = int(train_mask.sum())
        assert n_train > 0, "train_mask 至少包含一个样本"
        self.logger.info(f"[FAISS] Training index on TRAIN items only (train={n_train}/{embs.shape[0]})…")
        t0 = time.time(); index.train(embs[train_mask]); self.logger.info(f"[FAISS] train done in {time.time()-t0:.2f}s")
        t0 = time.time(); index.add(embs);               self.logger.info(f"[FAISS] add   done in {time.time()-t0:.2f}s")

        if used_gpu:
            index = faiss.index_gpu_to_cpu(index)

        if isinstance(index, faiss.IndexPreTransform):
            ivf_index = faiss.downcast_index(index.index)
        else:
            ivf_index = faiss.downcast_index(index)

        ivf = faiss.extract_index_ivf(ivf_index)
        inv = ivf.invlists
        ls = inv.list_size(0)
        self.logger.info(f"[FAISS] Invlists[0]: size={ls}, code_size={inv.code_size}")

        codes_ptr = inv.get_codes(0)
        ids_ptr = inv.get_ids(0)
        pq_codes_u8 = faiss.rev_swig_ptr(codes_ptr, ls * inv.code_size)
        ids = faiss.rev_swig_ptr(ids_ptr, ls).copy()
        pq_codes_u8 = pq_codes_u8.reshape(-1, inv.code_size)

        codes = np.zeros((embs.shape[0], n_digit), dtype=np.int64)
        for row, u8 in enumerate(pq_codes_u8):
            bs = faiss.BitstringReader(faiss.swig_ptr(u8), inv.code_size)
            for d in range(n_digit):
                codes[ids[row], d] = bs.read(nbits)
        return codes
    
    @staticmethod
    def _compute_min_n_digit(N: int, K: int) -> int:
        """
        最小的 n_digit，使得 K^n_digit >= N
        """
        assert K >= 2
        n_digit = int(math.ceil(math.log(N, K)))
        return max(1, n_digit)
    
    @staticmethod
    def _base_k_quantize(N: int, K: int) -> np.ndarray:
        """
        将 [0, 1, ..., N-1] 的整数 ID
        用 K 进制展开为最小长度的 code

        Returns:
            codes: np.ndarray, shape [N, n_digit], int64
        """
        n_digit = FAISSQuantizer._compute_min_n_digit(N, K)

        ids = np.arange(N, dtype=np.int64)          # [N]
        codes = np.zeros((N, n_digit), dtype=np.int64)

        for d in range(n_digit - 1, -1, -1):
            codes[:, d] = ids % K
            ids //= K

        return codes





if __name__ == "__main__":
    import argparse
    import logging

    parser = argparse.ArgumentParser(description="Train FAISS quantizer on item embeddings and export semantic codes.")
    parser.add_argument("--device", type=str, default="cuda:2",
                        help="Compute device: 'cpu' or 'cuda' or 'cuda:<id>' (used to select GPU id for FAISS).")
    parser.add_argument("--category", type=str, required=True,
                        default="Sports_and_Outdoors", help="Amazon2014 category")
    parser.add_argument("--cache_dir", type=str, default="../cache", help="Root cache directory.")
    parser.add_argument("--model_tag", type=str, default="sentence-t5-base",
                        help="Model tag suffix for item2emb_<model_tag>.pkl")
    parser.add_argument("--quantizer", type=str, default="opq_pq",
                        choices=["opq_pq", "rq_kmeans", "rq_vae", "base_k"], help="Quantizer type.")
    parser.add_argument("--n_digit", type=int, default=4, help="Number of subcodebooks (M).")
    parser.add_argument("--codebook_size", type=int, default=256, help="Codebook size (K, must be power of 2).")
    parser.add_argument("--disable_opq", action="store_true", help="Disable OPQ transform in opq_pq mode.")
    parser.add_argument("--rq_kmeans_niters", type=int, default=20, help="KMeans iterations per stage.")
    parser.add_argument("--rq_kmeans_seed", type=int, default=2025, help="Seed for RQ-KMeans.")
    parser.add_argument("--faiss_omp_num_threads", type=int, default=8, help="OpenMP threads for Faiss.")
    parser.add_argument("--verbose", action="store_true", help="Print detailed logs.")

    args = parser.parse_args()

    # ---- logging ----
    logger = logging.getLogger("FAISSQuantizer")
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

    # ---- parse device -> (opq_use_gpu, opq_gpu_id) ----
    dev = (args.device or "").lower().strip()
    if dev.startswith("cuda"):
        opq_use_gpu = True
        # support "cuda" or "cuda:<id>"
        if ":" in dev:
            try:
                opq_gpu_id = int(dev.split(":", 1)[1])
            except Exception:
                opq_gpu_id = 0
        else:
            opq_gpu_id = 0
    else:
        opq_use_gpu = False
        opq_gpu_id = 0

    # ---- config ----
    config = {
        "device": args.device,
        "category": args.category,
        "cache_dir": args.cache_dir,
        "model_tag": getattr(args, "model_tag", None),
        "quantizer": args.quantizer,
        "n_digit": args.n_digit,
        "codebook_size": args.codebook_size,
        "faiss_omp_num_threads": args.faiss_omp_num_threads,
        "disable_opq": args.disable_opq,
        "opq_use_gpu": opq_use_gpu,     # <— from device
        "opq_gpu_id": opq_gpu_id,       # <— from device
        "rq_kmeans_niters": args.rq_kmeans_niters,
        "rq_kmeans_seed": args.rq_kmeans_seed,
        "verbose": args.verbose,
        "logger": logger,
    }

    q = FAISSQuantizer(config)
    out_path = q.fit_transform()
    logger.info(f"✅ Done. Codes saved to: {out_path}")

