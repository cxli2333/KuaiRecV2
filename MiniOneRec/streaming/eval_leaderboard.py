import os
import re
import json
import math
import datetime
import requests
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import gradio as gr
from email.utils import parseaddr
from huggingface_hub import HfApi, hf_hub_download

# -- Configuration ---
HF_TOKEN = os.environ.get("HF_TOKEN")

REPO_ID_LEADERBOARD = os.environ.get("HF_LEADERBOARD_REPO_ID", "")
REPO_ID_GROUND_TRUTH = os.environ.get("HF_GROUND_TRUTH_REPO_ID", "")
REPO_ID_CONTACT = os.environ.get("HF_CONTACT_REPO_ID", "")

# Require explicit deployment targets before accessing private or remote data.
if HF_TOKEN and not all((REPO_ID_LEADERBOARD, REPO_ID_GROUND_TRUTH, REPO_ID_CONTACT)):
    raise ValueError(
        "Set HF_LEADERBOARD_REPO_ID, HF_GROUND_TRUTH_REPO_ID, and HF_CONTACT_REPO_ID "
        "when configuring HF_TOKEN. Ground-truth and contact repositories must be private."
    )

CSV_FILENAME = "leaderboard.csv"
GT_FILENAME = "ground_truth.tsv"
CONTACT_FILENAME = "contact_log.csv"

K_VALUES = [1, 5, 10, 20, 50]
MAX_GT_POSITIVES = max(K_VALUES)
MAX_DEPTH = 8

ACCURACY_METRICS = ["NDCG", "HR", "Recall", "MRR"]
DIST_METRICS = ["Coverage", "Gini"]
DIVERSITY_LAYER_METRICS = ["LayerUnique", "LayerEntropy"]

DIVERSITY_COLS = (
    [f"{m}@d{d}" for m in DIVERSITY_LAYER_METRICS for d in range(MAX_DEPTH)]
    + ["ILD"]
)
EVAL_INFO_COLS = ["Eval Users", "GT Interactions", "Avg GT/User"]
PREDICTION_TYPES = ["SID", "Item ID"]


BEHAVIORS = [
    "is_click", "is_long_view", "is_like", "is_follow",
    "is_forward", "is_comment", "is_search", "is_profile_enter",
]
BEHAVIOR_LABELS = {
    "is_click":         "Click",
    "is_long_view":     "Long View",
    "is_like":          "Like",
    "is_follow":        "Follow",
    "is_forward":       "Forward",
    "is_comment":       "Comment",
    "is_search":        "Search",
    "is_profile_enter": "Profile Enter",
}
LABEL_TO_BEHAVIOR = {v: k for k, v in BEHAVIOR_LABELS.items()}
SID_PATTERN = re.compile(r"^(?:<[^<>\s]+>){1,8}$")

MIN_ACCOUNT_AGE_DAYS = 60
EMAIL_REGEX = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")
CONTACT_COLUMNS = ["username", "email", "model_name", "organization", "submission_date"]
ADMIN_USERNAMES = set(
    u.strip() for u in os.environ.get("ADMIN_USERNAMES", "").split(",") if u.strip()
)

api = HfApi(token=HF_TOKEN)


# Column definitions
def get_columns():
    cols = ["Model Name", "Author", "Behavior", "ID Type"] + EVAL_INFO_COLS
    
    for m in ACCURACY_METRICS:
        for k in K_VALUES:
            cols.append(f"{m}@{k}")
    for m in DIST_METRICS:
        for k in K_VALUES:
            cols.append(f"{m}@{k}")
    cols += DIVERSITY_COLS      
    cols += ["Organization", "Date"]
    return cols

# Behavior / SID utilities
def normalize_behaviors(selected):
    if not selected:
        return []
    raw = set()
    for s in selected:
        if s in BEHAVIORS:
            raw.add(s)
        elif s in LABEL_TO_BEHAVIOR:
            raw.add(LABEL_TO_BEHAVIOR[s])
    return [b for b in BEHAVIORS if b in raw]


def behaviors_to_label(selected_behaviors):
    return " + ".join(BEHAVIOR_LABELS[b] for b in selected_behaviors)


def parse_sid_tokens(sid_str):
    """'<a><b><c>' -> ['a', 'b', 'c']"""
    if not sid_str:
        return []
    clean = str(sid_str).replace("<sid_start_flag>", "").replace("<sid_end_flag>", "")
    return re.findall(r"<([^>]+)>", clean)


def sid_list_to_string(sid_list_str):
    """Convert GT '383,429,486,...' -> '<383><429><486>...'"""
    if pd.isna(sid_list_str):
        return ""
    s = str(sid_list_str).strip()
    if not s or s.lower() == "nan":
        return ""
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return "".join(f"<{p}>" for p in parts)


# Ground truth
def load_ground_truth_df():
    print("🔄 [System] 正在加载 GT (TSV)...")
    if not HF_TOKEN:
        print("⚠️ [Warning] 无 Token，使用假数据。")
        rows = []
        for u in ["U101", "U102"]:
            for i in range(3):
                row = {
                    "time_ms": 1000 + i,
                    "item": f"item_{u}_{i}",
                    "user": u,
                    "sid_list": ",".join(str(j + i) for j in range(8)),
                }
                for b in BEHAVIORS:
                    row[b] = 1
                rows.append(row)
        df = pd.DataFrame(rows)
        df["sid_str"] = df["sid_list"].apply(sid_list_to_string)
        return df
    try:
        path = hf_hub_download(
            repo_id=REPO_ID_GROUND_TRUTH,
            filename=GT_FILENAME,
            repo_type="dataset",
            token=HF_TOKEN,
            force_download=True,
        )
        df = pd.read_csv(path, sep="\t")
        required = {"time_ms", "item", "user", "sid_list"} | set(BEHAVIORS)
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"GT 文件缺少列: {missing}")

        df["user"] = df["user"].astype(str)
        df["item"] = df["item"].astype(str)
        df["time_ms"] = pd.to_numeric(df["time_ms"], errors="coerce")
        for b in BEHAVIORS:
            df[b] = pd.to_numeric(df[b], errors="coerce").fillna(0).astype(int)
        df = df.dropna(subset=["time_ms"])
        df["sid_str"] = df["sid_list"].apply(sid_list_to_string)

        print(f"✅ GT 加载成功: rows={len(df)}, users={df['user'].nunique()}, "
              f"unique_sids={df['sid_str'].nunique()}")
        return df
    except Exception as e:
        print(f"❌ GT 加载失败: {e}")
        return pd.DataFrame(columns=["time_ms", "item", "user", "sid_list", "sid_str"] + BEHAVIORS)


GROUND_TRUTH_DF = load_ground_truth_df()
CATALOG_SIDS = (
    set(GROUND_TRUTH_DF["sid_str"].dropna().astype(str))
    if not GROUND_TRUTH_DF.empty
    else set()
)
CATALOG_SIDS.discard("")
CATALOG_SIZE = max(len(CATALOG_SIDS), 1)


def build_item_to_sid_mapping(df):
    if df.empty:
        return {}

    mapping_df = df[["item", "sid_str", "time_ms"]].copy()
    mapping_df["item"] = mapping_df["item"].astype(str).str.strip()
    mapping_df["sid_str"] = mapping_df["sid_str"].astype(str).str.strip()
    mapping_df = mapping_df[
        mapping_df["item"].ne("") & mapping_df["sid_str"].ne("")
    ]
    mapping_df["time_ms"] = pd.to_numeric(
        mapping_df["time_ms"], errors="coerce"
    )
    mapping_df = mapping_df.sort_values(
        ["time_ms"],
        ascending=[False],
        na_position="last",
        kind="mergesort",
    )
    mapping_df = mapping_df.drop_duplicates(subset=["item"], keep="first")
    item_to_sid = dict(zip(mapping_df["item"], mapping_df["sid_str"]))
    return item_to_sid


ITEM_TO_SID = build_item_to_sid_mapping(GROUND_TRUTH_DF)


def build_gt_for_behaviors(selected_behaviors):
    """Build a behavior-specific GT view from the full, untruncated test set."""
    if GROUND_TRUTH_DF.empty or not selected_behaviors:
        return {}
    df = GROUND_TRUTH_DF
    mask = pd.Series(True, index=df.index)
    for b in selected_behaviors:
        if b not in df.columns:
            return {}
        mask &= (df[b] == 1)
    df = df[mask].copy()
    if df.empty:
        return {}

    df = df[
        df["sid_str"].notna()
        & df["sid_str"].astype(str).str.strip().ne("")
    ]
    if df.empty:
        return {}

    # Filter behaviors first. Otherwise, truncating the raw history could drop
    # older but valid intersections such as Click + Like + Forward.
    df = df.sort_values(by=["user", "time_ms"], ascending=[True, False])
    df = df.drop_duplicates(subset=["user", "sid_str"], keep="first")
    df = df.groupby("user", sort=False).head(MAX_GT_POSITIVES)
    gt = {}
    for user, group in df.groupby("user", sort=False):
        gt[str(user)] = group["sid_str"].astype(str).tolist()
    return gt


# Leaderboard
def get_leaderboard_df():
    try:
        path = hf_hub_download(
            repo_id=REPO_ID_LEADERBOARD,
            filename=CSV_FILENAME,
            repo_type="dataset",
            token=HF_TOKEN,
            force_download=True,
        )
        df = pd.read_csv(path)
        for col in df.columns:
            if "@" in col or col in DIVERSITY_COLS or col in EVAL_INFO_COLS:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        sort_col = f"NDCG@{max(K_VALUES)}"
        if sort_col in df.columns:
            df = df.sort_values(by=[sort_col], ascending=False)
        return df.fillna("")
    except Exception as e:
        print(f"[LB] 加载失败: {e}")
        return pd.DataFrame(columns=get_columns())


# Email / account / submission frequency
def validate_email(raw_email):
    if not raw_email or not isinstance(raw_email, str):
        return False, "邮箱不能为空。"
    raw_email = raw_email.strip()
    if len(raw_email) > 254:
        return False, "邮箱过长。"
    _, parsed = parseaddr(raw_email)
    if not parsed or "@" not in parsed:
        return False, "邮箱格式错误。"
    if not EMAIL_REGEX.match(parsed):
        return False, "邮箱格式错误。"
    return True, parsed


def check_account_age(username):
    try:
        resp = requests.get(
            f"https://huggingface.co/api/users/{username}/overview",
            timeout=10,
        )
        if resp.status_code != 200:
            return False, f"无法查询 HF 账号 @{username}。"
        created_at = resp.json().get("createdAt")
        if not created_at:
            return False, "无法获取账号创建时间。"
        created_dt = datetime.datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        if created_dt.tzinfo is None:
            created_dt = created_dt.replace(tzinfo=datetime.timezone.utc)
        now = datetime.datetime.now(datetime.timezone.utc)
        if now - created_dt < datetime.timedelta(days=MIN_ACCOUNT_AGE_DAYS):
            return False, f"账号注册时间不足 {MIN_ACCOUNT_AGE_DAYS} 天。"
        return True, "ok"
    except Exception as e:
        return False, f"账号校验失败：{e}"

def load_contact_log():
    if not HF_TOKEN:
        return pd.DataFrame(columns=CONTACT_COLUMNS)
    try:
        path = hf_hub_download(
            repo_id=REPO_ID_CONTACT,
            filename=CONTACT_FILENAME,
            repo_type="dataset",
            token=HF_TOKEN,
            force_download=True,
        )
        df = pd.read_csv(path)
        for col in CONTACT_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df[CONTACT_COLUMNS]
    except Exception as e:
        print(f"[Contact] log 加载失败: {e}")
        return pd.DataFrame(columns=CONTACT_COLUMNS)

def has_submitted_today(username):
    df = load_contact_log()
    if df.empty:
        return False
    today = datetime.date.today().isoformat()
    mask = (df["username"].astype(str).str.strip() == str(username).strip()) & \
           (df["submission_date"].astype(str).str.strip() == today)
    return bool(mask.any())

def append_contact_log(username, email, model_name, organization):
    df = load_contact_log()
    new_row = {
        "username": str(username).strip(),
        "email": str(email).strip(),
        "model_name": str(model_name).strip(),
        "organization": str(organization).strip(),
        "submission_date": datetime.date.today().isoformat(),
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    tmp = "_contact_tmp.csv"
    df.to_csv(tmp, index=False)
    try:
        api.upload_file(
            path_or_fileobj=tmp,
            path_in_repo=CONTACT_FILENAME,
            repo_id=REPO_ID_CONTACT,
            repo_type="dataset",
            token=HF_TOKEN,
        )
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass

# Diversity / distribution utilities
def gini_coefficient(x):
    x = np.array(x, dtype=np.float64)
    if x.size == 0 or x.sum() == 0:
        return 0.0
    if np.amin(x) < 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    ranks = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(ranks * x)) / (n * x.sum()) - (n + 1) / n)

def calculate_layer_unique_entropy(parsed_batch, max_depth=MAX_DEPTH):
    layer_token_counts = [Counter() for _ in range(max_depth)]
    for tokens in parsed_batch:
        for d in range(min(len(tokens), max_depth)):
            layer_token_counts[d][tokens[d]] += 1
    unique_counts = np.zeros(max_depth)
    entropy = np.zeros(max_depth)
    for d in range(max_depth):
        c = layer_token_counts[d]
        if not c:
            continue
        unique_counts[d] = len(c)
        total = sum(c.values())
        if total > 0:
            probs = np.array([v / total for v in c.values()])
            entropy[d] = -float(np.sum(probs * np.log2(probs + 1e-10)))
    return unique_counts, entropy

def calculate_intra_list_diversity(parsed_batch, max_depth=MAX_DEPTH):
    if len(parsed_batch) < 2:
        return 0.0
    dissimilarities = []
    n = len(parsed_batch)
    for i in range(n):
        for j in range(i + 1, n):
            tokens1 = parsed_batch[i][:max_depth]
            tokens2 = parsed_batch[j][:max_depth]
            denominator = max(len(tokens1), len(tokens2))
            if denominator == 0:
                dissimilarities.append(0.0)
                continue
            common_prefix = 0
            for token1, token2 in zip(tokens1, tokens2):
                if token1 != token2:
                    break
                common_prefix += 1
            dissimilarities.append(1.0 - common_prefix / denominator)
    return float(np.mean(dissimilarities)) if dissimilarities else 0.0


def normalize_prediction_type(prediction_type):
    value = str(prediction_type or "").strip()
    return value if value in PREDICTION_TYPES else None


def normalize_prediction_list(raw_preds, prediction_type, uid):
    """Convert predictions to valid SIDs and silently drop unusable entries.

    Invalid syntax, catalog-outside SIDs, unknown item IDs, empty values, and
    duplicate predictions do not invalidate the whole submission. They are
    removed before ranking metrics are calculated for this user.
    """
    if not isinstance(raw_preds, list):
        return None, f"用户 {uid} 的预测必须是列表。"

    sid_values = []
    seen_sids = set()
    for raw_value in raw_preds[:max(K_VALUES)]:
        value = str(raw_value).strip()
        if not value:
            continue
        if prediction_type == "SID":
            if not SID_PATTERN.fullmatch(value):
                continue
            if value not in CATALOG_SIDS:
                continue
            sid = value
        else:
            if value not in ITEM_TO_SID:
                continue
            sid = ITEM_TO_SID[value]
        if sid in seen_sids:
            continue
        seen_sids.add(sid)
        sid_values.append(sid)

    return sid_values, None

# Scoring: SID string matching and multiple metrics
def _per_user_accuracy(pred_list, true_set, k_list):
    """Return {NDCG@k / HR@k / Recall@k / MRR@k}."""
    results = {}
    if not true_set:
        return results
    max_k = max(k_list)
    pred = pred_list[:max_k]

    # First hit position for MRR
    first_hit = -1
    hits_map = []
    for i, sid in enumerate(pred):
        h = 1 if sid in true_set else 0
        hits_map.append(h)
        if h and first_hit < 0:
            first_hit = i
    hits_map += [0] * (max_k - len(hits_map))

    for k in k_list:
        hits_k = hits_map[:k]
        num_hits = sum(hits_k)
        results[f"HR@{k}"]     = 1.0 if num_hits > 0 else 0.0
        results[f"Recall@{k}"] = num_hits / len(true_set)
        dcg  = sum(1.0 / math.log2(i + 2) for i, h in enumerate(hits_k) if h)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(true_set), k)))
        results[f"NDCG@{k}"]   = (dcg / idcg) if idcg > 0 else 0.0
        if 0 <= first_hit < k:
            results[f"MRR@{k}"] = 1.0 / (first_hit + 1)
        else:
            results[f"MRR@{k}"] = 0.0
    return results

def calculate_score(user_filepath, selected_behaviors, prediction_type="SID"):
    try:
        prediction_type = normalize_prediction_type(prediction_type)
        if prediction_type is None:
            return f"Error: ID Type 必须是以下之一: {PREDICTION_TYPES}"

        with open(user_filepath, "r", encoding="utf-8") as f:
            submissions = json.load(f)
        if not isinstance(submissions, dict):
            return "Error: JSON 格式应为 Dict[str, List[str]]。"

        gt = build_gt_for_behaviors(selected_behaviors)
        if not gt:
            return "Error: 所选行为组合下 GT 为空，请重新选择行为。"

        accuracy_total = {
            f"{m}@{k}": 0.0 for m in ACCURACY_METRICS for k in K_VALUES
        }
        count = len(gt)
        # Global recommendation counts for Coverage / Gini
        global_counts = {k: defaultdict(int) for k in K_VALUES}
        layer_unique_total = np.zeros(MAX_DEPTH)
        layer_entropy_total = np.zeros(MAX_DEPTH)
        ild_total = 0.0

        for uid, true_list in gt.items():
            raw_preds = submissions.get(uid, [])
            preds, error = normalize_prediction_list(
                raw_preds, prediction_type, uid
            )
            if error:
                return f"Error: {error}"

            true_set = set(true_list)
            res = _per_user_accuracy(preds, true_set, K_VALUES)
            for metric, value in res.items():
                accuracy_total[metric] += value

            for k in K_VALUES:
                for sid in preds[:k]:
                    # Unknown SIDs occupy a rank but cannot inflate coverage.
                    if sid in CATALOG_SIDS:
                        global_counts[k][sid] += 1

            parsed = [parse_sid_tokens(p) for p in preds[:max(K_VALUES)]]
            lu, le = calculate_layer_unique_entropy(parsed, MAX_DEPTH)
            layer_unique_total += lu
            layer_entropy_total += le
            ild_total += calculate_intra_list_diversity(parsed, MAX_DEPTH)

        gt_interactions = sum(len(true_list) for true_list in gt.values())
        result = {
            "Eval Users": count,
            "GT Interactions": gt_interactions,
            "Avg GT/User": round(gt_interactions / count, 5),
        }
        for k_, v_ in accuracy_total.items():
            result[k_] = round(v_ / count, 5)

        for k in K_VALUES:
            # Include zero-exposure catalog items in Gini.
            counts_list = [global_counts[k].get(sid, 0) for sid in CATALOG_SIDS]
            unique_n = len(global_counts[k])
            coverage = unique_n / CATALOG_SIZE if CATALOG_SIZE > 0 else 0.0
            gini = gini_coefficient(counts_list) if counts_list else 0.0
            result[f"Coverage@{k}"] = round(min(coverage, 1.0), 5)
            result[f"Gini@{k}"]     = round(gini, 5)

        layer_unique_avg  = layer_unique_total / count
        layer_entropy_avg = layer_entropy_total / count
        for d in range(MAX_DEPTH):
            result[f"LayerUnique@d{d}"]  = round(float(layer_unique_avg[d]), 5)
            result[f"LayerEntropy@d{d}"] = round(float(layer_entropy_avg[d]), 5)
        result["ILD"] = round(float(ild_total / count), 5)

        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error: {e}"

# Main submission entry point
def process_submission(
    model_name, author_name, org_name, user_email, model_url,
    file_obj, prediction_type, behaviors,
    profile: gr.OAuthProfile | None = None,
):
    cur_df = get_leaderboard_df()

    # 1) Authentication
    if profile is None or not getattr(profile, "username", None):
        return "❌ 请先点击 'Sign in with Hugging Face' 登录。", cur_df
    username = profile.username

    # 2) Account age, exempting administrators
    if username not in ADMIN_USERNAMES:
        ok, msg = check_account_age(username)
        if not ok:
            return f"❌ {msg}", cur_df

    # 3) Submission frequency, exempting administrators
    if username not in ADMIN_USERNAMES:
        try:
            if has_submitted_today(username):
                return "❌ 您今天已经提交过 1 次，请明天再试。", cur_df
        except Exception as e:
            return f"❌ 频率检查失败，请稍后重试: {e}", cur_df

    # 4) Email
    ok, email_or_msg = validate_email(user_email)
    if not ok:
        return f"❌ {email_or_msg}", cur_df
    valid_email = email_or_msg

    # 5) Behaviors
    sel = normalize_behaviors(behaviors)
    if not sel:
        return "❌ 请至少勾选一个 Behavior。", cur_df
    prediction_type = normalize_prediction_type(prediction_type)
    if prediction_type is None:
        return f"❌ ID Type 必须是以下之一: {PREDICTION_TYPES}", cur_df

    # 6) Basic fields
    if not file_obj:
        return "❌ 请上传 JSON 文件。", cur_df
    if not HF_TOKEN:
        return "❌ 服务器未配置 HF_TOKEN。", cur_df
    model_name = str(model_name).strip()
    author_name = str(author_name).strip()
    org_name = str(org_name).strip()
    if not model_name:
        return "❌ Model Name 必填。", cur_df
    if not author_name:
        return "❌ Author 必填。", cur_df
    if not org_name:
        return "❌ Organization 必填。", cur_df

    # 7) URL and injection mitigation
    model_url = str(model_url).strip()
    safe_name = model_name.replace("[", "(").replace("]", ")")
    if model_url.startswith("http"):
        display_name = f"[{safe_name}]({model_url})"
    else:
        display_name = safe_name

    # 8) Scoring; failed calculations do not consume the submission quota
    result = calculate_score(file_obj.name, sel, prediction_type)
    if isinstance(result, str):
        return f"❌ {result}", cur_df

    # 9) Write the leaderboard
    new_data = {
        "Model Name":   display_name,
        "Author":       author_name,
        "Behavior":     behaviors_to_label(sel),
        "ID Type":      prediction_type,
        **result,
        "Organization": org_name,
        "Date":         datetime.datetime.now().strftime("%Y-%m-%d"),
    }
    df = pd.concat([cur_df, pd.DataFrame([new_data])], ignore_index=True)
    for col in df.columns:
        if "@" in col or col in DIVERSITY_COLS or col in EVAL_INFO_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    main_k = f"NDCG@{max(K_VALUES)}"
    if main_k in df.columns:
        df = df.sort_values(by=[main_k], ascending=False)
    df = df.reindex(columns=get_columns())

    df.to_csv("temp.csv", index=False)
    try:
        api.upload_file(
            path_or_fileobj="temp.csv",
            path_in_repo=CSV_FILENAME,
            repo_id=REPO_ID_LEADERBOARD,
            repo_type="dataset",
            token=HF_TOKEN,
        )
    except Exception as e:
        return f"❌ 上传 leaderboard 失败，未消耗当日额度: {e}", df

    # 10) Write the contact log last to consume the quota
    if username not in ADMIN_USERNAMES:
        try:
            append_contact_log(username, valid_email, model_name, org_name)
        except Exception as e:
            print(f"[WARN] leaderboard 已写但 contact log 失败: {e}")
            return f"⚠️ 已上榜但频控记录失败: {e}", df

    return (
        f"✅ 提交成功 @{username} | ID Type: {prediction_type} | "
        f"Behavior: {behaviors_to_label(sel)}"
    ), df
