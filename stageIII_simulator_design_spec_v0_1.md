# stageIII_simulator 設計仕様書 v0.1

作成日: 2026-01-31
対応要件定義書: stageIII_simulator 要件定義書 v0.1

---

## 1. 概要

### 1.1 目的
本設計仕様書は、低分子創薬における組み合わせ合成ステージ（StageIII）シミュレーターの内部設計を定義する。

### 1.2 設計方針
- **シミュレーションコア（Python API）とUI（Streamlit）の分離**: コアロジックは独立したPythonパッケージとして実装し、UIから疎結合で利用可能とする。
- **Operatorのプラグイン設計**: 新しい探索戦略を容易に追加できるよう、抽象基底クラスと登録機構を提供する。
- **型安全性**: Python 3.10+の型ヒントを活用し、静的解析ツール（mypy）との互換性を確保する。

---

## 2. システム構成

### 2.1 ディレクトリ構造

```
stgiii_simulate/
├── stgiii_core/                 # シミュレーションコアパッケージ
│   ├── __init__.py
│   ├── config.py                # 設定・パラメータ定義
│   ├── matrix.py                # Matrix生成・管理
│   ├── indexer.py               # セルインデックス変換
│   ├── disclosure.py            # 開示管理
│   ├── operators/               # Operator実装
│   │   ├── __init__.py
│   │   ├── base.py              # 抽象基底クラス
│   │   ├── registry.py          # Operator登録機構
│   │   ├── random_operator.py   # 完全ランダム戦略
│   │   ├── fw_ridge.py          # Free-Wilson Ridge戦略
│   │   └── bayesian_fw.py       # ベイジアンFree-Wilson戦略
│   ├── simulation.py            # シミュレーション実行エンジン
│   ├── metrics.py               # 評価指標計算
│   └── results.py               # 結果データ構造
├── app/                         # Streamlit UI
│   ├── __init__.py
│   ├── main.py                  # エントリーポイント
│   ├── sidebar.py               # サイドバーコンポーネント
│   ├── display.py               # 結果表示コンポーネント
│   └── utils.py                 # UIユーティリティ
├── tests/                       # テストコード
│   ├── __init__.py
│   ├── test_matrix.py
│   ├── test_indexer.py
│   ├── test_disclosure.py
│   ├── test_operators.py
│   ├── test_simulation.py
│   └── test_integration.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

### 2.2 依存ライブラリ

| ライブラリ | バージョン | 用途 |
|-----------|-----------|------|
| numpy | >=1.24 | 数値計算、配列操作 |
| scipy | >=1.11 | 線形代数、統計関数 |
| scikit-learn | >=1.3 | Ridge回帰 |
| pandas | >=2.0 | データフレーム操作、CSV出力 |
| matplotlib | >=3.7 | ヒストグラム描画 |
| streamlit | >=1.28 | WebUI |

---

## 3. データ構造

### 3.1 設定パラメータ

```python
# stgiii_core/config.py

from dataclasses import dataclass, field
from typing import Literal
from enum import Enum

class OperatorType(Enum):
    """Operator戦略の種別"""
    RANDOM = "Random"
    FW_RIDGE = "FW-Ridge"
    BAYESIAN_FW_UCB = "Bayesian-FW-UCB"


@dataclass(frozen=True)
class SlotConfig:
    """スロット設定"""
    name: str                    # スロット名（"A", "B", "C", "D"）
    n_building_blocks: int       # BB数（10〜50）


@dataclass(frozen=True)
class SimulationConfig:
    """シミュレーション全体の設定（イミュータブル）"""

    # ユーザ指定パラメータ
    operator_type: OperatorType
    n_trials: int                          # 試行数（10〜1000）
    slots: tuple[SlotConfig, ...]          # スロット設定（2〜4個）
    main_effect_range: tuple[float, float] # 主作用の一様分布範囲 [low, high]
    error_clip_range: tuple[float, float]  # 誤差のclip範囲 [low, high]
    k_per_step: int                        # 1ステップで開示するセル数K（1〜5）
    topk_k: int                            # Top-kのk値（5, 10, 20）
    random_seed: int | None = None         # 再現性用シード（オプション）

    # システム内部デフォルト（変更不可）
    bias_range: tuple[float, float] = field(default=(7.5, 8.5))
    slot_bias_range: tuple[float, float] = field(default=(-0.5, 0.5))
    ridge_alpha: float = field(default=1.0)
    sigma_min: float = field(default=0.05)
    sigma_iter_max: int = field(default=5)
    sigma_convergence_threshold: float = field(default=1e-3)
    ucb_beta: float = field(default=1.0)
    obs_clip_range: tuple[float, float] = field(default=(5.0, 11.0))
    max_matrix_regeneration: int = field(default=5)
    max_initial_bb_retry: int = field(default=100)
    max_total_cells: int = field(default=100_000)

    def __post_init__(self) -> None:
        """バリデーション"""
        if not (2 <= len(self.slots) <= 4):
            raise ValueError("スロット数は2〜4である必要があります")
        for slot in self.slots:
            if not (10 <= slot.n_building_blocks <= 50):
                raise ValueError(f"BB数は10〜50である必要があります: {slot.name}")
        if self.n_total_cells > self.max_total_cells:
            raise ValueError(f"総セル数が上限を超えています: {self.n_total_cells} > {self.max_total_cells}")

    @property
    def n_slots(self) -> int:
        """スロット数"""
        return len(self.slots)

    @property
    def n_total_cells(self) -> int:
        """総セル数"""
        result = 1
        for slot in self.slots:
            result *= slot.n_building_blocks
        return result

    @property
    def slot_sizes(self) -> tuple[int, ...]:
        """各スロットのBB数のタプル"""
        return tuple(s.n_building_blocks for s in self.slots)

    @property
    def sigma_gen(self) -> float:
        """データ生成用の誤差標準偏差"""
        low, high = self.error_clip_range
        return (high - low) / 6.0
```

### 3.2 Matrix（評価値テンソル）

```python
# stgiii_core/matrix.py

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

@dataclass
class Matrix:
    """全組み合わせセルの評価値を保持するデータ構造"""

    y_true: NDArray[np.float64]       # 真値配列（1D, 長さ n_total）
    y_obs: NDArray[np.float64]        # 観測値配列（clipped, 1D, 長さ n_total）
    main_effects: list[NDArray[np.float64]]  # 各スロットの主作用（slot_bias込み）
    global_bias: float                # グローバルバイアス
    slot_biases: NDArray[np.float64]  # 各スロットのslot_bias
    errors: NDArray[np.float64]       # 各セルの誤差項（1D, 長さ n_total）
    top1_index: int                   # 正解セル（argmax(y_true)）のインデックス
    topk_indices: NDArray[np.int64]   # Top-kセルのインデックス配列

    @property
    def n_total(self) -> int:
        """総セル数"""
        return len(self.y_true)
```

### 3.3 セルインデックス変換

```python
# stgiii_core/indexer.py

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

@dataclass
class CellIndexer:
    """線形インデックスとN次元座標の相互変換"""

    slot_sizes: tuple[int, ...]  # 各スロットのBB数

    def __post_init__(self) -> None:
        # ストライド計算（row-major order）
        self._strides: tuple[int, ...] = self._compute_strides()
        self._n_total: int = int(np.prod(self.slot_sizes))

    def _compute_strides(self) -> tuple[int, ...]:
        """各スロットのストライドを計算"""
        strides = []
        stride = 1
        for size in reversed(self.slot_sizes):
            strides.append(stride)
            stride *= size
        return tuple(reversed(strides))

    @property
    def n_total(self) -> int:
        return self._n_total

    def coords_to_index(self, coords: tuple[int, ...]) -> int:
        """
        N次元座標を線形インデックスに変換

        Args:
            coords: 各スロットのBBインデックス（0-indexed）

        Returns:
            線形インデックス（0 <= index < n_total）
        """
        if len(coords) != len(self.slot_sizes):
            raise ValueError(f"座標の次元が不正: {len(coords)} != {len(self.slot_sizes)}")
        index = 0
        for coord, stride in zip(coords, self._strides):
            index += coord * stride
        return index

    def index_to_coords(self, index: int) -> tuple[int, ...]:
        """
        線形インデックスをN次元座標に変換

        Args:
            index: 線形インデックス（0 <= index < n_total）

        Returns:
            各スロットのBBインデックス（0-indexed）
        """
        if not (0 <= index < self._n_total):
            raise ValueError(f"インデックスが範囲外: {index}")
        coords = []
        remaining = index
        for stride in self._strides:
            coords.append(remaining // stride)
            remaining %= stride
        return tuple(coords)

    def batch_coords_to_indices(
        self, coords_array: NDArray[np.int64]
    ) -> NDArray[np.int64]:
        """
        複数の座標を一括で線形インデックスに変換

        Args:
            coords_array: 座標配列（shape: (n_cells, n_slots)）

        Returns:
            線形インデックス配列（shape: (n_cells,)）
        """
        strides = np.array(self._strides, dtype=np.int64)
        return np.sum(coords_array * strides, axis=1)

    def batch_indices_to_coords(
        self, indices: NDArray[np.int64]
    ) -> NDArray[np.int64]:
        """
        複数の線形インデックスを一括で座標に変換

        Args:
            indices: 線形インデックス配列（shape: (n_cells,)）

        Returns:
            座標配列（shape: (n_cells, n_slots)）
        """
        n_cells = len(indices)
        coords = np.zeros((n_cells, len(self.slot_sizes)), dtype=np.int64)
        remaining = indices.copy()
        for i, stride in enumerate(self._strides):
            coords[:, i] = remaining // stride
            remaining %= stride
        return coords
```

### 3.4 開示状態管理

```python
# stgiii_core/disclosure.py

from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray

@dataclass
class DisclosureState:
    """開示状態を管理するクラス"""

    n_total: int                                    # 総セル数
    _disclosed_mask: NDArray[np.bool_] = field(init=False)
    _disclosed_indices: list[int] = field(default_factory=list)
    _disclosed_values: list[float] = field(default_factory=list)
    _disclosure_order: list[int] = field(default_factory=list)  # 開示順序

    def __post_init__(self) -> None:
        self._disclosed_mask = np.zeros(self.n_total, dtype=bool)

    @property
    def n_disclosed(self) -> int:
        """開示済みセル数（ユニーク）"""
        return int(np.sum(self._disclosed_mask))

    @property
    def disclosed_indices(self) -> NDArray[np.int64]:
        """開示済みセルのインデックス配列"""
        return np.array(self._disclosed_indices, dtype=np.int64)

    @property
    def disclosed_values(self) -> NDArray[np.float64]:
        """開示済みセルの観測値配列"""
        return np.array(self._disclosed_values, dtype=np.float64)

    def is_disclosed(self, index: int) -> bool:
        """指定セルが開示済みか判定"""
        return bool(self._disclosed_mask[index])

    def disclose(self, indices: list[int], values: list[float]) -> int:
        """
        セルを開示する

        Args:
            indices: 開示するセルのインデックスリスト
            values: 対応する観測値リスト

        Returns:
            新規に開示されたセル数（重複除外）
        """
        new_count = 0
        for idx, val in zip(indices, values):
            if not self._disclosed_mask[idx]:
                self._disclosed_mask[idx] = True
                self._disclosed_indices.append(idx)
                self._disclosed_values.append(val)
                self._disclosure_order.append(idx)
                new_count += 1
        return new_count

    def get_undisclosed_indices(self) -> NDArray[np.int64]:
        """未開示セルのインデックス配列を取得"""
        return np.where(~self._disclosed_mask)[0]

    def contains_any(self, indices: NDArray[np.int64] | list[int]) -> bool:
        """指定したインデックスのいずれかが開示済みか判定"""
        return np.any(self._disclosed_mask[indices])
```

### 3.5 試行結果

```python
# stgiii_core/results.py

from dataclasses import dataclass
import pandas as pd
from typing import Literal

@dataclass
class TrialResult:
    """単一試行の結果"""

    trial_id: int
    method: str
    n_total_cells: int
    n_initial_disclosed: int
    k_value: int
    topk_k: int
    p_top1: int                    # Top-1到達時の開示セル数
    p_topk: int                    # Top-k到達時の開示セル数
    n_steps: int                   # 反復ステップ数（初期開示除く）
    hit_in_initial_top1: bool      # 初期開示でTop-1到達したか
    hit_in_initial_topk: bool      # 初期開示でTop-k到達したか
    runtime_ms: float | None = None


@dataclass
class SimulationResults:
    """全試行の結果を集約"""

    trials: list[TrialResult]
    config_summary: dict

    def to_dataframe(self) -> pd.DataFrame:
        """結果をDataFrameに変換"""
        records = []
        for t in self.trials:
            records.append({
                "trial_id": t.trial_id,
                "method": t.method,
                "n_total_cells": t.n_total_cells,
                "n_initial_disclosed": t.n_initial_disclosed,
                "k_value": t.k_value,
                "topk_k": t.topk_k,
                "P_top1": t.p_top1,
                "P_topk": t.p_topk,
                "n_steps": t.n_steps,
                "hit_in_initial_topk": t.hit_in_initial_topk,
            })
        return pd.DataFrame(records)

    def to_csv(self, path: str) -> None:
        """結果をCSVに出力"""
        self.to_dataframe().to_csv(path, index=False)

    def compute_statistics(self) -> dict:
        """統計量を計算"""
        df = self.to_dataframe()
        return {
            "P_top1": {
                "median": df["P_top1"].median(),
                "mean": df["P_top1"].mean(),
                "std": df["P_top1"].std(),
                "max": df["P_top1"].max(),
                "min": df["P_top1"].min(),
            },
            "P_topk": {
                "median": df["P_topk"].median(),
                "mean": df["P_topk"].mean(),
                "std": df["P_topk"].std(),
                "max": df["P_topk"].max(),
                "min": df["P_topk"].min(),
            },
        }
```

---

## 4. Operator設計（プラグイン機構）

### 4.1 抽象基底クラス

```python
# stgiii_core/operators/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol
import numpy as np
from numpy.typing import NDArray

from ..config import SimulationConfig
from ..indexer import CellIndexer
from ..disclosure import DisclosureState


class OperatorProtocol(Protocol):
    """Operatorが満たすべきインターフェース（静的型チェック用）"""

    def select_next_cells(
        self,
        disclosure_state: DisclosureState,
        k: int
    ) -> list[int]:
        """次に開示するセルを選択"""
        ...

    def update(
        self,
        new_indices: list[int],
        new_values: list[float]
    ) -> None:
        """新規開示データでモデルを更新"""
        ...


@dataclass
class OperatorContext:
    """Operatorに渡すコンテキスト情報"""

    config: SimulationConfig
    indexer: CellIndexer
    rng: np.random.Generator


class BaseOperator(ABC):
    """Operator抽象基底クラス"""

    # サブクラスで定義必須
    name: str = ""

    def __init__(self, context: OperatorContext) -> None:
        self.context = context
        self.config = context.config
        self.indexer = context.indexer
        self.rng = context.rng

    @abstractmethod
    def select_next_cells(
        self,
        disclosure_state: DisclosureState,
        k: int
    ) -> list[int]:
        """
        次に開示するセルを選択

        Args:
            disclosure_state: 現在の開示状態
            k: 選択するセル数

        Returns:
            選択したセルのインデックスリスト（長さ k）

        Note:
            - 未開示セルからのみ選択すること
            - 既開示セルを選択した場合はエラー
        """
        pass

    @abstractmethod
    def update(
        self,
        new_indices: list[int],
        new_values: list[float]
    ) -> None:
        """
        新規開示データでモデルを更新

        Args:
            new_indices: 新規開示セルのインデックスリスト
            new_values: 対応する観測値リスト
        """
        pass

    def reset(self) -> None:
        """
        Operatorの内部状態をリセット（新規試行開始時に呼ばれる）

        Note:
            サブクラスで必要に応じてオーバーライド
        """
        pass

    def _validate_selection(
        self,
        selected: list[int],
        disclosure_state: DisclosureState
    ) -> None:
        """選択の妥当性を検証"""
        for idx in selected:
            if disclosure_state.is_disclosed(idx):
                raise ValueError(f"既開示セルを選択: {idx}")

    def _random_tiebreak(
        self,
        candidates: NDArray[np.int64],
        scores: NDArray[np.float64],
        k: int
    ) -> list[int]:
        """
        スコア上位k個を選択（同点時はランダム）

        Args:
            candidates: 候補セルのインデックス配列
            scores: 対応するスコア配列
            k: 選択数

        Returns:
            選択されたインデックスのリスト
        """
        # スコアでソート（降順）、同点時はランダム順
        n = len(candidates)
        random_tiebreaker = self.rng.random(n)
        sorted_idx = np.lexsort((random_tiebreaker, -scores))
        return candidates[sorted_idx[:k]].tolist()
```

### 4.2 Operator登録機構

```python
# stgiii_core/operators/registry.py

from typing import Type, Callable
from .base import BaseOperator, OperatorContext
from ..config import OperatorType

# Operatorクラスの登録用辞書
_OPERATOR_REGISTRY: dict[OperatorType, Type[BaseOperator]] = {}


def register_operator(
    operator_type: OperatorType
) -> Callable[[Type[BaseOperator]], Type[BaseOperator]]:
    """
    Operatorクラスを登録するデコレータ

    Usage:
        @register_operator(OperatorType.RANDOM)
        class RandomOperator(BaseOperator):
            ...
    """
    def decorator(cls: Type[BaseOperator]) -> Type[BaseOperator]:
        if operator_type in _OPERATOR_REGISTRY:
            raise ValueError(f"Operator already registered: {operator_type}")
        _OPERATOR_REGISTRY[operator_type] = cls
        return cls
    return decorator


def get_operator(
    operator_type: OperatorType,
    context: OperatorContext
) -> BaseOperator:
    """
    登録されたOperatorをインスタンス化して取得

    Args:
        operator_type: Operatorの種別
        context: Operatorコンテキスト

    Returns:
        初期化されたOperatorインスタンス

    Raises:
        ValueError: 未登録のOperatorType
    """
    if operator_type not in _OPERATOR_REGISTRY:
        raise ValueError(f"Unknown operator type: {operator_type}")
    cls = _OPERATOR_REGISTRY[operator_type]
    return cls(context)


def list_operators() -> list[OperatorType]:
    """登録済みのOperatorType一覧を取得"""
    return list(_OPERATOR_REGISTRY.keys())
```

### 4.3 完全ランダム戦略

```python
# stgiii_core/operators/random_operator.py

from .base import BaseOperator, OperatorContext
from .registry import register_operator
from ..config import OperatorType
from ..disclosure import DisclosureState


@register_operator(OperatorType.RANDOM)
class RandomOperator(BaseOperator):
    """完全ランダム戦略"""

    name = "Random"

    def __init__(self, context: OperatorContext) -> None:
        super().__init__(context)

    def select_next_cells(
        self,
        disclosure_state: DisclosureState,
        k: int
    ) -> list[int]:
        """未開示セルから一様ランダムにK個を選択"""
        undisclosed = disclosure_state.get_undisclosed_indices()
        if len(undisclosed) < k:
            k = len(undisclosed)
        selected = self.rng.choice(undisclosed, size=k, replace=False)
        return selected.tolist()

    def update(
        self,
        new_indices: list[int],
        new_values: list[float]
    ) -> None:
        """ランダム戦略では更新不要"""
        pass
```

### 4.4 Free-Wilson Ridge戦略

```python
# stgiii_core/operators/fw_ridge.py

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import Ridge

from .base import BaseOperator, OperatorContext
from .registry import register_operator
from ..config import OperatorType
from ..disclosure import DisclosureState


@register_operator(OperatorType.FW_RIDGE)
class FreeWilsonRidgeOperator(BaseOperator):
    """古典的Free-Wilson（リッジ回帰）戦略"""

    name = "FW-Ridge"

    def __init__(self, context: OperatorContext) -> None:
        super().__init__(context)
        self.alpha = self.config.ridge_alpha
        self.model: Ridge | None = None
        self._all_X: NDArray[np.float64] | None = None  # 全セルの特徴量行列

    def reset(self) -> None:
        """状態リセット"""
        self.model = None

    def _build_design_matrix(
        self,
        indices: NDArray[np.int64] | list[int]
    ) -> NDArray[np.float64]:
        """
        Reference coding による設計行列を構築

        Args:
            indices: セルインデックス配列

        Returns:
            設計行列 X（shape: (n_samples, n_features)）

        Note:
            各スロットの最初のBBを基準カテゴリとする
            特徴量数 = sum(n_bb - 1 for each slot) + 1 (intercept)
        """
        indices_arr = np.asarray(indices)
        coords = self.indexer.batch_indices_to_coords(indices_arr)
        n_samples = len(indices_arr)

        # 特徴量数の計算
        n_features = 1  # intercept
        for size in self.config.slot_sizes:
            n_features += size - 1

        X = np.zeros((n_samples, n_features), dtype=np.float64)
        X[:, 0] = 1.0  # intercept

        col_offset = 1
        for slot_idx, size in enumerate(self.config.slot_sizes):
            for bb_idx in range(1, size):  # 0番目は基準
                mask = coords[:, slot_idx] == bb_idx
                X[mask, col_offset] = 1.0
                col_offset += 1

        return X

    def _precompute_all_features(self) -> NDArray[np.float64]:
        """全セルの特徴量行列を事前計算（予測用）"""
        if self._all_X is None:
            all_indices = np.arange(self.indexer.n_total)
            self._all_X = self._build_design_matrix(all_indices)
        return self._all_X

    def select_next_cells(
        self,
        disclosure_state: DisclosureState,
        k: int
    ) -> list[int]:
        """推定値μ_predの高いセルを上位からK個選択"""
        if self.model is None:
            # モデル未学習時はランダム選択
            undisclosed = disclosure_state.get_undisclosed_indices()
            selected = self.rng.choice(undisclosed, size=min(k, len(undisclosed)), replace=False)
            return selected.tolist()

        # 全セルの予測値を計算
        all_X = self._precompute_all_features()
        predictions = self.model.predict(all_X)

        # 未開示セルのみを対象に上位K個を選択
        undisclosed = disclosure_state.get_undisclosed_indices()
        undisclosed_scores = predictions[undisclosed]

        return self._random_tiebreak(undisclosed, undisclosed_scores, k)

    def update(
        self,
        new_indices: list[int],
        new_values: list[float]
    ) -> None:
        """Ridge回帰でモデルを更新"""
        # 累積データで再学習（オンライン更新ではなく全データで再学習）
        # Note: 効率化のため、disclosure_stateを参照する設計も考えられる
        pass  # select_next_cellsの直前に学習するよう変更

    def fit(
        self,
        indices: NDArray[np.int64],
        values: NDArray[np.float64]
    ) -> None:
        """
        開示済みデータでモデルを学習

        Args:
            indices: 開示済みセルのインデックス配列
            values: 対応する観測値配列
        """
        X = self._build_design_matrix(indices)
        self.model = Ridge(alpha=self.alpha, fit_intercept=False)
        self.model.fit(X, values)

    def get_coefficients_sum_to_zero(self) -> dict[str, NDArray[np.float64]]:
        """
        係数をsum-to-zero表現に変換して取得

        Returns:
            各スロットの係数辞書（キー: スロット名, 値: BB係数配列）
        """
        if self.model is None:
            raise ValueError("モデルが未学習です")

        coef = self.model.coef_
        result = {}

        col_offset = 1  # interceptをスキップ
        for slot_idx, slot_config in enumerate(self.config.slots):
            size = slot_config.n_building_blocks
            # Reference coding係数を取得（基準は0）
            ref_coefs = np.zeros(size)
            ref_coefs[1:] = coef[col_offset:col_offset + size - 1]

            # Sum-to-zeroに変換
            mean_coef = np.mean(ref_coefs)
            sum_to_zero_coefs = ref_coefs - mean_coef

            result[slot_config.name] = sum_to_zero_coefs
            col_offset += size - 1

        return result
```

### 4.5 ベイジアンFree-Wilson戦略

```python
# stgiii_core/operators/bayesian_fw.py

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve, cho_factor, cho_solve

from .base import BaseOperator, OperatorContext
from .registry import register_operator
from ..config import OperatorType
from ..disclosure import DisclosureState


@register_operator(OperatorType.BAYESIAN_FW_UCB)
class BayesianFreeWilsonOperator(BaseOperator):
    """ベイジアンFree-Wilson（MAP + Laplace + UCB）戦略"""

    name = "Bayesian-FW-UCB"

    def __init__(self, context: OperatorContext) -> None:
        super().__init__(context)
        self.alpha = self.config.ridge_alpha  # prior precision = 1/alpha
        self.beta = self.config.ucb_beta
        self.sigma_min = self.config.sigma_min
        self.sigma_iter_max = self.config.sigma_iter_max
        self.sigma_convergence = self.config.sigma_convergence_threshold

        # 内部状態
        self.theta_map: NDArray[np.float64] | None = None  # MAP推定値
        self.Sigma_theta: NDArray[np.float64] | None = None  # 係数共分散
        self.sigma: float = self.config.sigma_gen  # 観測ノイズ推定値
        self._all_X: NDArray[np.float64] | None = None

    def reset(self) -> None:
        """状態リセット"""
        self.theta_map = None
        self.Sigma_theta = None
        self.sigma = self.config.sigma_gen

    def _build_design_matrix(
        self,
        indices: NDArray[np.int64] | list[int]
    ) -> NDArray[np.float64]:
        """Reference codingによる設計行列を構築"""
        indices_arr = np.asarray(indices)
        coords = self.indexer.batch_indices_to_coords(indices_arr)
        n_samples = len(indices_arr)

        n_features = 1
        for size in self.config.slot_sizes:
            n_features += size - 1

        X = np.zeros((n_samples, n_features), dtype=np.float64)
        X[:, 0] = 1.0

        col_offset = 1
        for slot_idx, size in enumerate(self.config.slot_sizes):
            for bb_idx in range(1, size):
                mask = coords[:, slot_idx] == bb_idx
                X[mask, col_offset] = 1.0
                col_offset += 1

        return X

    def _precompute_all_features(self) -> NDArray[np.float64]:
        """全セルの特徴量行列を事前計算"""
        if self._all_X is None:
            all_indices = np.arange(self.indexer.n_total)
            self._all_X = self._build_design_matrix(all_indices)
        return self._all_X

    def fit(
        self,
        indices: NDArray[np.int64],
        values: NDArray[np.float64]
    ) -> None:
        """
        開示済みデータでMAP推定 + Laplace近似

        Args:
            indices: 開示済みセルのインデックス配列
            values: 対応する観測値配列
        """
        X = self._build_design_matrix(indices)
        y = values
        n_features = X.shape[1]

        # Prior precision matrix: Lambda^{-1} = alpha * I
        Lambda_inv = self.alpha * np.eye(n_features)

        # σ推定の反復
        sigma = self.sigma
        for _ in range(self.sigma_iter_max):
            sigma_old = sigma

            # MAP推定: theta = (X'X + sigma^2 * Lambda^{-1})^{-1} X'y
            XtX = X.T @ X
            H = XtX / (sigma ** 2) + Lambda_inv
            Xty = X.T @ y / (sigma ** 2)

            # コレスキー分解による安定した解法
            try:
                c, lower = cho_factor(H)
                theta = cho_solve((c, lower), Xty)
            except np.linalg.LinAlgError:
                # 特異行列の場合は通常の解法
                theta = np.linalg.solve(H, Xty)

            # 残差からσを更新
            residuals = y - X @ theta
            sigma_hat = np.sqrt(np.var(residuals, ddof=1))
            sigma = max(sigma_hat, self.sigma_min)

            # 収束判定
            if abs(sigma - sigma_old) / sigma_old < self.sigma_convergence:
                break

        self.sigma = sigma
        self.theta_map = theta

        # Laplace近似による係数共分散
        # Sigma_theta = H^{-1} = (X'X/sigma^2 + Lambda^{-1})^{-1}
        H = XtX / (self.sigma ** 2) + Lambda_inv
        try:
            c, lower = cho_factor(H)
            self.Sigma_theta = cho_solve((c, lower), np.eye(n_features))
        except np.linalg.LinAlgError:
            self.Sigma_theta = np.linalg.inv(H)

    def select_next_cells(
        self,
        disclosure_state: DisclosureState,
        k: int
    ) -> list[int]:
        """UCBスコアの高いセルを上位からK個選択"""
        if self.theta_map is None:
            # モデル未学習時はランダム選択
            undisclosed = disclosure_state.get_undisclosed_indices()
            selected = self.rng.choice(undisclosed, size=min(k, len(undisclosed)), replace=False)
            return selected.tolist()

        # 全セルの予測値と不確実性を計算
        all_X = self._precompute_all_features()
        mu_pred = all_X @ self.theta_map

        # 係数由来の不確実性: sigma_param^2 = x' Sigma_theta x
        # 効率化: 対角成分のみ計算
        sigma_param_sq = np.sum((all_X @ self.Sigma_theta) * all_X, axis=1)
        sigma_param = np.sqrt(np.maximum(sigma_param_sq, 0))

        # UCBスコア
        ucb_scores = mu_pred + self.beta * sigma_param

        # 未開示セルのみを対象に上位K個を選択
        undisclosed = disclosure_state.get_undisclosed_indices()
        undisclosed_scores = ucb_scores[undisclosed]

        return self._random_tiebreak(undisclosed, undisclosed_scores, k)

    def update(
        self,
        new_indices: list[int],
        new_values: list[float]
    ) -> None:
        """ベイズ更新（実際はfit()で全データ再計算）"""
        pass

    def predict_with_uncertainty(
        self,
        indices: NDArray[np.int64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        予測値と不確実性を計算

        Args:
            indices: 予測対象セルのインデックス配列

        Returns:
            (mu_pred, sigma_param, sigma_total)
            - mu_pred: 予測平均
            - sigma_param: 係数由来の不確実性（探索項用）
            - sigma_total: 総不確実性（予測区間用）
        """
        if self.theta_map is None or self.Sigma_theta is None:
            raise ValueError("モデルが未学習です")

        X = self._build_design_matrix(indices)
        mu_pred = X @ self.theta_map

        sigma_param_sq = np.sum((X @ self.Sigma_theta) * X, axis=1)
        sigma_param = np.sqrt(np.maximum(sigma_param_sq, 0))
        sigma_total = np.sqrt(sigma_param_sq + self.sigma ** 2)

        return mu_pred, sigma_param, sigma_total

    def get_coefficients_sum_to_zero(self) -> dict[str, NDArray[np.float64]]:
        """係数をsum-to-zero表現に変換して取得"""
        if self.theta_map is None:
            raise ValueError("モデルが未学習です")

        coef = self.theta_map
        result = {}

        col_offset = 1
        for slot_idx, slot_config in enumerate(self.config.slots):
            size = slot_config.n_building_blocks
            ref_coefs = np.zeros(size)
            ref_coefs[1:] = coef[col_offset:col_offset + size - 1]

            mean_coef = np.mean(ref_coefs)
            sum_to_zero_coefs = ref_coefs - mean_coef

            result[slot_config.name] = sum_to_zero_coefs
            col_offset += size - 1

        return result
```

---

## 5. Matrix生成

```python
# stgiii_core/matrix.py

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass

from .config import SimulationConfig
from .indexer import CellIndexer


@dataclass
class Matrix:
    """全組み合わせセルの評価値を保持"""
    y_true: NDArray[np.float64]
    y_obs: NDArray[np.float64]
    main_effects: list[NDArray[np.float64]]
    global_bias: float
    slot_biases: NDArray[np.float64]
    errors: NDArray[np.float64]
    top1_index: int
    topk_indices: NDArray[np.int64]

    @property
    def n_total(self) -> int:
        return len(self.y_true)


class MatrixGenerator:
    """Matrix生成器"""

    def __init__(
        self,
        config: SimulationConfig,
        indexer: CellIndexer,
        rng: np.random.Generator
    ) -> None:
        self.config = config
        self.indexer = indexer
        self.rng = rng

    def generate(self) -> Matrix:
        """
        Matrixを生成

        Returns:
            生成されたMatrix

        Raises:
            RuntimeError: max_matrix_regeneration回の再生成でも
                          argmaxが一意にならない場合
        """
        for attempt in range(self.config.max_matrix_regeneration):
            matrix = self._generate_single()
            if matrix is not None:
                return matrix

        raise RuntimeError(
            f"{self.config.max_matrix_regeneration}回の再生成でも "
            "argmaxが一意になりませんでした。設定条件を見直してください。"
        )

    def _generate_single(self) -> Matrix | None:
        """
        単一のMatrix生成を試行

        Returns:
            成功時はMatrix、argmaxが一意でない場合はNone
        """
        n_total = self.indexer.n_total

        # グローバルバイアス
        global_bias = self.rng.uniform(*self.config.bias_range)

        # スロットバイアス
        n_slots = self.config.n_slots
        slot_biases = self.rng.uniform(
            *self.config.slot_bias_range,
            size=n_slots
        )

        # 主作用（各スロット）
        main_low, main_high = self.config.main_effect_range
        main_effects: list[NDArray[np.float64]] = []
        for slot_idx, slot_config in enumerate(self.config.slots):
            n_bb = slot_config.n_building_blocks
            raw_main = self.rng.uniform(main_low, main_high, size=n_bb)
            main_with_bias = raw_main + slot_biases[slot_idx]
            main_effects.append(main_with_bias)

        # 誤差
        err_low, err_high = self.config.error_clip_range
        sigma_gen = self.config.sigma_gen
        errors_raw = self.rng.normal(0, sigma_gen, size=n_total)
        errors = np.clip(errors_raw, err_low, err_high)

        # y_trueの計算
        y_true = np.zeros(n_total, dtype=np.float64)
        for idx in range(n_total):
            coords = self.indexer.index_to_coords(idx)
            value = global_bias
            for slot_idx, bb_idx in enumerate(coords):
                value += main_effects[slot_idx][bb_idx]
            value += errors[idx]
            y_true[idx] = value

        # argmaxの一意性チェック
        max_val = np.max(y_true)
        max_indices = np.where(y_true == max_val)[0]
        if len(max_indices) > 1:
            return None  # 一意でない

        top1_index = int(max_indices[0])

        # Top-kインデックス
        topk_k = self.config.topk_k
        topk_indices = np.argsort(y_true)[-topk_k:][::-1]

        # y_obs（観測値、clipped）
        obs_low, obs_high = self.config.obs_clip_range
        y_obs = np.clip(y_true, obs_low, obs_high)

        return Matrix(
            y_true=y_true,
            y_obs=y_obs,
            main_effects=main_effects,
            global_bias=global_bias,
            slot_biases=slot_biases,
            errors=errors,
            top1_index=top1_index,
            topk_indices=topk_indices,
        )
```

---

## 6. シミュレーション実行エンジン

```python
# stgiii_core/simulation.py

import time
from dataclasses import dataclass
from typing import Callable
import numpy as np
from numpy.typing import NDArray

from .config import SimulationConfig
from .matrix import Matrix, MatrixGenerator
from .indexer import CellIndexer
from .disclosure import DisclosureState
from .operators.base import BaseOperator, OperatorContext
from .operators.registry import get_operator
from .results import TrialResult, SimulationResults


@dataclass
class InitialDisclosureResult:
    """初期開示の結果"""
    disclosed_indices: list[int]
    center_coords: tuple[int, ...]
    contains_top1: bool


class SimulationEngine:
    """シミュレーション実行エンジン"""

    def __init__(
        self,
        config: SimulationConfig,
        progress_callback: Callable[[int, int], None] | None = None
    ) -> None:
        """
        Args:
            config: シミュレーション設定
            progress_callback: 進捗コールバック (current, total) -> None
        """
        self.config = config
        self.progress_callback = progress_callback

        # 乱数生成器
        self.rng = np.random.default_rng(config.random_seed)

        # インデクサー
        self.indexer = CellIndexer(config.slot_sizes)

    def run(self) -> SimulationResults:
        """
        全試行を実行

        Returns:
            シミュレーション結果
        """
        trials: list[TrialResult] = []

        for trial_id in range(self.config.n_trials):
            if self.progress_callback:
                self.progress_callback(trial_id, self.config.n_trials)

            result = self._run_single_trial(trial_id)
            trials.append(result)

        if self.progress_callback:
            self.progress_callback(self.config.n_trials, self.config.n_trials)

        config_summary = {
            "operator_type": self.config.operator_type.value,
            "n_trials": self.config.n_trials,
            "n_slots": self.config.n_slots,
            "slot_sizes": self.config.slot_sizes,
            "n_total_cells": self.config.n_total_cells,
            "main_effect_range": self.config.main_effect_range,
            "error_clip_range": self.config.error_clip_range,
            "k_per_step": self.config.k_per_step,
            "topk_k": self.config.topk_k,
        }

        return SimulationResults(trials=trials, config_summary=config_summary)

    def _run_single_trial(self, trial_id: int) -> TrialResult:
        """単一試行を実行"""
        start_time = time.perf_counter()

        # Matrix生成
        generator = MatrixGenerator(self.config, self.indexer, self.rng)
        matrix = generator.generate()

        # Operator初期化
        context = OperatorContext(
            config=self.config,
            indexer=self.indexer,
            rng=self.rng
        )
        operator = get_operator(self.config.operator_type, context)
        operator.reset()

        # 開示状態初期化
        disclosure = DisclosureState(n_total=self.indexer.n_total)

        # 初期開示
        initial_result = self._initial_disclosure(matrix, disclosure)
        n_initial = disclosure.n_disclosed

        # 初期開示でTop-1/Top-k到達チェック
        hit_top1_initial = matrix.top1_index in initial_result.disclosed_indices
        hit_topk_initial = any(
            idx in initial_result.disclosed_indices
            for idx in matrix.topk_indices
        )

        # P_top1, P_topkの初期値（初期開示で到達した場合）
        p_top1: int | None = n_initial if hit_top1_initial else None
        p_topk: int | None = n_initial if hit_topk_initial else None

        # Operatorの初期学習
        if hasattr(operator, 'fit'):
            operator.fit(
                disclosure.disclosed_indices,
                disclosure.disclosed_values
            )

        # 反復ステップ
        n_steps = 0
        k = self.config.k_per_step

        while p_top1 is None:
            n_steps += 1

            # 次に開示するセルを選択
            selected = operator.select_next_cells(disclosure, k)

            # 開示
            values = [float(matrix.y_obs[idx]) for idx in selected]
            disclosure.disclose(selected, values)

            # Operatorの更新
            if hasattr(operator, 'fit'):
                operator.fit(
                    disclosure.disclosed_indices,
                    disclosure.disclosed_values
                )
            else:
                operator.update(selected, values)

            # Top-k到達チェック
            if p_topk is None:
                for idx in selected:
                    if idx in matrix.topk_indices:
                        p_topk = disclosure.n_disclosed
                        break

            # Top-1到達チェック
            if matrix.top1_index in selected:
                p_top1 = disclosure.n_disclosed
                break

        # Top-kが未到達の場合（Top-1より先にTop-kに到達していないケース）
        if p_topk is None:
            p_topk = p_top1

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return TrialResult(
            trial_id=trial_id,
            method=self.config.operator_type.value,
            n_total_cells=self.config.n_total_cells,
            n_initial_disclosed=n_initial,
            k_value=k,
            topk_k=self.config.topk_k,
            p_top1=p_top1,
            p_topk=p_topk,
            n_steps=n_steps,
            hit_in_initial_top1=hit_top1_initial,
            hit_in_initial_topk=hit_topk_initial,
            runtime_ms=elapsed_ms,
        )

    def _initial_disclosure(
        self,
        matrix: Matrix,
        disclosure: DisclosureState
    ) -> InitialDisclosureResult:
        """
        初期開示を実行

        Note:
            - 各スロットからBBを1つずつランダムに選択（中心座標）
            - 「N-1個のスロットを固定、残り1スロットを全開示」の和集合
            - 正解セルが含まれる場合は再抽選
        """
        for _ in range(self.config.max_initial_bb_retry):
            # 中心座標をランダムに選択
            center_coords = tuple(
                self.rng.integers(0, size)
                for size in self.config.slot_sizes
            )

            # 初期開示セル集合を構築
            disclosed_set: set[int] = set()

            for vary_slot in range(self.config.n_slots):
                # vary_slot以外を固定、vary_slotを全開示
                for bb_idx in range(self.config.slot_sizes[vary_slot]):
                    coords = list(center_coords)
                    coords[vary_slot] = bb_idx
                    idx = self.indexer.coords_to_index(tuple(coords))
                    disclosed_set.add(idx)

            disclosed_indices = list(disclosed_set)

            # 正解セルが含まれていないかチェック
            if matrix.top1_index not in disclosed_set:
                # 開示を実行
                values = [float(matrix.y_obs[idx]) for idx in disclosed_indices]
                disclosure.disclose(disclosed_indices, values)

                return InitialDisclosureResult(
                    disclosed_indices=disclosed_indices,
                    center_coords=center_coords,
                    contains_top1=False,
                )

        # 正解を回避できなかった場合はそのまま続行（実運用では稀）
        values = [float(matrix.y_obs[idx]) for idx in disclosed_indices]
        disclosure.disclose(disclosed_indices, values)

        return InitialDisclosureResult(
            disclosed_indices=disclosed_indices,
            center_coords=center_coords,
            contains_top1=True,
        )

    @staticmethod
    def calculate_initial_disclosure_count(slot_sizes: tuple[int, ...]) -> int:
        """
        初期開示セル数を計算（静的メソッド、UI表示用）

        Args:
            slot_sizes: 各スロットのBB数

        Returns:
            初期開示セル数（ユニーク）

        Note:
            和集合のサイズ = Σ(slot_size) - (n_slots - 1)
            （中心セルが重複してカウントされるため）
        """
        n_slots = len(slot_sizes)
        total = sum(slot_sizes)
        # 中心セルは各スロットで1回ずつカウントされるが、実際は1つ
        # 重複 = n_slots - 1
        return total - (n_slots - 1)
```

---

## 7. Streamlit UI設計

### 7.1 エントリーポイント

```python
# app/main.py

import streamlit as st
from .sidebar import render_sidebar
from .display import render_results


def main() -> None:
    """Streamlitアプリのエントリーポイント"""
    st.set_page_config(
        page_title="StageIII Simulator",
        page_icon="🧪",
        layout="wide"
    )

    st.title("StageIII Simulator")
    st.markdown("低分子創薬 組み合わせ合成ステージ シミュレーター")

    # サイドバーでパラメータ入力
    config, run_clicked = render_sidebar()

    # メイン領域
    if run_clicked and config is not None:
        render_results(config)
    elif config is None:
        st.warning("設定に問題があります。サイドバーを確認してください。")


if __name__ == "__main__":
    main()
```

### 7.2 サイドバー

```python
# app/sidebar.py

import streamlit as st
from stgiii_core.config import SimulationConfig, SlotConfig, OperatorType
from stgiii_core.simulation import SimulationEngine


def render_sidebar() -> tuple[SimulationConfig | None, bool]:
    """
    サイドバーをレンダリング

    Returns:
        (設定オブジェクト or None, 実行ボタンが押されたか)
    """
    st.sidebar.header("シミュレーション設定")

    # 手法選択
    operator_options = {
        "Random": OperatorType.RANDOM,
        "FW-Ridge": OperatorType.FW_RIDGE,
        "Bayesian-FW-UCB": OperatorType.BAYESIAN_FW_UCB,
    }
    operator_name = st.sidebar.selectbox(
        "探索手法",
        options=list(operator_options.keys()),
        index=0
    )
    operator_type = operator_options[operator_name]

    # 試行数
    n_trials = st.sidebar.number_input(
        "試行数",
        min_value=10,
        max_value=1000,
        value=100,
        step=10
    )

    # スロット数
    n_slots = st.sidebar.selectbox(
        "スロット数",
        options=[2, 3, 4],
        index=1  # デフォルト3
    )

    # 各スロットのBB数
    st.sidebar.subheader("各スロットのBB数")
    slot_names = ["A", "B", "C", "D"][:n_slots]
    slots: list[SlotConfig] = []
    for name in slot_names:
        n_bb = st.sidebar.slider(
            f"スロット {name}",
            min_value=10,
            max_value=50,
            value=20
        )
        slots.append(SlotConfig(name=name, n_building_blocks=n_bb))

    # 総セル数の計算と表示
    n_total = 1
    for s in slots:
        n_total *= s.n_building_blocks

    # 初期開示セル数の計算
    slot_sizes = tuple(s.n_building_blocks for s in slots)
    n_initial = SimulationEngine.calculate_initial_disclosure_count(slot_sizes)

    st.sidebar.markdown(f"**総セル数**: {n_total:,}")
    st.sidebar.markdown(f"**初期開示セル数**: {n_initial:,}")

    # 制限チェック
    if n_total > 100_000:
        st.sidebar.error("総セル数が100,000を超えています。条件を下げてください。")
        return None, False

    # 主作用範囲
    st.sidebar.subheader("主作用の範囲")
    main_col1, main_col2 = st.sidebar.columns(2)
    main_low = main_col1.number_input("下限", value=-1.0, step=0.1)
    main_high = main_col2.number_input("上限", value=1.0, step=0.1)

    # 誤差範囲
    st.sidebar.subheader("誤差の範囲")
    err_col1, err_col2 = st.sidebar.columns(2)
    err_low = err_col1.number_input("下限", value=-0.5, step=0.1)
    err_high = err_col2.number_input("上限", value=0.5, step=0.1)

    # 1ステップで開示するセル数K
    k_per_step = st.sidebar.selectbox(
        "1ステップで開示するセル数 (K)",
        options=[1, 2, 3, 4, 5],
        index=0
    )

    # Top-k の k
    topk_k = st.sidebar.selectbox(
        "Top-k の k",
        options=[5, 10, 20],
        index=1
    )

    # 実行ボタン
    run_clicked = st.sidebar.button("シミュレーション実行", type="primary")

    if run_clicked:
        try:
            config = SimulationConfig(
                operator_type=operator_type,
                n_trials=int(n_trials),
                slots=tuple(slots),
                main_effect_range=(float(main_low), float(main_high)),
                error_clip_range=(float(err_low), float(err_high)),
                k_per_step=int(k_per_step),
                topk_k=int(topk_k),
            )
            return config, True
        except ValueError as e:
            st.sidebar.error(f"設定エラー: {e}")
            return None, False

    return None, False
```

### 7.3 結果表示

```python
# app/display.py

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

from stgiii_core.config import SimulationConfig
from stgiii_core.simulation import SimulationEngine


def render_results(config: SimulationConfig) -> None:
    """シミュレーション結果を表示"""

    # 実行条件サマリ
    st.subheader("実行条件")
    col1, col2, col3 = st.columns(3)
    col1.metric("手法", config.operator_type.value)
    col2.metric("総セル数", f"{config.n_total_cells:,}")
    col3.metric("試行数", config.n_trials)

    # プログレスバー
    progress_bar = st.progress(0)
    status_text = st.empty()

    def progress_callback(current: int, total: int) -> None:
        progress = current / total if total > 0 else 0
        progress_bar.progress(progress)
        status_text.text(f"実行中... {current}/{total} 試行完了")

    # シミュレーション実行
    engine = SimulationEngine(config, progress_callback=progress_callback)
    results = engine.run()

    status_text.text("完了!")
    progress_bar.progress(1.0)

    # 統計量
    stats = results.compute_statistics()

    st.subheader("結果サマリ")

    # P_top1 統計量
    st.markdown("**P_top1（Top-1到達までの開示セル数）**")
    p1 = stats["P_top1"]
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Median", f"{p1['median']:.1f}")
    col2.metric("Mean", f"{p1['mean']:.1f}")
    col3.metric("STD", f"{p1['std']:.1f}")
    col4.metric("Min", f"{p1['min']:.0f}")
    col5.metric("Max", f"{p1['max']:.0f}")

    # P_topk 統計量
    st.markdown(f"**P_top{config.topk_k}（Top-{config.topk_k}到達までの開示セル数）**")
    pk = stats["P_topk"]
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Median", f"{pk['median']:.1f}")
    col2.metric("Mean", f"{pk['mean']:.1f}")
    col3.metric("STD", f"{pk['std']:.1f}")
    col4.metric("Min", f"{pk['min']:.0f}")
    col5.metric("Max", f"{pk['max']:.0f}")

    # ヒストグラム
    st.subheader("P_top1 ヒストグラム")
    df = results.to_dataframe()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["P_top1"], bins=30, edgecolor="black", alpha=0.7)
    ax.set_xlabel("P_top1 (Number of Disclosed Cells)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of P_top1 ({config.operator_type.value})")
    ax.axvline(p1["median"], color="red", linestyle="--", label=f"Median: {p1['median']:.1f}")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    # 試行別結果テーブル
    st.subheader("試行別結果")
    st.dataframe(df, use_container_width=True)

    # CSVダウンロード
    csv = df.to_csv(index=False)
    st.download_button(
        label="結果をCSVでダウンロード",
        data=csv,
        file_name="simulation_results.csv",
        mime="text/csv"
    )
```

---

## 8. エラーハンドリング

### 8.1 例外クラス定義

```python
# stgiii_core/exceptions.py

class StgIIISimulatorError(Exception):
    """シミュレータの基底例外クラス"""
    pass


class ConfigurationError(StgIIISimulatorError):
    """設定パラメータに関するエラー"""
    pass


class MatrixGenerationError(StgIIISimulatorError):
    """Matrix生成に関するエラー"""
    pass


class OperatorError(StgIIISimulatorError):
    """Operator処理に関するエラー"""
    pass


class CellLimitExceededError(ConfigurationError):
    """総セル数が上限を超えた"""
    def __init__(self, n_total: int, limit: int) -> None:
        self.n_total = n_total
        self.limit = limit
        super().__init__(
            f"総セル数が上限を超えています: {n_total:,} > {limit:,}"
        )


class UniqueArgmaxError(MatrixGenerationError):
    """argmaxが一意でない"""
    def __init__(self, attempts: int) -> None:
        self.attempts = attempts
        super().__init__(
            f"{attempts}回の再生成でもargmaxが一意になりませんでした"
        )
```

### 8.2 エラーハンドリング方針

| エラー種別 | 発生箇所 | 対応 |
|-----------|---------|------|
| 総セル数超過 | 設定時 | UI上でエラー表示、実行ボタン無効化 |
| argmax非一意 | Matrix生成 | 最大5回再生成、失敗時はエラー表示で停止 |
| 初期開示に正解含む | 初期開示 | 最大100回BB再抽選、失敗時は警告付きで続行 |
| Operator選択エラー | 反復ステップ | 既開示セル選択時は例外発生（バグ検出用） |

---

## 9. テスト設計

### 9.1 単体テスト

```python
# tests/test_indexer.py

import pytest
import numpy as np
from stgiii_core.indexer import CellIndexer


class TestCellIndexer:
    """CellIndexerの単体テスト"""

    def test_coords_to_index_2slots(self) -> None:
        """2スロットでの座標→インデックス変換"""
        indexer = CellIndexer((3, 4))
        assert indexer.coords_to_index((0, 0)) == 0
        assert indexer.coords_to_index((0, 1)) == 1
        assert indexer.coords_to_index((1, 0)) == 4
        assert indexer.coords_to_index((2, 3)) == 11

    def test_index_to_coords_2slots(self) -> None:
        """2スロットでのインデックス→座標変換"""
        indexer = CellIndexer((3, 4))
        assert indexer.index_to_coords(0) == (0, 0)
        assert indexer.index_to_coords(1) == (0, 1)
        assert indexer.index_to_coords(4) == (1, 0)
        assert indexer.index_to_coords(11) == (2, 3)

    def test_roundtrip(self) -> None:
        """座標⇔インデックスの往復変換"""
        indexer = CellIndexer((10, 20, 15))
        for idx in range(indexer.n_total):
            coords = indexer.index_to_coords(idx)
            assert indexer.coords_to_index(coords) == idx

    def test_batch_operations(self) -> None:
        """バッチ操作のテスト"""
        indexer = CellIndexer((5, 6, 7))
        indices = np.array([0, 10, 50, 100])
        coords = indexer.batch_indices_to_coords(indices)
        recovered = indexer.batch_coords_to_indices(coords)
        np.testing.assert_array_equal(indices, recovered)
```

```python
# tests/test_disclosure.py

import pytest
import numpy as np
from stgiii_core.disclosure import DisclosureState


class TestDisclosureState:
    """DisclosureStateの単体テスト"""

    def test_initial_state(self) -> None:
        """初期状態のテスト"""
        state = DisclosureState(n_total=100)
        assert state.n_disclosed == 0
        assert len(state.disclosed_indices) == 0

    def test_disclose(self) -> None:
        """開示操作のテスト"""
        state = DisclosureState(n_total=100)
        new_count = state.disclose([0, 5, 10], [1.0, 2.0, 3.0])
        assert new_count == 3
        assert state.n_disclosed == 3
        assert state.is_disclosed(0)
        assert state.is_disclosed(5)
        assert not state.is_disclosed(1)

    def test_disclose_duplicates(self) -> None:
        """重複開示のテスト"""
        state = DisclosureState(n_total=100)
        state.disclose([0, 5], [1.0, 2.0])
        new_count = state.disclose([5, 10], [2.0, 3.0])
        assert new_count == 1  # 5は既開示なのでカウントされない
        assert state.n_disclosed == 3

    def test_get_undisclosed(self) -> None:
        """未開示セル取得のテスト"""
        state = DisclosureState(n_total=10)
        state.disclose([0, 5, 9], [1.0, 2.0, 3.0])
        undisclosed = state.get_undisclosed_indices()
        assert len(undisclosed) == 7
        assert 0 not in undisclosed
        assert 5 not in undisclosed
        assert 1 in undisclosed
```

```python
# tests/test_operators.py

import pytest
import numpy as np
from stgiii_core.config import SimulationConfig, SlotConfig, OperatorType
from stgiii_core.indexer import CellIndexer
from stgiii_core.disclosure import DisclosureState
from stgiii_core.operators.base import OperatorContext
from stgiii_core.operators.registry import get_operator


class TestOperators:
    """Operatorの単体テスト"""

    @pytest.fixture
    def simple_config(self) -> SimulationConfig:
        return SimulationConfig(
            operator_type=OperatorType.RANDOM,
            n_trials=10,
            slots=(
                SlotConfig("A", 10),
                SlotConfig("B", 10),
            ),
            main_effect_range=(-1.0, 1.0),
            error_clip_range=(-0.5, 0.5),
            k_per_step=1,
            topk_k=5,
        )

    def test_random_operator_no_duplicate_selection(
        self,
        simple_config: SimulationConfig
    ) -> None:
        """ランダム戦略が既開示セルを選択しないこと"""
        indexer = CellIndexer(simple_config.slot_sizes)
        rng = np.random.default_rng(42)
        context = OperatorContext(simple_config, indexer, rng)

        operator = get_operator(OperatorType.RANDOM, context)
        disclosure = DisclosureState(n_total=indexer.n_total)

        # 初期開示
        disclosure.disclose([0, 1, 2, 3, 4], [1.0] * 5)

        # 選択テスト（100回）
        for _ in range(100):
            selected = operator.select_next_cells(disclosure, 1)
            assert selected[0] not in [0, 1, 2, 3, 4]
            # 開示を進める
            disclosure.disclose(selected, [1.0])
```

### 9.2 統合テスト

```python
# tests/test_integration.py

import pytest
from stgiii_core.config import SimulationConfig, SlotConfig, OperatorType
from stgiii_core.simulation import SimulationEngine


class TestIntegration:
    """統合テスト"""

    @pytest.mark.parametrize("n_slots", [2, 3, 4])
    def test_simulation_completes(self, n_slots: int) -> None:
        """各スロット数でシミュレーションが完了すること"""
        slots = tuple(
            SlotConfig(name, 10)
            for name in ["A", "B", "C", "D"][:n_slots]
        )

        config = SimulationConfig(
            operator_type=OperatorType.RANDOM,
            n_trials=5,
            slots=slots,
            main_effect_range=(-1.0, 1.0),
            error_clip_range=(-0.5, 0.5),
            k_per_step=1,
            topk_k=5,
        )

        engine = SimulationEngine(config)
        results = engine.run()

        assert len(results.trials) == 5
        for trial in results.trials:
            assert trial.p_top1 > 0
            assert trial.p_topk > 0
            assert trial.p_topk <= trial.p_top1

    @pytest.mark.parametrize("operator_type", list(OperatorType))
    def test_all_operators(self, operator_type: OperatorType) -> None:
        """全Operatorが動作すること"""
        config = SimulationConfig(
            operator_type=operator_type,
            n_trials=3,
            slots=(
                SlotConfig("A", 10),
                SlotConfig("B", 10),
            ),
            main_effect_range=(-1.0, 1.0),
            error_clip_range=(-0.5, 0.5),
            k_per_step=1,
            topk_k=5,
        )

        engine = SimulationEngine(config)
        results = engine.run()

        assert len(results.trials) == 3
        df = results.to_dataframe()
        assert len(df) == 3
```

---

## 10. 今後の拡張ポイント

1. **新規Operator追加**: `@register_operator`デコレータで新戦略を追加可能
2. **複数手法の一括比較**: `SimulationEngine`を複数インスタンス化して並列実行
3. **結果の永続化**: SQLite/PostgreSQLへの保存機能
4. **設定プリセット**: YAML/JSONによる設定ファイルのサポート
5. **可視化の拡充**: インタラクティブなPlotlyグラフへの対応

---

## 付録: 設定パラメータ一覧

| パラメータ | 型 | 範囲/選択肢 | デフォルト | 説明 |
|-----------|-----|------------|-----------|------|
| operator_type | Enum | Random, FW-Ridge, Bayesian-FW-UCB | - | 探索戦略 |
| n_trials | int | 10〜1000 | 100 | 試行数 |
| n_slots | int | 2〜4 | 3 | スロット数 |
| n_building_blocks | int | 10〜50/スロット | 20 | 各スロットのBB数 |
| main_effect_range | (float, float) | - | (-1.0, 1.0) | 主作用の範囲 |
| error_clip_range | (float, float) | - | (-0.5, 0.5) | 誤差の範囲 |
| k_per_step | int | 1〜5 | 1 | 1ステップの開示数 |
| topk_k | int | 5, 10, 20 | 10 | Top-kのk値 |
| bias_range | (float, float) | - | (7.5, 8.5) | グローバルバイアス範囲 |
| slot_bias_range | (float, float) | - | (-0.5, 0.5) | スロットバイアス範囲 |
| ridge_alpha | float | - | 1.0 | Ridge正則化強度 |
| ucb_beta | float | - | 1.0 | UCBの探索係数 |
| sigma_min | float | - | 0.05 | σの下限 |
| obs_clip_range | (float, float) | - | (5.0, 11.0) | 観測値のclip範囲 |

---

以上
