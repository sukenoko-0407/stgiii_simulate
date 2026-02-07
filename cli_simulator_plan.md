# CLI Simulator 実装プラン

## 1. 概要

WebApp版のシミュレーターをCLIから実行できる**完全に独立したモジュール**を作成する。
コマンドライン引数で設定を指定し、**すべての戦略（11種類）を自動的に試行**して結果を比較する。

**重要**: `cli_simulator/`ディレクトリは単体で動作する。必要なコアモジュール（`stgiii_core`）はすべてこのディレクトリ内にコピーする。

## 2. ディレクトリ構造

```
stgiii_simulate/
├── cli_simulator/                    # 新規作成（完全独立）
│   ├── __init__.py
│   ├── __main__.py                   # python -m cli_simulator で実行
│   ├── main.py                       # エントリーポイント・引数パース
│   ├── runner.py                     # 全戦略の実行ループ
│   ├── config_builder.py             # 引数からSimulationConfig構築
│   ├── reporter.py                   # 結果の比較表示・CSV出力
│   │
│   └── stgiii_core/                  # ← コアモジュールをここにコピー
│       ├── __init__.py
│       ├── config.py                 # ★ max_total_cells=20,000 に変更
│       ├── simulation.py
│       ├── matrix.py
│       ├── indexer.py
│       ├── disclosure.py
│       ├── results.py
│       ├── exceptions.py
│       └── operators/
│           ├── __init__.py
│           ├── base.py
│           ├── registry.py
│           ├── random_operator.py
│           ├── fw_ols.py
│           ├── fw_ridge.py
│           ├── bayesian_fw.py
│           └── fw_interactions.py
│
├── stgiii_core/                      # 既存（WebApp用、変更なし）
├── app/                              # 既存（WebApp用、変更なし）
└── ... (その他既存ファイル)
```

## 3. コピーするファイル一覧

### 3.1 stgiii_core/ からコピー

| ファイル | 変更 | 備考 |
|----------|------|------|
| `__init__.py` | なし | パッケージ定義 |
| `config.py` | **あり** | `max_total_cells: int = 20_000` に変更 |
| `simulation.py` | なし | シミュレーションエンジン |
| `matrix.py` | なし | 評価値マトリックス生成 |
| `indexer.py` | なし | セルインデックス管理 |
| `disclosure.py` | なし | 開示状態管理 |
| `results.py` | なし | 結果データ構造 |
| `exceptions.py` | なし | カスタム例外 |

### 3.2 stgiii_core/operators/ からコピー

| ファイル | 変更 | 備考 |
|----------|------|------|
| `__init__.py` | なし | Operator登録・エクスポート |
| `base.py` | なし | 基底クラス |
| `registry.py` | なし | プラグインレジストリ |
| `random_operator.py` | なし | ランダム戦略 |
| `fw_ols.py` | なし | Free-Wilson (OLS) |
| `fw_ridge.py` | なし | Free-Wilson (Ridge) |
| `bayesian_fw.py` | なし | Bayesian FW (UCB/TS) |
| `fw_interactions.py` | なし | Discrete/Continuous相互作用 |

## 4. 新規作成ファイル

| ファイル | 役割 |
|----------|------|
| `__init__.py` | パッケージ定義 |
| `__main__.py` | `python -m cli_simulator` エントリ |
| `main.py` | 引数パース、全体制御 |
| `runner.py` | 全戦略実行ループ |
| `config_builder.py` | 引数→SimulationConfig変換 |
| `reporter.py` | 結果比較表・CSV出力 |

## 5. コマンドライン引数設計

### 5.1 基本引数

| 引数 | 短縮 | 型 | デフォルト | 説明 |
|------|------|-----|-----------|------|
| `--slots` | `-s` | str | "20,20,20" | 各スロットのBB数（カンマ区切り） |
| `--trials` | `-t` | int | 100 | 試行回数 |
| `--k-per-step` | `-k` | int | 1 | 1ステップで開示するセル数 |
| `--seed` | | int | None | 乱数シード（再現性用） |
| `--output` | `-o` | str | None | 結果出力先CSVファイル |

### 5.2 難易度・生成モデル引数

| 引数 | 短縮 | 型 | デフォルト | 説明 |
|------|------|-----|-----------|------|
| `--preset` | `-p` | str | "balanced" | プリセット: easy/balanced/hard |
| `--f-main` | | float | None | main effectの分散比率（preset上書き） |
| `--f-int` | | float | None | interactionの分散比率（preset上書き） |
| `--f-res` | | float | None | residualの分散比率（preset上書き） |
| `--eta-spike` | | float | None | activity cliffの寄与度 |
| `--spike-hotspots` | | int | None | ホットスポット数 |
| `--residual-nu` | | float | None | 残差t分布の自由度 |
| `--distance-lambda` | | float | None | スロット距離スケール |

### 5.3 連続相互作用Operator用（Advanced）

| 引数 | 型 | デフォルト | 説明 |
|------|-----|-----------|------|
| `--operator-high-dim` | int | 256 | 高次元変換の出力次元 |
| `--operator-pca-dim` | int | 16 | PCA次元 |
| `--operator-mlp-hidden-dim` | int | 16 | MLP中間次元 |
| `--operator-nonlinearity` | str | "tanh" | 非線形関数 (tanh/gelu) |
| `--continuous-model` | str | "kron" | 連続相互作用モデル (kron/low_rank) |
| `--continuous-rank` | int | 4 | low-rank使用時のrank |

### 5.4 出力制御

| 引数 | 型 | デフォルト | 説明 |
|------|-----|-----------|------|
| `--verbose` | flag | False | 詳細出力 |
| `--no-progress` | flag | False | プログレスバー非表示 |
| `--format` | str | "table" | 出力形式: table/csv/json |

## 6. 実行フロー

```
1. 引数パース (main.py)
   ↓
2. 設定構築 (config_builder.py)
   - 引数 → SimulationConfig
   - セル数検証（20,000以下か確認）
   ↓
3. 全戦略実行ループ (runner.py)
   for strategy in ALL_STRATEGIES:
       config = replace(config, operator_type=strategy)
       engine = SimulationEngine(config)
       results = engine.run()
       store_results(strategy, results)
   ↓
4. 結果比較・出力 (reporter.py)
   - 全戦略の統計量を表形式で比較
   - CSVファイル出力（指定時）
```

## 7. 出力フォーマット

### 7.1 コンソール出力（表形式）

```
================================================================================
                    StageIII CLI Simulator - Results Comparison
================================================================================
Configuration:
  Slots: A(20) × B(20) × C(20) = 8,000 cells
  Trials: 100
  K per Step: 1
  Preset: balanced (f_main=0.35, f_int=0.35, f_res=0.30)
--------------------------------------------------------------------------------

Strategy Comparison (P_top1 - cells to find #1):
--------------------------------------------------------------------------------
Strategy                    | Median |  Mean  |  STD   |  Min   |  Max   |
--------------------------------------------------------------------------------
RANDOM                      |  4032  |  4128  |  2341  |   245  |  7891  |
FW_OLS                      |   892  |   956  |   412  |   123  |  2341  |
FW_RIDGE                    |   845  |   912  |   398  |   112  |  2156  |
BAYESIAN_FW_UCB             |   623  |   687  |   312  |    89  |  1823  |
BAYESIAN_FW_TS              |   598  |   654  |   298  |    85  |  1756  |
FW_OLS_DISCRETE             |   712  |   789  |   356  |   102  |  1987  |
FW_RIDGE_DISCRETE           |   678  |   745  |   334  |    98  |  1876  |
BAYESIAN_FW_DISCRETE        |   534  |   589  |   267  |    76  |  1534  |
FW_OLS_CONTINUOUS           |   689  |   756  |   345  |    95  |  1923  |
FW_RIDGE_CONTINUOUS         |   656  |   723  |   323  |    91  |  1845  |
BAYESIAN_FW_CONTINUOUS      |   512  |   567  |   256  |    72  |  1467  |
--------------------------------------------------------------------------------

Strategy Comparison (P_top100_50 - cells to find 50 of top-100):
--------------------------------------------------------------------------------
(同様の表形式)
--------------------------------------------------------------------------------

Best Strategy: BAYESIAN_FW_CONTINUOUS (Median P_top1: 512)
================================================================================
```

### 7.2 CSV出力

```csv
strategy,metric,median,mean,std,min,max
RANDOM,p_top1,4032,4128,2341,245,7891
RANDOM,p_top100_50,3245,3312,1876,198,6543
FW_OLS,p_top1,892,956,412,123,2341
...
```

## 8. 全戦略リスト（11種類）

1. `RANDOM` - ランダム選択（ベースライン）
2. `FW_OLS` - Free-Wilson + OLS
3. `FW_RIDGE` - Free-Wilson + Ridge回帰
4. `BAYESIAN_FW_UCB` - Bayesian FW + UCB
5. `BAYESIAN_FW_TS` - Bayesian FW + Thompson Sampling
6. `FW_OLS_DISCRETE` - FW + Discrete相互作用（OLS）
7. `FW_RIDGE_DISCRETE` - FW + Discrete相互作用（Ridge）
8. `BAYESIAN_FW_DISCRETE` - FW + Discrete相互作用（Bayesian）
9. `FW_OLS_CONTINUOUS` - FW + Continuous相互作用（OLS）
10. `FW_RIDGE_CONTINUOUS` - FW + Continuous相互作用（Ridge）
11. `BAYESIAN_FW_CONTINUOUS` - FW + Continuous相互作用（Bayesian）

## 9. 主要な変更点（WebApp版との差異）

| 項目 | WebApp | CLI |
|------|--------|-----|
| セル数上限 | 100,000 | **20,000** |
| 戦略選択 | UI で1つ選択 | **全戦略自動実行** |
| 設定入力 | Streamlit UI | コマンドライン引数 |
| 結果表示 | グラフ＋テーブル | 比較表（テキスト）|
| 出力 | ブラウザ表示 | stdout + CSV |
| 進捗表示 | Streamlit progress | tqdm（オプション）|
| コード配置 | stgiii_core参照 | **stgiii_coreを内包** |

## 10. 依存関係

### 10.1 内包するコアモジュール

`cli_simulator/stgiii_core/` にコピー：
- 完全に独立して動作するため、親ディレクトリへの依存なし

### 10.2 外部ライブラリ依存

コアモジュールが使用（environment.ymlで既に管理済み）：
- `numpy`
- `scipy`
- `scikit-learn`
- `pandas`

CLI固有：
- `argparse` - 引数パース（標準ライブラリ）
- `tqdm` - プログレスバー（オプション、なければスキップ）

## 11. 使用例

### 基本実行
```bash
cd stgiii_simulate
python -m cli_simulator
```

### スロット構成を指定
```bash
python -m cli_simulator --slots 25,30,20
```

### 難易度プリセットと試行回数
```bash
python -m cli_simulator --preset hard --trials 200
```

### カスタム設定
```bash
python -m cli_simulator \
  --slots 20,25,30,15 \
  --trials 50 \
  --f-main 0.4 \
  --f-int 0.3 \
  --eta-spike 0.25 \
  --output results.csv
```

### 再現性確保
```bash
python -m cli_simulator --seed 42 --trials 100
```

## 12. 実装順序

### Phase 1: ディレクトリ構造とコアコピー
- [ ] `cli_simulator/` ディレクトリ作成
- [ ] `cli_simulator/stgiii_core/` ディレクトリ作成
- [ ] `cli_simulator/stgiii_core/operators/` ディレクトリ作成
- [ ] 既存 `stgiii_core/` からファイルをコピー
- [ ] `config.py` の `max_total_cells` を `20_000` に変更

### Phase 2: CLI基本構造
- [ ] `cli_simulator/__init__.py` 作成
- [ ] `cli_simulator/__main__.py` 作成

### Phase 3: 設定構築
- [ ] `cli_simulator/config_builder.py` 実装
  - 引数 → SimulationConfig 変換
  - プリセット定義（easy/balanced/hard）

### Phase 4: 実行エンジン
- [ ] `cli_simulator/runner.py` 実装
  - 全戦略ループ
  - 進捗表示

### Phase 5: 結果出力
- [ ] `cli_simulator/reporter.py` 実装
  - 比較表生成
  - CSV出力

### Phase 6: メインエントリ
- [ ] `cli_simulator/main.py` 実装
  - argparse設定
  - 全体統合

### Phase 7: 動作確認
- [ ] 基本実行テスト
- [ ] 各引数オプションのテスト
- [ ] エラーハンドリング確認

## 13. config.py の変更箇所

```python
# 変更前（stgiii_core/config.py）
max_total_cells: int = field(default=100_000)

# 変更後（cli_simulator/stgiii_core/config.py）
max_total_cells: int = field(default=20_000)
```

この1箇所のみ変更。他のコードは完全にそのまま。

## 14. 注意事項

- 親ディレクトリの `stgiii_core/` は変更しない（WebApp用として維持）
- `cli_simulator/` は完全に独立して動作する
- 結果の再現性のため、シードを指定可能
- 11戦略 × 100試行 = 1,100回のシミュレーション（時間がかかる可能性）
- tqdmがなければプログレスバーなしで動作

---

*作成日: 2026-02-07*
*更新日: 2026-02-07*
*対象: stgiii_simulate CLI版*
