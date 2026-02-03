# codex_bayesian_modify_plan_v1.0

Bayesian戦略改修プラン（v1.0）

---

## 0. 現状確認（固定αの根拠）
- `stgiii_core/config.py` の `SimulationConfig.ridge_alpha` がデフォルト `1.0` 固定。
- `stgiii_core/operators/bayesian_fw.py` では `self.alpha = self.config.ridge_alpha` として prior precision を固定使用。
- よって **Bayesian-FW-UCB の α（事前精度）は現状固定**。

---

## 1. 目的（v2.0の範囲）
1) ベイズ事前αを **経験ベイズで動的更新** できるようにする  
2) 既存の **UCB戦略は維持**  
3) 新規に **Thompson Sampling戦略** を追加  
4) これら完了時点を **v2.0** とする  
（Operator拡充は次フェーズ）

---

## 2. 変更方針（高レベル）
- **学習（Bayesianモデル）**は共通化し、**提案戦略のみ差分**にする  
  - UCB: `μ + βσ_param`  
  - TS: `θ ~ N(θ_MAP, Σ)` をサンプル → `μ = Xθ` で選択  
- α（事前精度）は **経験ベイズ**で更新

---

## 3. 経験ベイズ（α更新）設計方針
### 3.1 更新方式（採用）
**Evidence Maximization（ML-II）を採用。実装は最もシンプルな閉形式更新を使用。**

対象モデル（現行と一致）：
- 事前：`p(θ)=N(0, α^{-1}I)`  
- 尤度：`y|θ ~ N(Xθ, σ^2 I)`  
- 事後：`θ_MAP`, `Σθ = (X'X/σ^2 + αI)^{-1}`

**更新式（推奨）**：
- `γ = n_features - α * trace(Σθ)`  
- `α_new = γ / (θ_MAP^T θ_MAP)`  

※ 数値安定化のため `α_min ≤ α_new ≤ α_max` でクリップし、`θ_MAP^T θ_MAP` が小さすぎる場合は更新をスキップ。

### 3.2 実装上の制御パラメータ
新規パラメータ例：
- `alpha_min`, `alpha_max`（安定化のため）
  - **推奨値**: `alpha_min=1e-6`, `alpha_max=1e6`（広い範囲で数値不安定のみ抑制）

既存パラメータの扱い：
- `ridge_alpha` を **初期α** として使用（分離しない）

更新頻度：
- **fit毎に1回更新**（ユーザ指定）

### 3.3 実装位置
- `stgiii_core/operators/bayesian_fw.py` の `fit()` に α更新ループ追加  
- `SimulationConfig` に EB関連設定を追加  
- UI / CLI での表示は **Advanced** に限定（必要最小限）

---

## 4. Thompson Sampling戦略の追加
### 4.1 新規OperatorType
例：
- `BAYESIAN_FW_UCB`（既存）
- `BAYESIAN_FW_TS`（新規）

### 4.2 選択ロジック
- **TS**: `θ ~ N(θ_MAP, Σθ)` を **1回サンプル**  
  - `score = Xθ` を全セルに適用  
  - 上位Kを選択（同点はランダム）

※ TSは本来「1サンプル」が標準で、複数サンプルは探索/楽観性を変質させる。  
　性能向上が確実ではないため **v2.0では1回サンプルに固定**（計算コストも最小）。

### 4.3 依存コンポーネントの更新箇所
（実装時に適用）
- `stgiii_core/config.py`（OperatorType追加）
- `stgiii_core/operators/registry.py`（登録）
- `stgiii_core/operators/bayesian_fw.py`（UCB/TS共通化 or TS用クラス追加）
- `app/sidebar.py`（UIリスト追加）
- `tests/test_operators.py`、`tests/test_integration.py`（戦略追加に伴う更新）
- `stageIII_simulator_design_spec_v0_1.md`（設計仕様に反映）

---

## 5. 実装ステップ（順序）
1) **現状固定αの確認**（済）  
2) **EB仕様の確定**（更新式・頻度・制約）  
3) **Bayesian学習部の共通化**（UCB/TSで再利用）  
4) **TS戦略追加**（OperatorType + UI + registry）  
5) **テスト更新**（既存UCB保持 + TS動作確認）  
6) **v2.0完了宣言**  

---

## 6. 未確定事項（要確認）
なし（v2.0範囲の仕様は確定）

---

## 7. 期待する成果（v2.0）
- Bayesian-FW-UCB は現行挙動を維持  
- Bayesian-FW-TS を新戦略として選択可能  
- αが経験ベイズにより動的更新される  
- テストとUIが破綻せず動作  
