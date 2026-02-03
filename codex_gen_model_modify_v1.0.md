# codex_gen_model_modify_v1.0

StageIII Simulator 生成モデル変更案（v1.0）

---

## 0. 決定事項（v1.0実装方針・ユーザ回答反映）

本ドキュメントは、以下の意思決定を前提として記述する。

- **生成モデルはv1.0へ全面置換**（旧additiveモデル/旧パラメータへの互換性は重視しない）
- スロット距離 `D`：デフォルトは **直鎖距離（A-B-C-D）**、Advancedで任意行列入力を許可
- 相互作用スケール：`scale_{s,t} = exp(-D_{s,t}/λ)`、`λ` をUIに出す
- BB埋め込み：シリーズ混合（クラスタ中心＋微小ノイズ）、初期値は `d=16, K_s=ceil(N_s/5), σ_z=0.3`
- 主作用 `m_s[i]`：**一様分布で生成**（従来方針）  
  ただしユーザは `main_effect_range` のような絶対スケールを直接指定せず、**寄与比率（`f_main/f_int/f_res`）**で難易度を制御する
- 相互作用（smooth）：低ランク双線形 `z^T W z`、初期値 `r=4`
- 相互作用（spike）：ホットスポット和＋中心は実在BBペアからサンプル
- cliff UI：**H（ホットスポット数）をUIで**。鋭さ `ℓ` はUIで必須にしない（固定orAdvanced）。  
  `η_spike`（smoothとspikeの混合）は概念説明を併記し、UIラベルも分かりやすくする
- 残差：`t` 分布（`ν_res` をUIで指定）  
  なぜ `t` を使うかの理由を本文に明記する（後述）
- 出力レンジ：観測Range（`obs_low/obs_high`）は現状維持。  
  UIでパラメータを動かしてもレンジ外へ大きく逸脱しにくいよう、**最終段でμ/σを正規化**し、必要なら **clip率を監視**する
- 正解定義：`top1_index = argmax(y_latent)`（latent基準）
- Matrix保存：`y_main/y_int/y_res` と **寄与比率**を保存（デバッグ・可視化用途）
- UIプリセット：3プリセットを用意（現実寄りに難易度を切替）
- 検証手段：現行同様に **サンプルCSVをダウンロード**できるように実装する（新生成モデルの動作確認用）

---

目的：
- **現実の組み合わせ合成（combinatorial synthesis）段階**で観測される振る舞いに近い生成モデルを、シミュレータ側に実装できるレベルまで具体化する
- 特に以下を同時に再現する  
  1) 主作用（Free-Wilsonで説明可能な寄与）は存在する  
  2) 相互作用（slot×slotのペア由来の寄与）は存在する  
  3) それでも説明できない残差（Error）は存在する  
  4) **類似な置換基ペアは相互作用も似やすい**一方で、**activity cliff（突然変異的な跳ね）**も起こり得る

注意：
- 本提案は「ある特定戦略を勝たせる」ためではなく、**現実っぽい難しさ**（構造化された相互作用＋局所的cliff＋残差）を入れるためのもの。
- 戦略比較の公平性は後工程で設計する（ただし、主作用と相互作用が混ざらないよう **ANOVA分解に相当する制約（double-centering）**は、現実的にも“主作用/相互作用の分離”として自然なので先に入れる）。

---

## 1. 記法（Notation）

- スロット数：`S`（例：2〜4）
- スロット `s ∈ {1..S}` のBB数：`N_s`
- BBインデックス：`i_s ∈ {0..N_s-1}`
- 1つの化合物（セル）を `x = (i_1, i_2, ..., i_S)` と表す
- 出力（pIC50などの活性スコア）：
  - 潜在スコア（真値に相当）：`y_latent(x)`
  - 観測スコア：`y_obs(x)`（本v1.0では測定誤差は入れず、clipのみ）

---

## 2. 生成モデルの全体式（Main + Interaction + Residual）

### 2.1 ベースモデル

各セル `x=(i_1,...,i_S)` に対して：

`y_latent(x) = μ0 + Σ_s m_s[i_s] + Σ_{s<t} I_{s,t}[i_s, i_t] + ε_res(x)`

- `μ0`：グローバルバイアス（全体平均）
- `m_s[i]`：主作用（slot s の BB i の効果）
- `I_{s,t}[i,j]`：スロット対(s,t)の相互作用（BBペア(i,j)の追加寄与）
- `ε_res(x)`：残差（主作用・相互作用で説明できない誤差；heavy-tail推奨）

### 2.2 観測モデル（v1.0）

再測定は現実的でない／n=3で平均化されている前提より、ここでは**測定誤差を入れない**：

`y_obs(x) = clip(y_latent(x), obs_low, obs_high)`

（必要なら将来v1.xで `+ ε_meas` を導入可能だが、本仕様では採用しない）

---

## 3. 相互作用の設計方針（現実っぽさ重視）

要件：
- (R1) 似たBBペア `(a1,b1)` と `(a2,b2)` で、`a1≈a2` かつ `b1≈b2` なら相互作用も似るのは自然
- (R2) しかし「似ているから常に似た相互作用」だと **activity cliff** が出ない  
  → 類似性は効かせつつ、局所的に飛ぶ（cliff）成分を混ぜる

よって、相互作用は **滑らかな成分 + 特異点（spike）成分** の和で作る：

`I_{s,t} = scale_{s,t} * ( sqrt(1-η_spike) * Î_smooth_{s,t} + sqrt(η_spike) * Î_spike_{s,t} )`  
（最後にdouble-centeringして主作用と分離）

ここで：
- `η_spike ∈ [0,1]`：**cliff（特異点）の寄与比率ノブ**（0で完全に滑らか、1で特異点が支配）
  - UI上の意味（直感的な説明）：
    - `η_spike = 0`：相互作用は滑らかな成分のみ（activity cliffがほぼ出ない）
    - `η_spike = 0.5`：相互作用の“エネルギー（分散）”のうち半分程度がcliff由来（局所的な跳ねが増える）
    - `η_spike = 1`：相互作用はcliff成分が支配（近いのに急に変わる、が頻発しやすい）
  - 実装上の定義：
    - `Î_smooth` と `Î_spike` は、それぞれ **double-centering前に0平均化し、標準偏差を1に正規化**した相互作用テーブル  
      （これにより `η_spike` を“混合比”ではなく“寄与比率”として解釈しやすくする）
- `scale_{s,t}`：スロット距離に基づく相互作用スケール（ただし**“近い構造ほど値が滑らか”を強制しない**）

---

## 4. BBの潜在ベクトル（埋め込み）z の生成

### 4.1 目的

現実のBBは、化学的記述子（疎水性・電子・立体…）を高次元に持つ。  
Simulatorでは実記述子が無いので、**潜在ベクトル `z_s[i] ∈ R^d` を割り当てることで「似ている/似てない」を作る**。

### 4.2 推奨パラメータ

- 埋め込み次元 `d`：8〜32（推奨 16）
  - あまり大きいと距離が一様化しやすく「似ている」が効きにくくなるため、**“やや高次元”**に留める

### 4.3 生成方法（シリーズ＋微小変化：現実っぽさ重視）

BBを完全にiidにせず、**類縁体シリーズ（series）**を作る：

1) スロットsに対してシリーズ数 `K_s` を決める（例：`K_s = ceil(N_s / 5)`）
2) 各シリーズkの中心 `c_{s,k} ~ N(0, I_d)`
3) 各BB i にシリーズID `g_s[i] ∈ {1..K_s}` を割り当てる（均等orランダム）
4) BB埋め込み：
   - `z_s[i] = c_{s,g_s[i]} + δ_s[i]`
   - `δ_s[i] ~ N(0, σ_z^2 I_d)`（例：`σ_z = 0.3`）

この構造により：
- 同シリーズ内は「似ている」
- シリーズ間は「それなりに違う」
が自然に出る（現実のSARシリーズっぽい）。

---

## 5. スロット間距離と相互作用スケール（scale_{s,t}）

スロットは「近い/遠い」が想定できる前提：

- 任意の距離行列 `D` を用意：`D_{s,t} ≥ 0`、`D_{s,s}=0`
  - 例：A-B-C-Dの直鎖なら `D_{A,B}=1, D_{A,C}=2 ...`
  - 例：ポケット構造が分かれば、それに合わせてユーザが手で指定

相互作用スケールは距離の単調減衰でよい（値そのものの滑らかさは強制しない）：

`scale_{s,t} = exp( - D_{s,t} / λ )`

- `λ`：距離の効き具合（大きいほど遠距離にも相互作用が残る）
- 重要：ここは「起こりやすさ/大きさ」を調整するだけで、**“類似なら同じ相互作用になる”を距離で強化しない**。

---

## 6. 相互作用：滑らかな成分（I_smooth）

### 6.1 直感

似ているBB同士なら似た相互作用が出るようにする。  
最も簡単で制御しやすいのは双線形（bilinear）：

`I_smooth_{s,t}[i,j] = z_s[i]^T W_{s,t} z_t[j]`

### 6.2 W_{s,t} の生成（過度に滑らかにしない）

`W_{s,t}` をフル行列にすると自由度が大きすぎて“何でもあり”になりやすい。  
現実の「少数の潜在物性が効く」を反映して **低ランク** を推奨：

`W_{s,t} = U_{s,t} diag(w_{s,t}) V_{s,t}^T`

- `U_{s,t}, V_{s,t} ∈ R^{d×r}`：ランダム直交っぽい基底（例：乱数→QR）
- `w_{s,t} ∈ R^r`：強度（例：`w ~ N(0, σ_W^2)`）
- ランク `r`：2〜8（推奨 4）

これにより：
- 「似ていれば似る」は満たす
- しかし表現は限定され、過度に“何でも滑らか”になりすぎない

---

## 7. 相互作用：特異点（cliff）成分（I_spike）

### 7.1 直感

activity cliff は「局所的に、近傍と不連続に見えるほど跳ねる」現象。  
Simulatorでは、ペア空間上の **局所ホットスポット** として実装するのが自然：

- ほとんどのペアは0に近い
- 一部の近傍にだけ大きい寄与が入る（符号も正負あり得る）

### 7.2 生成式（ホットスポットの和）

スロット対(s,t)ごとにホットスポット数 `H_{s,t}` を決める（例：1〜5）。

ホットスポット k（k=1..H_{s,t}）について：
- 中心：
  - `c^A_k ∈ R^d`, `c^B_k ∈ R^d`  
  - 生成方法：
    - 方法1（推奨）：実在BBから選ぶ → `c^A_k = z_s[i_k]`, `c^B_k = z_t[j_k]`（“特定のBBペアで事故が起きる”）
    - 方法2：連続空間からサンプル → `c ~ N(0, I_d)`
- 振幅（heavy-tail推奨）：
  - `a_k ~ t_ν(0, σ_spike)`（例：`ν=3〜8`）
- 幅（局所性）：
  - `ℓ`（例：0.2〜0.6）

BBペア(i,j)への寄与：

`I_spike_{s,t}[i,j] += a_k * exp( -||z_s[i]-c^A_k||^2 / (2ℓ^2) ) * exp( -||z_t[j]-c^B_k||^2 / (2ℓ^2) )`

この構造で：
- 中心近傍だけが強く変わる（cliff）
- しかも中心がBB埋め込み上にあるため、完全なセル単位ランダムより現実に近い

### 7.3 “似ているのに跳ねる”の制御

- `ℓ` を小さく → cliffが鋭くなる（局所だけ大きく飛ぶ）
- `σ_spike` を大きく、`ν` を小さく → 大当たり/大外れが増える
- `H_{s,t}` を増やす → cliff領域が増える（ただし多すぎると“世界中にcliff”になるので注意）

---

## 8. 主作用（m_s）の生成

主作用は Free-Wilson で説明可能な寄与として生成する。  
ここは現実のプロジェクトによって様々だが、少なくとも「スロットごとにBBの傾向がある」は自然。

推奨例：
- まず `raw_m_s[i] ~ Uniform(-1, 1)` を生成（従来方針：一様分布）
- スロットごとにスケールを持たせてもよい（slot biasや“このスロットは効きやすい”）
- あるいは既存設定に合わせ、`Uniform(main_low, main_high)` を生成してもよい

重要：
- 本v1.0では、ユーザは `main_effect_range` のような“絶対スケール”をUIで直接指定しない。  
  代わりに後段の **寄与比率（`f_main/f_int/f_res`）** によって、「主作用がどの程度支配的か」を制御する。
- よって、ここでの `raw_m_s` は“形”を作るための乱数であり、最終スケールは 11章の正規化で決まる。

---

## 9. 残差（ε_res）：主作用でも相互作用でもないもの

残差は「モデル化できない要因の総体」。  
現実では外れ値が出るので heavy-tail を推奨：

- `ε_res(x) ~ t_ν(0, σ_res)`（例：`ν=3〜10`）
- 必要なら極端値を抑えるために clip してもよい（ただし分布の裾が変わる）

補足：なぜ `t` 分布（heavy-tail）を採用するのか
- 組み合わせ合成では、説明しきれない要因（微小な合成差、測定系の揺らぎ、未知の高次相互作用、実験バッチ差など）が残り、**“たまに大きく外れる”**現象が起きる。
- 正規分布（Gaussian）は外れ値を過小に見積もりやすく、Simulatorが「素直すぎる世界」になりやすい。
- `t` 分布は「普段は小さいが、たまに大きい」を自然に表現でき、**activity cliffとは別枠の“説明不能外れ”**を入れやすい。  
  （cliffは系統成分として `I_spike` に入れ、残差は“どうしても説明できないもの”として `ε_res` に残す、という役割分担を明確にできる）

ここは「相互作用を間引く」より現実に近いことが多い：
- 相互作用は系統成分（学習可能な部分）として残す
- それでも説明できない“真の誤差”は残差に入れる

---

## 10. 主作用・相互作用の分離（double-centering）

相互作用テーブル `I_{s,t}` をそのまま使うと、行・列方向の平均成分が主作用と混ざる。

現実的にも「主作用として説明される分」は主作用へ、相互作用は純粋な相互作用へ、という分解が自然なので、各(s,t)で次を行う：

`I <- I - row_mean(I) - col_mean(I) + grand_mean(I)`

（行平均：各j固定で平均、列平均：各i固定で平均）

この処理により：
- `Σ_i I[i,j] = 0`（全j）
- `Σ_j I[i,j] = 0`（全i）
が満たされ、主作用と相互作用が混ざりにくくなる。

---

## 11. スケーリング：寄与比率の“現実っぽさ”を作る最重要工程

同じ構造でも、寄与の比率が現実とズレると難易度が激変する。  
そのため、生成後に **全セル上の標準偏差** で各成分を正規化し、重みを掛けて合成するのが安定。

### 11.1 分解成分の定義

全セルに対して以下を計算する（ベクトルとして保持）：
- `y_main(x) = Σ_s m_s[i_s]`
- `y_int(x) = Σ_{s<t} I_{s,t}[i_s,i_t]`（double-centering後）
- `y_res(x) = ε_res(x)`

### 11.2 標準化して重み付け合成

`ŷ_main = (y_main - mean)/std`（stdが0なら0扱い）  
`ŷ_int  = (y_int  - mean)/std`  
`ŷ_res  = (y_res  - mean)/std`

`y_latent_raw = w_main*ŷ_main + w_int*ŷ_int + w_res*ŷ_res`

ここで `w_main, w_int, w_res` で「主作用が限定的／残差が大きい」を直接指定できる。

推奨（“難しめ現実寄り”の一例）：
- `w_main = 0.5`
- `w_int  = 0.8`（相互作用がそこそこ）
- `w_res  = 1.5`（残差が支配的）

補足（UIからの制御を意識した表現）：
- `w_main, w_int, w_res` は「成分の寄与比率（=難易度）」を最も直感的に動かせるノブである。
- UIでは重みそのものではなく、以下のような**分散比率（fraction）**で指定してもよい：
  - `f_main + f_int + f_res = 1`（各成分の“狙い分散比率”）
  - 本モデルは `ŷ_main, ŷ_int, ŷ_res` を全セル上で標準化してから合成するため、まずは近似的に
    - `w_main = sqrt(f_main)`
    - `w_int  = sqrt(f_int)`
    - `w_res  = sqrt(f_res)`
    と置けば、概ね意図した比率で合成される（成分間の共分散が小さい設計のため）。
  - 実装では生成後に実測の分散比率を再計算し、UIに表示して確認できるようにする（ズレが大きい場合は警告 or 自動微調整）。

### 11.3 pIC50スケールへの写像（obs_rangeと整合）

最終的に、目標平均 `μ_target` と目標標準偏差 `σ_target` に合わせる：

`y_latent = μ_target + (y_latent_raw - mean(y_latent_raw)) * (σ_target / std(y_latent_raw))`

推奨：
- `μ_target = (obs_low + obs_high)/2`
- `σ_target ≈ (obs_high - obs_low)/6`（±3σが概ね範囲内）

最後に `y_obs = clip(y_latent, obs_low, obs_high)`。

---

## 12. 生成手順（アルゴリズム：疑似コード）

```text
Input:
  slot_sizes N_s, obs range, slot distance matrix D
  embedding dim d, interaction rank r
  spike params (η_spike, H_{s,t}, nu_spike, sigma_spike, ell)
  residual params (nu_res)  # strength is controlled by f_res / w_res (variance fraction)
  weights (w_main, w_int, w_res)
  target mean/stdev (mu_target, sigma_target)

1) Sample global bias mu0 (or fixed)
2) For each slot s:
     generate BB embeddings z_s[i] via mixture-of-series (cluster centers + noise)
     generate main effects m_s[i] (Normal/Uniform)
3) For each slot pair (s,t):
     scale_{s,t} = exp(-D_{s,t}/lambda)
     generate W_{s,t} (low-rank)
     build I_smooth[i,j] = z_s[i]^T W z_t[j]
     build I_spike via H hotspots on embedding space
     center+standardize I_smooth -> Î_smooth (0 mean, unit std)
     center+standardize I_spike  -> Î_spike  (0 mean, unit std)
     I = scale * ( sqrt(1-η_spike)*Î_smooth + sqrt(η_spike)*Î_spike )
     I = double_center(I)
4) For every cell x=(i_1..i_S):
     y_main(x) = sum_s m_s[i_s]
     y_int(x)  = sum_{s<t} I_{s,t}[i_s,i_t]
     y_res(x)  = sample t_nu_res(0,1)  # then scaling step
5) Standardize y_main, y_int, y_res over all cells; combine with weights:
     y_latent_raw = w_main*ŷ_main + w_int*ŷ_int + w_res*ŷ_res
6) Affine transform y_latent_raw to match (mu_target, sigma_target):
     y_latent = ...
7) y_obs = clip(y_latent, obs_range)
8) top1_index = argmax(y_latent)  # uniqueness check as needed
```

---

## 13. “現実っぽさ”のチェック項目（実装後に必ず確認）

生成モデルが狙い通りかを確認するため、以下をログ/可視化できると良い。

- (C1) 寄与分解の分散比率：`Var(main) : Var(int) : Var(res)` が意図通りか
- (C2) 相互作用の局所性：embedding上で近いペアほど `I_smooth` が近いか
- (C3) cliff：近傍なのに大差が出る点が存在するか（spikeの効果）
- (C4) slot距離の効果：遠いスロット対では相互作用スケールが下がっているか
- (C5) “学習可能な構造”があるか：ランダムセルより、主作用推定で説明できる成分が残っているか（w_mainがゼロに近いと学習不能になる）

---

## 14. 実装メモ（StageIII Simulatorへの反映イメージ）

現行 `MatrixGenerator._generate_single()` は
- main_effect（slotごとのBB効果）
- error（セルごと独立）
だけで `y_true` を作っている。

本v1.0では追加で以下を生成・保持するとよい：
- `bb_embeddings`: slotごとの `z_s[i]`
- `interaction_tables`: 各(s,t)の `I_{s,t}`（double-centering後）
- `components`（任意）：`y_main`, `y_int`, `y_res`（デバッグ・分析用）
- `variance_fractions`（推奨）：`VarFrac(main/int/res)` と、可能なら `VarFrac(int_smooth/int_spike)`

Matrix構造体に「主作用」「相互作用」の分解を保存するかは任意だが、検証のためには保存推奨。

---

## 15. 難易度（Difficulty）の制御方針（UIから調整できる形）

ここでいう「難易度」は、少なくとも以下の2つを含む：
- **(D1) どれだけ“学習可能な構造”があるか**（主作用・相互作用が残差に埋もれていないか）
- **(D2) 構造があっても“素直に一般化できるか”**（cliffの頻度・鋭さ、heavy-tailの外れやすさ、clipで情報が潰れる度合い）

本v1.0生成モデルでは、難易度は主に次のノブで制御する。

### 15.1 UIで最低限出したいノブ（推奨）

**A. 寄与比率（最重要）**
- `f_main, f_int, f_res`（分散比率、合計=1）  
  - `f_main` が大きいほど「主作用で説明できる割合」が増え、一般に難易度は下がる
  - `f_res` が大きいほど「説明不能な残差」が増え、一般に難易度は上がる
  - `f_int` は「主作用だけでは足りない度合い」（モデルミスマッチの強さ）を作る
- 実装は `w = sqrt(f)` を初期値として利用し、合成後の実測比率も表示する（11.2参照）

**B. cliff強度（特異点の出方）**
- `H`：ホットスポット数（UIで理解しやすい概念、1〜5程度が現実的）  
- `η_spike`（0〜1）：相互作用のうち **cliff由来が占める“寄与比率”**  
  - UIラベル例：`Cliff Contribution (0=smooth, 1=cliff-dominant)`
  - `η_spike` は“混合比”ではなく、3章の定義通り **I_smooth/I_spikeを正規化した上で分散寄与を割り当てるノブ**である
- `ℓ`：ホットスポットの幅（小さいほど局所的・cliffが鋭い）
  - 本方針では **UI必須にしない**（固定デフォルト or Advanced）
- `σ_spike, ν_spike`：スパイク振幅のスケールとheavy-tail（νが小さいほど極端値）

**C. 残差の“外れやすさ”**
- `ν_res`（t分布の自由度）：小さいほど外れやすい（外れ値が現実にあるなら重要）
  - `w_res`（または `f_res`）は残差の“量”、`ν_res` は残差の“質（外れ方）”を担当

（これら3つが揃えば、UIから「主作用支配で易しい」「cliffが多くて難しい」「残差が重くて難しい」を直感的に作れる）

### 15.2 “自動で破綻しない”ための補助ノブ（必要ならUIに追加）

**D. クリップ率（情報損失）**
観測Range（`obs_low/obs_high`）は固定のまま、**パラメータを動かしてもレンジ外へ大きく逸脱しにくい**ことを優先する。

推奨実装：
- 生成後に `y_latent_raw` を `μ_target, σ_target` へ正規化し、最後に `clip` する（11.3）
- `σ_target` はデフォルトで `(obs_high-obs_low)/6` とする（±3σが概ねレンジ内）
- ただし heavy-tail（spike/residual）が強いと clip が増えるため、**生成後にclip率を計測し、過剰ならσ_targetを下げる**  
  - 例：`clip_rate_max = 1%` を上限に、`σ_target` を二分探索で調整
  - UIが何をどう動かしても「レンジ外が大量に出る」状態を避けられる
- UIには「Clip Rate（上下限に張り付く割合）」を表示して、情報損失を可視化する

**E. 相互作用の“滑らかさ”側の複雑度**
- `d`（埋め込み次元）、`r`（低ランク相互作用のランク）、`σ_z`（シリーズ内のばらつき）
  - ここは難易度というより「現象の複雑さ」を動かすノブ  
  - 通常はプリセット固定でよい（UIのAdvancedに逃がすのが無難）

### 15.3 UIスライダー → 内部パラメータのマッピング例（実装方針）

UIを「簡単モード」で運用するなら、ユーザが触るスライダーを2〜3本に圧縮できる：

- `Signal (Main ↔ Residual)`（0〜1）
  - 例：`f_res = lerp(0.2, 0.7, slider)`、`f_main = 1 - f_res - f_int`（`f_int`は別スライダー or 固定）
- `Interaction Strength`（0〜1）
  - 例：`f_int = lerp(0.0, 0.4, slider)`（主作用/残差とのトレードオフ）
- `Cliffiness`（0〜1）
  - 例：`η_spike = slider`
  - `H = round(lerp(1, 5, slider))`
  - `ℓ` は固定（or Advanced）とし、UIは理解しやすい `H` と `η_spike` を中心にする
  - `ν_spike/σ_spike` も固定（or Advanced）に逃がせる

（UIの簡単モードではこのくらいで十分に難易度を動かせる。Advancedで各パラメータを直接触れるようにしてもよい。）

### 15.4 UIに必ず表示したい“難易度サマリ”（生成後に自動計算）

パラメータは直感的でも、実際の出力が狙い通りとは限らないため、生成後に以下を表示する：

- **寄与分解（実測）**：`VarFrac(main)`, `VarFrac(int)`, `VarFrac(res)`（合成前の標準化済み成分から算出）
- **spike寄与（実測）**：`VarFrac(int_smooth)` vs `VarFrac(int_spike)`（可能なら）
- **Clip率**：`P(y_obs == obs_low or obs_high)`（情報損失の目安）
- **Cliff指標（簡易）**：embedding近傍（距離が小さいBBペア）と遠方で、`|I[i,j]-I[i',j']|` の分位点（例：95%）がどれだけ違うか

---

## 16. サンプルCSV（検証用ダウンロード）の仕様（推奨）

目的：
- 生成モデルが意図通り動作しているかを、UIから即座に確認できるようにする
- 特に「寄与比率」「cliff」「clip率」が想定通りかを、CSVの列から検証できるようにする

推奨CSV内容：
- **ヘッダー（コメント行）**：
  - Seed、slot_sizes、`d/r/σ_z`、`λ`、`H`、`η_spike`、`ν_res`、`f_main/f_int/f_res`、`clip_rate`
  - 実測 `VarFrac(main/int/res)`、可能なら `VarFrac(int_smooth/int_spike)`
- **データ列（少なくとも）**：
  - `CellIndex`, 各スロットのBB index（`Slot_A`…）
  - `y_main`, `y_int`, `y_res`, `y_latent`, `y_obs`
  - 任意：`is_top1_latent`、`rank_latent`、`in_topk_latent`

注意：
- 全セルをCSVに出すと大きくなるため、UI上は「全セル」か「先頭N行」かを選べると便利（後回しでもよい）。

これがUIに出れば、ユーザは「難しくしすぎた/簡単すぎた」を定量で調整できる。
