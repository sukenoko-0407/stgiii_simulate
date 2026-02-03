"""
Sidebar component for Streamlit UI
"""

from typing import Tuple
import io
import numpy as np
import streamlit as st

from stgiii_core.config import SimulationConfig, SlotConfig, OperatorType, InitialDisclosureType
from stgiii_core.indexer import CellIndexer
from stgiii_core.matrix import MatrixGenerator


def render_sidebar() -> Tuple[SimulationConfig | None, bool]:
    """
    サイドバーをレンダリング

    Returns:
        (設定オブジェクト or None, 実行ボタンが押されたか)
    """
    st.sidebar.header("Simulation Settings")

    # 手法選択
    operator_options = {
        "Random": OperatorType.RANDOM,
        "Free-Wilson (OLS)": OperatorType.FW_OLS,
        "Free-Wilson (Ridge)": OperatorType.FW_RIDGE,
        "Bayesian Free-Wilson (UCB)": OperatorType.BAYESIAN_FW_UCB,
        "Bayesian Free-Wilson (TS)": OperatorType.BAYESIAN_FW_TS,
    }
    operator_name = st.sidebar.selectbox(
        "Operator (Search Strategy)",
        options=list(operator_options.keys()),
        index=3,  # デフォルトはBayesian-FW-UCB
        help="Select the search strategy to evaluate"
    )
    operator_type = operator_options[operator_name]

    # 試行数
    n_trials = st.sidebar.number_input(
        "Number of Trials",
        min_value=10,
        max_value=1000,
        value=100,
        step=10,
        help="Number of simulation runs"
    )

    # スロット数
    n_slots = st.sidebar.selectbox(
        "Number of Slots",
        options=[2, 3, 4],
        index=1,  # デフォルト3
        help="Number of building block categories (A, B, C, D)"
    )

    # 各スロットのBB数
    st.sidebar.subheader("Building Blocks per Slot")
    slot_names = ["A", "B", "C", "D"][:n_slots]
    slots: list[SlotConfig] = []

    for name in slot_names:
        n_bb = st.sidebar.slider(
            f"Slot {name}",
            min_value=10,
            max_value=50,
            value=20,
            help=f"Number of building blocks in slot {name}"
        )
        slots.append(SlotConfig(name=name, n_building_blocks=n_bb))

    # 総セル数の計算と表示
    n_total = 1
    for s in slots:
        n_total *= s.n_building_blocks

    # スロットサイズを保存
    slot_sizes = tuple(s.n_building_blocks for s in slots)

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Total Cells**: {n_total:,}")

    # 制限チェック
    max_cells = 100_000
    if n_total > max_cells:
        st.sidebar.error(
            f"Total cells exceed limit ({max_cells:,}). "
            "Please reduce the number of building blocks."
        )
        return None, False

    st.sidebar.markdown("---")

    # 難易度プリセット（3つ）
    st.sidebar.subheader("Difficulty Preset")
    preset_name = st.sidebar.selectbox(
        "Preset",
        options=[
            "Easy (Main-dominant)",
            "Balanced",
            "Hard (Residual-dominant)",
        ],
        index=1,
        help="Start from a preset, then fine-tune below"
    )

    if preset_name == "Easy (Main-dominant)":
        preset = {
            "f_main": 0.60,
            "f_int": 0.20,
            "eta_spike": 0.15,
            "spike_hotspots": 1,
            "residual_nu": 10.0,
            "distance_lambda": 1.0,
        }
    elif preset_name == "Hard (Residual-dominant)":
        preset = {
            "f_main": 0.20,
            "f_int": 0.30,
            "eta_spike": 0.50,
            "spike_hotspots": 3,
            "residual_nu": 4.0,
            "distance_lambda": 1.0,
        }
    else:  # Balanced
        preset = {
            "f_main": 0.35,
            "f_int": 0.35,
            "eta_spike": 0.30,
            "spike_hotspots": 2,
            "residual_nu": 6.0,
            "distance_lambda": 1.0,
        }

    st.sidebar.subheader("Signal / Interaction / Residual (Variance Fractions)")
    f_main = st.sidebar.slider(
        "f_main (Main fraction)",
        min_value=0.0,
        max_value=1.0,
        value=float(preset["f_main"]),
        step=0.05,
        help="Variance fraction explained by main effects (Free-Wilson)"
    )
    f_int = st.sidebar.slider(
        "f_int (Interaction fraction)",
        min_value=0.0,
        max_value=float(max(0.0, 1.0 - f_main)),
        value=float(min(preset["f_int"], 1.0 - f_main)),
        step=0.05,
        help="Variance fraction explained by pairwise interactions"
    )
    f_res = float(max(0.0, 1.0 - f_main - f_int))
    st.sidebar.caption(f"f_res (Residual fraction) = {f_res:.2f}")

    st.sidebar.subheader("Cliffs (Interaction Spikes)")
    spike_hotspots = st.sidebar.slider(
        "Hotspots per slot-pair (H)",
        min_value=0,
        max_value=5,
        value=int(preset["spike_hotspots"]),
        step=1,
        help="Number of local hotspots (activity cliffs) for each slot-pair"
    )
    eta_spike = st.sidebar.slider(
        "Cliff contribution (η_spike)",
        min_value=0.0,
        max_value=1.0,
        value=float(preset["eta_spike"]),
        step=0.05,
        help=(
            "Controls how much of the interaction term comes from cliffs (spikes). "
            "η=0: smooth-only interactions. η=1: cliff-dominant interactions."
        )
    )

    st.sidebar.subheader("Residual Heavy-Tail")
    residual_nu = st.sidebar.slider(
        "Residual ν (t-distribution)",
        min_value=3.0,
        max_value=30.0,
        value=float(preset["residual_nu"]),
        step=1.0,
        help="Smaller ν -> more outliers (harder). Must be > 2 for finite variance."
    )

    st.sidebar.subheader("Slot Distance Scale")
    distance_lambda = st.sidebar.slider(
        "Distance λ (scale=exp(-D/λ))",
        min_value=0.2,
        max_value=5.0,
        value=float(preset["distance_lambda"]),
        step=0.1,
        help="Larger λ makes long-range slot interactions stronger"
    )

    # Advanced: custom slot distance matrix
    st.sidebar.subheader("Advanced: Slot Distance Matrix")
    use_custom_d = st.sidebar.checkbox(
        "Use custom distance matrix",
        value=False,
        help="Provide an n_slots x n_slots non-negative matrix (diagonal must be 0)"
    )
    slot_distance_matrix = None
    if use_custom_d:
        default_d = "\n".join(
            ",".join(str(abs(i - j)) for j in range(n_slots))
            for i in range(n_slots)
        )
        text = st.sidebar.text_area(
            "D matrix (CSV rows)",
            value=default_d,
            help="Example (3 slots):\n0,1,2\n1,0,1\n2,1,0"
        )
        try:
            rows = []
            for line in text.strip().splitlines():
                if not line.strip():
                    continue
                rows.append([float(x.strip()) for x in line.split(",") if x.strip() != ""])
            if len(rows) != n_slots or any(len(r) != n_slots for r in rows):
                raise ValueError("Matrix shape mismatch")
            if any(rows[i][i] != 0 for i in range(n_slots)):
                raise ValueError("Diagonal must be 0")
            if any(rows[i][j] < 0 for i in range(n_slots) for j in range(n_slots)):
                raise ValueError("Matrix must be non-negative")
            slot_distance_matrix = tuple(tuple(r) for r in rows)
        except Exception:
            st.sidebar.error("Invalid distance matrix. Please provide a valid n_slots x n_slots matrix.")
            return None, False

    # サンプルデータのダウンロード
    st.sidebar.subheader("Sample Data Preview")
    sample_seed = st.sidebar.number_input(
        "Sample Seed",
        min_value=0,
        max_value=2**31 - 1,
        value=12345,
        step=1,
        help="Seed for generating sample data"
    )

    # サンプルデータを生成してダウンロードボタンを表示
    sample_csv = _generate_sample_data(
        operator_type=operator_type,
        slots=slots,
        f_main=float(f_main),
        f_int=float(f_int),
        f_res=float(f_res),
        eta_spike=float(eta_spike),
        spike_hotspots=int(spike_hotspots),
        residual_nu=float(residual_nu),
        distance_lambda=float(distance_lambda),
        slot_distance_matrix=slot_distance_matrix,
        seed=int(sample_seed),
    )
    st.sidebar.download_button(
        label="Download Sample Data",
        data=sample_csv,
        file_name="sample_matrix_data.csv",
        mime="text/csv",
        use_container_width=True,
        help="Download a sample matrix with current settings"
    )

    st.sidebar.markdown("---")

    # 初期開示方式（None固定）
    initial_disclosure_type = InitialDisclosureType.NONE
    st.sidebar.caption("Initial Disclosure: 0 cells (fixed: None)")

    # 1ステップで開示するセル数K
    k_per_step = st.sidebar.selectbox(
        "Cells per Step (K)",
        options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        index=0,
        help="Number of cells to disclose in each iteration"
    )

    # Top-k の k（100固定）
    topk_k = 100
    st.sidebar.caption("Top-k (k value): 100 (fixed)")

    # Random seed（オプション）
    use_seed = st.sidebar.checkbox(
        "Use fixed random seed",
        value=False,
        help="Enable for reproducible results"
    )
    random_seed: int | None = None
    if use_seed:
        random_seed = st.sidebar.number_input(
            "Random Seed",
            min_value=0,
            max_value=2**31 - 1,
            value=42,
            step=1
        )

    st.sidebar.markdown("---")

    # 実行ボタン
    run_clicked = st.sidebar.button(
        "Run Simulation",
        type="primary",
        use_container_width=True
    )

    if run_clicked:
        try:
            config = SimulationConfig(
                operator_type=operator_type,
                n_trials=int(n_trials),
                slots=tuple(slots),
                k_per_step=int(k_per_step),
                topk_k=int(topk_k),
                initial_disclosure_type=initial_disclosure_type,
                random_seed=random_seed,
                f_main=float(f_main),
                f_int=float(f_int),
                f_res=float(f_res),
                eta_spike=float(eta_spike),
                spike_hotspots=int(spike_hotspots),
                residual_nu=float(residual_nu),
                distance_lambda=float(distance_lambda),
                slot_distance_matrix=slot_distance_matrix,
            )
            return config, True
        except ValueError as e:
            st.sidebar.error(f"Configuration Error: {e}")
            return None, False

    return None, False


def _generate_sample_data(
    operator_type: OperatorType,
    slots: list[SlotConfig],
    f_main: float,
    f_int: float,
    f_res: float,
    eta_spike: float,
    spike_hotspots: int,
    residual_nu: float,
    distance_lambda: float,
    slot_distance_matrix: Tuple[Tuple[float, ...], ...] | None,
    seed: int
) -> str:
    """
    サンプルデータを生成してCSV形式で返す

    Args:
        operator_type: operator（設定用、サンプル生成自体には直接影響しない）
        slots: スロット設定リスト
        f_main/f_int/f_res: 分散寄与比率
        eta_spike: cliff寄与
        spike_hotspots: ホットスポット数（各スロット対）
        residual_nu: 残差t分布の自由度
        distance_lambda: スロット距離スケール
        slot_distance_matrix: Advanced距離行列（未指定なら直鎖）
        seed: 乱数シード

    Returns:
        CSV形式の文字列
    """
    rng = np.random.default_rng(seed)

    slot_sizes = tuple(s.n_building_blocks for s in slots)
    indexer = CellIndexer(slot_sizes)
    config = SimulationConfig(
        operator_type=operator_type,
        n_trials=1,
        slots=tuple(slots),
        k_per_step=1,
        topk_k=100,
        random_seed=None,
        f_main=f_main,
        f_int=f_int,
        f_res=f_res,
        eta_spike=eta_spike,
        spike_hotspots=spike_hotspots,
        residual_nu=residual_nu,
        distance_lambda=distance_lambda,
        slot_distance_matrix=slot_distance_matrix,
    )

    generator = MatrixGenerator(config, indexer, rng)
    matrix = generator.generate()
    n_total = indexer.n_total
    all_indices = np.arange(n_total, dtype=np.int64)
    all_coords = indexer.batch_indices_to_coords(all_indices)

    # CSVの作成
    output = io.StringIO()

    # ヘッダー情報
    output.write(f"# Seed: {seed}\n")
    output.write(f"# Global Bias (mu_target): {config.global_bias:.6f}\n")
    output.write(f"# Slot Sizes: {slot_sizes}\n")
    output.write(f"# Embedding: d={config.embedding_dim}, series_divisor={config.embedding_series_divisor}, sigma_z={config.embedding_sigma}\n")
    output.write(f"# Interaction: rank={config.interaction_rank}, lambda={config.distance_lambda}\n")
    output.write(f"# Cliffs: eta_spike={config.eta_spike}, hotspots={config.spike_hotspots}, nu_spike={config.spike_nu}, sigma_spike={config.spike_sigma}, ell={config.spike_ell}\n")
    output.write(f"# Residual: nu_res={config.residual_nu}\n")
    output.write(f"# Fractions: f_main={config.f_main}, f_int={config.f_int}, f_res={config.f_res}\n")
    output.write(f"# Clip Rate: {matrix.clip_rate:.6f}\n")
    output.write(f"# VarFrac: main={matrix.variance_fractions.get('main', 0.0):.6f}, int={matrix.variance_fractions.get('int', 0.0):.6f}, res={matrix.variance_fractions.get('res', 0.0):.6f}\n")
    output.write(f"# VarFracInt: smooth={matrix.variance_fractions.get('int_smooth', 0.0):.6f}, spike={matrix.variance_fractions.get('int_spike', 0.0):.6f}\n")
    output.write("#\n")

    # Main Effect値の出力
    output.write("# Main Effects per Slot/BB:\n")
    for slot_idx, slot_config in enumerate(slots):
        output.write(f"# Slot {slot_config.name}: ")
        output.write(", ".join(f"{v:.4f}" for v in matrix.main_effects[slot_idx]))
        output.write("\n")
    output.write("#\n")

    # セルデータのヘッダー
    slot_names = [s.name for s in slots]
    header_cols = ["CellIndex"] + [f"Slot_{name}" for name in slot_names]
    header_cols += [
        "y_main",
        "y_int",
        "y_int_smooth",
        "y_int_spike",
        "y_res",
        "y_latent",
        "y_obs",
        "is_top1_latent",
    ]
    output.write(",".join(header_cols) + "\n")

    # セルデータの出力
    for idx in range(n_total):
        coords = all_coords[idx]
        row = [str(idx)]
        row += [str(c) for c in coords]
        row.append(f"{matrix.y_main[idx]:.6f}")
        row.append(f"{matrix.y_int[idx]:.6f}")
        row.append(f"{matrix.y_int_smooth[idx]:.6f}")
        row.append(f"{matrix.y_int_spike[idx]:.6f}")
        row.append(f"{matrix.y_res[idx]:.6f}")
        row.append(f"{matrix.y_latent[idx]:.6f}")
        row.append(f"{matrix.y_obs[idx]:.6f}")
        row.append("1" if idx == matrix.top1_index else "0")
        output.write(",".join(row) + "\n")

    return output.getvalue()
