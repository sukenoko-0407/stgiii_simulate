"""
Sidebar component for Streamlit UI
"""

from typing import Tuple
import io
import numpy as np
import streamlit as st

from stgiii_core.config import SimulationConfig, SlotConfig, OperatorType, InitialDisclosureType
from stgiii_core.simulation import SimulationEngine
from stgiii_core.indexer import CellIndexer


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
        "Bayesian Free-Wilson": OperatorType.BAYESIAN_FW_UCB,
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

    # 主作用範囲
    st.sidebar.subheader("Main Effect Range")
    main_col1, main_col2 = st.sidebar.columns(2)
    main_low = main_col1.number_input(
        "Lower",
        value=-1.0,
        step=0.1,
        format="%.1f",
        help="Lower bound of main effect uniform distribution"
    )
    main_high = main_col2.number_input(
        "Upper",
        value=1.0,
        step=0.1,
        format="%.1f",
        help="Upper bound of main effect uniform distribution"
    )

    if main_low >= main_high:
        st.sidebar.error("Lower bound must be less than upper bound")
        return None, False

    # 誤差範囲
    st.sidebar.subheader("Error Clip Range")
    err_col1, err_col2 = st.sidebar.columns(2)
    err_low = err_col1.number_input(
        "Lower",
        value=-0.5,
        step=0.1,
        format="%.1f",
        key="err_low",
        help="Lower bound of error clipping"
    )
    err_high = err_col2.number_input(
        "Upper",
        value=0.5,
        step=0.1,
        format="%.1f",
        key="err_high",
        help="Upper bound of error clipping"
    )

    if err_low >= err_high:
        st.sidebar.error("Lower bound must be less than upper bound")
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
        slots=slots,
        main_effect_range=(float(main_low), float(main_high)),
        error_clip_range=(float(err_low), float(err_high)),
        seed=int(sample_seed)
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

    # 初期開示方式
    disclosure_options = {
        "Cross": InitialDisclosureType.CROSS,
        "Random Each BB": InitialDisclosureType.RANDOM_EACH_BB,
        "None": InitialDisclosureType.NONE,
    }
    disclosure_name = st.sidebar.selectbox(
        "Initial Disclosure Method",
        options=list(disclosure_options.keys()),
        index=0,
        help="Cross: N-1 slots fixed, 1 slot varies. Random Each BB: Each BB appears once randomly. None: No initial disclosure."
    )
    initial_disclosure_type = disclosure_options[disclosure_name]

    # 初期開示セル数の表示
    n_initial = SimulationEngine.calculate_initial_disclosure_count(
        slot_sizes, initial_disclosure_type
    )
    if initial_disclosure_type == InitialDisclosureType.RANDOM_EACH_BB:
        st.sidebar.caption(f"Initial Disclosure: up to {n_initial:,} cells (may vary due to overlaps)")
    elif initial_disclosure_type == InitialDisclosureType.NONE:
        st.sidebar.caption("Initial Disclosure: 0 cells (starts with random selection)")
    else:
        st.sidebar.caption(f"Initial Disclosure: {n_initial:,} cells")

    # 1ステップで開示するセル数K
    k_per_step = st.sidebar.selectbox(
        "Cells per Step (K)",
        options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        index=0,
        help="Number of cells to disclose in each iteration"
    )

    # Top-k の k
    topk_k = st.sidebar.selectbox(
        "Top-k (k value)",
        options=[5, 10, 25, 50, 100],
        index=1,
        help="k value for Top-k evaluation"
    )

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
                main_effect_range=(float(main_low), float(main_high)),
                error_clip_range=(float(err_low), float(err_high)),
                k_per_step=int(k_per_step),
                topk_k=int(topk_k),
                initial_disclosure_type=initial_disclosure_type,
                random_seed=random_seed,
            )
            return config, True
        except ValueError as e:
            st.sidebar.error(f"Configuration Error: {e}")
            return None, False

    return None, False


def _generate_sample_data(
    slots: list[SlotConfig],
    main_effect_range: Tuple[float, float],
    error_clip_range: Tuple[float, float],
    seed: int
) -> str:
    """
    サンプルデータを生成してCSV形式で返す

    Args:
        slots: スロット設定リスト
        main_effect_range: 主作用の範囲
        error_clip_range: 誤差のクリップ範囲
        seed: 乱数シード

    Returns:
        CSV形式の文字列
    """
    rng = np.random.default_rng(seed)

    # 設定値
    global_bias = 6.0  # 固定値
    slot_bias_range = (-0.5, 0.5)

    # インデクサー作成
    slot_sizes = tuple(s.n_building_blocks for s in slots)
    indexer = CellIndexer(slot_sizes)
    n_total = indexer.n_total

    # スロットバイアス
    n_slots = len(slots)
    slot_biases = rng.uniform(*slot_bias_range, size=n_slots)

    # 主作用（各スロット）
    main_low, main_high = main_effect_range
    main_effects: list[np.ndarray] = []
    for slot_idx, slot_config in enumerate(slots):
        n_bb = slot_config.n_building_blocks
        raw_main = rng.uniform(main_low, main_high, size=n_bb)
        main_with_bias = raw_main + slot_biases[slot_idx]
        main_effects.append(main_with_bias)

    # 誤差
    err_low, err_high = error_clip_range
    sigma_gen = (err_high - err_low) / 6.0
    errors_raw = rng.normal(0, sigma_gen, size=n_total)
    errors = np.clip(errors_raw, err_low, err_high)

    # 全セルの座標を取得
    all_indices = np.arange(n_total, dtype=np.int64)
    all_coords = indexer.batch_indices_to_coords(all_indices)

    # 主作用合計の計算
    main_effect_sum = np.zeros(n_total, dtype=np.float64)
    for slot_idx, main_effect in enumerate(main_effects):
        bb_indices = all_coords[:, slot_idx]
        main_effect_sum += main_effect[bb_indices]

    # y_trueの計算
    y_true = np.full(n_total, global_bias, dtype=np.float64)
    y_true += main_effect_sum
    y_true += errors

    # CSVの作成
    output = io.StringIO()

    # ヘッダー情報
    output.write(f"# Seed: {seed}\n")
    output.write(f"# Global Bias: {global_bias:.6f}\n")
    output.write(f"# Slot Biases: {', '.join(f'{b:.6f}' for b in slot_biases)}\n")
    output.write(f"# Main Effect Range: [{main_low}, {main_high}]\n")
    output.write(f"# Error Clip Range: [{err_low}, {err_high}]\n")
    output.write(f"# Sigma (Error Std): {sigma_gen:.6f}\n")
    output.write("#\n")

    # Main Effect値の出力
    output.write("# Main Effects per Slot/BB:\n")
    for slot_idx, slot_config in enumerate(slots):
        output.write(f"# Slot {slot_config.name}: ")
        output.write(", ".join(f"{v:.4f}" for v in main_effects[slot_idx]))
        output.write("\n")
    output.write("#\n")

    # セルデータのヘッダー
    slot_names = [s.name for s in slots]
    header_cols = ["CellIndex"] + [f"Slot_{name}" for name in slot_names]
    header_cols += ["MainEffectSum", "Error", "Total"]
    output.write(",".join(header_cols) + "\n")

    # セルデータの出力
    for idx in range(n_total):
        coords = all_coords[idx]
        row = [str(idx)]
        row += [str(c) for c in coords]
        row.append(f"{main_effect_sum[idx]:.6f}")
        row.append(f"{errors[idx]:.6f}")
        row.append(f"{y_true[idx]:.6f}")
        output.write(",".join(row) + "\n")

    return output.getvalue()
