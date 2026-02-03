"""
Display component for Streamlit UI
"""

import streamlit as st
import matplotlib.pyplot as plt

from stgiii_core.config import SimulationConfig
from stgiii_core.simulation import SimulationEngine
from app.utils import create_histogram, format_number


def render_results(config: SimulationConfig) -> None:
    """
    シミュレーションを実行し結果を表示

    Args:
        config: シミュレーション設定
    """
    # 実行条件サマリ
    st.subheader("Execution Settings")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Method", config.operator_type.value)
    col2.metric("Total Cells", f"{config.n_total_cells:,}")
    col3.metric("Trials", config.n_trials)
    col4.metric("K per Step", config.k_per_step)

    # スロット情報
    slot_info = " × ".join(
        f"{s.name}={s.n_building_blocks}" for s in config.slots
    )
    st.caption(f"Slots: {slot_info} | Initial Disclosure: None (fixed)")

    st.markdown("---")

    # プログレスバー
    progress_bar = st.progress(0)
    status_text = st.empty()

    def progress_callback(current: int, total: int) -> None:
        progress = current / total if total > 0 else 0
        progress_bar.progress(progress)
        status_text.text(f"Running... {current}/{total} trials completed")

    # シミュレーション実行
    try:
        engine = SimulationEngine(config, progress_callback=progress_callback)
        results = engine.run()
    except Exception as e:
        st.error(f"Simulation failed: {e}")
        return

    status_text.text("Completed!")
    progress_bar.progress(1.0)

    st.markdown("---")

    # 統計量
    stats = results.compute_statistics()

    st.subheader("Results Summary")

    # P_top1 統計量
    st.markdown("**P_top1** (Cells to reach Top-1)")
    p1 = stats["P_top1"]
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Median", format_number(p1["median"], 1))
    col2.metric("Mean", format_number(p1["mean"], 1))
    col3.metric("STD", format_number(p1["std"], 1))
    col4.metric("Min", format_number(p1["min"], 0))
    col5.metric("Max", format_number(p1["max"], 0))

    # P_top100_50 統計量
    st.markdown("**P_top100_50** (Cells to reach 50 of Top-100)")
    pk = stats["P_top100_50"]
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Median", format_number(pk["median"], 1))
    col2.metric("Mean", format_number(pk["mean"], 1))
    col3.metric("STD", format_number(pk["std"], 1))
    col4.metric("Min", format_number(pk["min"], 0))
    col5.metric("Max", format_number(pk["max"], 0))

    # 初期ヒット情報
    initial_hits = results.count_initial_hits()
    if initial_hits["top1"] > 0 or initial_hits["topk"] > 0:
        st.info(
            f"Initial disclosure hits - Top-1: {initial_hits['top1']}, "
            f"Top-{config.topk_k}: {initial_hits['topk']}"
        )

    st.markdown("---")

    # ヒストグラム（左右に並べて表示）
    st.subheader("Distribution Histograms")
    df = results.to_dataframe()

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**P_top1 Distribution**")
        fig = create_histogram(
            values=df["P_top1"].tolist(),
            title=f"P_top1 ({config.operator_type.value})",
            xlabel="P_top1 (Number of Disclosed Cells)",
            ylabel="Frequency",
            bins=min(30, len(df)),
            median_line=True
        )
        st.pyplot(fig)
        plt.close(fig)

    with col_right:
        st.markdown("**P_top100_50 Distribution**")
        fig_topk = create_histogram(
            values=df["P_top100_50"].tolist(),
            title=f"P_top100_50 ({config.operator_type.value})",
            xlabel="P_top100_50 (Number of Disclosed Cells)",
            ylabel="Frequency",
            bins=min(30, len(df)),
            median_line=True
        )
        st.pyplot(fig_topk)
        plt.close(fig_topk)

    st.markdown("---")

    # 試行別結果テーブル
    st.subheader("Trial Results")

    # 表示用にカラム名を調整
    display_df = df.copy()
    display_df = display_df.rename(columns={
        "trial_id": "Trial",
        "method": "Method",
        "n_total_cells": "Total Cells",
        "n_initial_disclosed": "Initial Disclosed",
        "k_value": "K",
        "topk_k": "Top-k",
        "P_top1": "P_top1",
        "P_topk": "P_topk",
        "P_top100_50": "P_top100_50",
        "n_steps": "Steps",
        "hit_in_initial_topk": "Top100 in Initial",
    })

    # 不要なカラムを削除
    columns_to_show = [
        "Trial", "P_top1", "P_top100_50", "Steps",
        "Initial Disclosed", "Top100 in Initial"
    ]
    display_df = display_df[columns_to_show]

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # CSVダウンロード
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download Results (CSV)",
        data=csv,
        file_name=f"simulation_results_{config.operator_type.value}.csv",
        mime="text/csv",
        use_container_width=True
    )

    # 設定サマリのダウンロード
    config_text = f"""Simulation Configuration
========================
Operator: {config.operator_type.value}
Trials: {config.n_trials}
Slots: {config.n_slots}
Slot Sizes: {config.slot_sizes}
Total Cells: {config.n_total_cells}
Initial Disclosure: None (fixed)
K per Step: {config.k_per_step}
Top-k: {config.topk_k}
Generation (v1.0):
  Fractions (variance): f_main={config.f_main}, f_int={config.f_int}, f_res={config.f_res}
  Distance: lambda={config.distance_lambda}
  Cliffs: eta_spike={config.eta_spike}, hotspots={config.spike_hotspots}
  Residual: nu_res={config.residual_nu}
  Embedding: d={config.embedding_dim}, series_divisor={config.embedding_series_divisor}, sigma_z={config.embedding_sigma}
  Interaction: rank={config.interaction_rank}
Random Seed: {config.random_seed}

Results Summary
===============
P_top1:
  Median: {p1['median']:.1f}
  Mean: {p1['mean']:.1f}
  STD: {p1['std']:.1f}
  Min: {p1['min']:.0f}
  Max: {p1['max']:.0f}

P_top100_50:
  Median: {pk['median']:.1f}
  Mean: {pk['mean']:.1f}
  STD: {pk['std']:.1f}
  Min: {pk['min']:.0f}
  Max: {pk['max']:.0f}
"""

    st.download_button(
        label="Download Summary (TXT)",
        data=config_text,
        file_name=f"simulation_summary_{config.operator_type.value}.txt",
        mime="text/plain",
        use_container_width=True
    )
