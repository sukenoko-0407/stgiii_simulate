"""
Streamlit application entry point for StageIII Simulator
"""

import streamlit as st

from app.sidebar import render_sidebar
from app.display import render_results


def main() -> None:
    """Streamlitアプリのエントリーポイント"""

    # ページ設定
    st.set_page_config(
        page_title="StageIII Simulator",
        page_icon=":test_tube:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # タイトル
    st.title("StageIII Simulator")
    st.markdown(
        "**Combinatorial Synthesis Stage Simulator for Drug Discovery**"
    )
    st.markdown(
        "Compare search strategies to efficiently find maximum-value compounds "
        "in a combinatorial space."
    )

    st.markdown("---")

    # サイドバーでパラメータ入力
    config, run_clicked = render_sidebar()

    # メイン領域
    if run_clicked:
        if config is not None:
            render_results(config)
        else:
            st.error(
                "Configuration error. Please check the sidebar settings."
            )
    else:
        # 初期表示
        st.info(
            "Configure simulation parameters in the sidebar and click "
            "**Run Simulation** to start."
        )

        # 使い方の説明
        with st.expander("How to Use", expanded=True):
            st.markdown("""
            ### Overview

            This simulator evaluates search strategies for finding the
            maximum-value compound (highest pIC50) in a combinatorial
            chemical space.

            ### Parameters

            1. **Operator**: Select a search strategy
               - **Random**: Uniform random selection
               - **Free-Wilson**: Free-Wilson model with Ridge regression
               - **Bayesian Free-Wilson**: Bayesian Free-Wilson with UCB acquisition

            2. **Number of Trials**: How many simulation runs to perform

            3. **Slots & Building Blocks**: Define the combinatorial space
               - 2-4 slots (A, B, C, D)
               - 10-50 building blocks per slot
               - Total combinations = product of all slot sizes

            4. **Main Effect Range**: Range for building block contributions

            5. **Error Range**: Range for noise/interaction terms

            6. **K per Step**: Number of cells to disclose per iteration

            7. **Top-k**: k value for Top-k evaluation metrics

            ### Metrics

            - **P_top1**: Number of cells disclosed to find the best compound
            - **P_topk**: Number of cells disclosed to find any Top-k compound
            """)

        with st.expander("About the Model"):
            st.markdown("""
            ### Data Generation

            Each cell's value is generated as:

            ```
            y = bias + Σ main_effect(slot_i, bb_i) + error
            ```

            - **bias**: Global bias = 6.0 (fixed)
            - **main_effect**: BB contribution ~ Uniform(range) + slot_bias
            - **error**: Noise ~ Normal(0, σ²), clipped to range
              - σ = (error_clip_high - error_clip_low) / 6.0
              - With default error range [-0.5, 0.5]: σ ≈ 0.167

            ### Initial Disclosure Methods

            Three methods are available:

            1. **Cross**: N-1 slots are fixed, 1 slot varies completely.
               Creates a "cross" pattern of initial observations.

            2. **Random Each BB**: Each building block appears at least once.
               For each BB in each slot, a cell is randomly selected where
               that BB is used. May have overlaps across slots.

            3. **None**: No initial disclosure. The search starts with
               random selection until the model can be trained.

            ### Evaluation

            - Simulations run until the Top-1 compound is found
            - P_top1 and P_topk are recorded for each trial
            - Statistics are computed across all trials
            """)


if __name__ == "__main__":
    main()
