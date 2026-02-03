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
               - **Bayesian Free-Wilson (UCB)**: Bayesian Free-Wilson with UCB acquisition
               - **Bayesian Free-Wilson (TS)**: Bayesian Free-Wilson with Thompson Sampling

            2. **Number of Trials**: How many simulation runs to perform

            3. **Slots & Building Blocks**: Define the combinatorial space
               - 2-4 slots (A, B, C, D)
               - 10-50 building blocks per slot
               - Total combinations = product of all slot sizes

            4. **Main Effect Range**: Range for building block contributions
               - Controlled indirectly via variance fractions (f_main/f_int/f_res)

            5. **Difficulty Controls**:
               - **f_main / f_int / f_res**: Variance fractions for
                 main effects / interactions / residual
               - **Hotspots (H)**: Number of interaction "cliffs"
               - **η_spike**: How cliff-dominant interactions are
               - **Residual ν**: Heavy-tail strength for unexplained error
               - **Distance λ**: How slot-distance attenuates interactions

            6. **K per Step**: Number of cells to disclose per iteration

            7. **Top-k**: fixed at 100 for evaluation metrics

            ### Metrics

            - **P_top1**: Number of cells disclosed to find the best compound
            - **P_topk**: Number of cells disclosed to find any Top-100 compound
            - **P_top100_50**: Number of cells disclosed to find 50 of Top-100
            """)

        with st.expander("About the Model"):
            st.markdown("""
            ### Data Generation

            Each cell's value is generated as:

            ```
            y_latent = μ0 + Σ main_effect(slot_i, bb_i)
                       + Σ pair_interaction(slot_s, bb_s, slot_t, bb_t)
                       + residual_error
            y_obs = clip(y_latent, [5, 11])
            ```

            - **main_effect**: BB contribution ~ Uniform(-1, 1) (scaled by f_main)
            - **pair_interaction**: Smooth + spike ("cliffs") interactions
              based on BB embeddings and slot-distance scaling exp(-D/λ)
            - **residual_error**: t-distribution (heavy-tail) controlled by ν
            - **Difficulty**: Controlled by variance fractions f_main/f_int/f_res

            ### Initial Disclosure

            - **None (fixed)**: No initial disclosure. The search starts with
              random selection until the model can be trained.

            ### Evaluation

            - Simulations run until the Top-1 compound is found
            - P_top1 and P_topk are recorded for each trial
            - Statistics are computed across all trials
            """)


if __name__ == "__main__":
    main()
