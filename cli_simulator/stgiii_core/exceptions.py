"""
Exception classes for StageIII Simulator
"""


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
            f"{attempts}回の再生成でもargmaxが一意になりませんでした。"
            "設定条件を見直してください。"
        )


class InitialDisclosureError(StgIIISimulatorError):
    """初期開示に関するエラー"""
    def __init__(self, attempts: int) -> None:
        self.attempts = attempts
        super().__init__(
            f"{attempts}回の再抽選でも正解セルを回避できませんでした"
        )


class OperatorSelectionError(OperatorError):
    """Operatorが既開示セルを選択した"""
    def __init__(self, selected_index: int) -> None:
        self.selected_index = selected_index
        super().__init__(
            f"既開示セルを選択しました: index={selected_index}"
        )
