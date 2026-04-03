import pandas as pd
from pydantic import BaseModel


class PhaseTapChangerUpdate(BaseModel):
    id: str
    tap: int
    regulating: bool | None = None
    regulation_mode: str | None = None
    regulation_value: float | None = None
    regulated_side: str | None = None
    target_deadband: float | None = None
    fictitious: bool | None = None


class RatioTapChangerUpdate(BaseModel):
    id: str
    tap: int
    oltc: bool | None = None
    regulating: bool | None = None
    regulated_side: str | None = None
    target_v: float | None = None
    target_deadband: float | None = None


class PhaseTapChangerUpdatePayload(BaseModel):
    updates: list[PhaseTapChangerUpdate]

    def to_df(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.DataFrame(
            [u.model_dump(exclude_none=True) for u in self.updates]
        ).set_index("id")
        if "regulating" in df.columns:
            other_cols = [c for c in df.columns if c != "regulating"]
            df_enabling = df[df["regulating"]][other_cols + ["regulating"]]
            df_disabling = df[~df["regulating"]][["regulating"] + other_cols]
        else:
            df_enabling = df.iloc[0:0]
            df_disabling = df.iloc[0:0]
        return df_enabling, df_disabling


class RatioTapChangerUpdatePayload(BaseModel):
    updates: list[RatioTapChangerUpdate]

    def to_df(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.DataFrame(
            [u.model_dump(exclude_none=True) for u in self.updates]
        ).set_index("id")
        if "regulating" in df.columns:
            other_cols = [c for c in df.columns if c != "regulating"]
            df_enabling = df[df["regulating"]][other_cols + ["regulating"]]
            df_disabling = df[~df["regulating"]][["regulating"] + other_cols]
        else:
            df_enabling = df.iloc[0:0]
            df_disabling = df.iloc[0:0]
        return df_enabling, df_disabling


class ShuntUpdate(BaseModel):
    id: str
    section_count: str
    voltage_regulator_on: bool
    target_v: float | None
    target_deadband: float | None
    connected: bool


class ShuntUpdatePayload(BaseModel):
    updates: list[ShuntUpdate]

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame([u.model_dump() for u in self.updates]).set_index("id")
