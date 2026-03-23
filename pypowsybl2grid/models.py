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

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [u.model_dump(exclude_none=True) for u in self.updates]
        ).set_index("id")


class RatioTapChangerUpdatePayload(BaseModel):
    updates: list[RatioTapChangerUpdate]

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [u.model_dump(exclude_none=True) for u in self.updates]
        ).set_index("id")
