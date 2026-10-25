"""Complex CSI -> real amplitude in the common 256-column schema."""
import numpy as np
import pandas as pd

N_SUB = 256
COLUMNS = [f"subcarrier_{i:03d}" for i in range(1, N_SUB + 1)]


def to_amplitude(csi: np.ndarray) -> np.ndarray:
    """|H| per subcarrier, kept in FFT-bin order (bin 0 = DC).

    Room B's supplied amplitude CSVs are also in FFT-bin order (their guard
    band sits at bins 123..133 and the VHT80 pilots at bins 11/39/75/103/
    153/181/217/245), so no reordering is applied to either room.
    """
    return np.abs(csi).astype(np.float32)


def to_frame(amp: np.ndarray) -> pd.DataFrame:
    assert amp.ndim == 2 and amp.shape[1] == N_SUB, amp.shape
    return pd.DataFrame(amp.astype(np.float32), columns=COLUMNS)


def read_common_csv(path) -> np.ndarray:
    """Read an intermediate (or Room-B-style) amplitude CSV as (n, 256) float32.

    Accepts any header naming as long as there are exactly 256 numeric columns
    in subcarrier order.
    """
    df = pd.read_csv(path)
    if df.shape[1] != N_SUB:
        raise ValueError(f"{path}: expected {N_SUB} columns, got {df.shape[1]}")
    return df.apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
