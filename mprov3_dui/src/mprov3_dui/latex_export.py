from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _escape_text(s: str) -> str:
    out: list[str] = []
    for ch in s:
        if ch == "\\":
            out.append(r"\textbackslash{}")
        elif ch == "_":
            out.append(r"\_")
        elif ch == "&":
            out.append(r"\&")
        elif ch == "%":
            out.append(r"\%")
        elif ch == "#":
            out.append(r"\#")
        elif ch == "$":
            out.append(r"\$")
        elif ch == "{":
            out.append(r"\{")
        elif ch == "}":
            out.append(r"\}")
        elif ch == "~":
            out.append(r"\textasciitilde{}")
        elif ch == "^":
            out.append(r"\textasciicircum{}")
        else:
            out.append(ch)
    return "".join(out)


def _format_cell(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, (float, np.floating)) and (np.isnan(val) or np.isinf(val)):
        return ""
    if isinstance(val, (bool, np.bool_)):
        return "True" if bool(val) else "False"
    if isinstance(val, (int, np.integer)):
        return str(int(val))
    if isinstance(val, (float, np.floating)):
        return f"{float(val):.6f}"
    if pd.isna(val):
        return ""
    return _escape_text(str(val))


def dataframe_to_booktabs_latex(df: pd.DataFrame, *, caption: str | None = None) -> str:
    """Tabular with booktabs rules; numeric floats rounded to 6 decimals."""
    ncols = len(df.columns)
    colfmt = "l" * ncols
    lines: list[str] = [
        r"% Requires: \usepackage{booktabs}",
        r"\begin{tabular}{" + colfmt + "}",
        r"\toprule",
    ]
    header = " & ".join(_escape_text(str(c)) for c in df.columns) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")
    for _, row in df.iterrows():
        line = " & ".join(_format_cell(v) for v in row.tolist()) + r" \\"
        lines.append(line)
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if caption:
        lines.insert(
            0,
            r"\begin{table}[htbp]" + "\n"
            + r"\centering" + "\n"
            + r"\caption{" + _escape_text(caption) + "}" + "\n",
        )
        lines.append(r"\end{table}")
    return "\n".join(lines)
