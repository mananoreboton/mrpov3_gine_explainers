"""
SDF-based 2D molecular visualization with GNNExplainer bond importance coloring.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdDepictor

_DRAW_SIZE = 500
_LOG = logging.getLogger(__name__)


def _html_escape(text: str) -> str:
    """Escape for safe HTML content."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _bond_importance_map(
    edge_index: Union[List[List[int]], Any],
    edge_mask: Union[List[float], Any],
) -> Dict[tuple, float]:
    """
    Build per-bond importance: key (min(u,v), max(u,v)), value = max of the two
    edge_mask entries for that bond. edge_index shape (2, E), edge_mask (E,).
    """
    if hasattr(edge_index, "cpu"):
        edge_index = edge_index.cpu()
    if hasattr(edge_mask, "cpu"):
        edge_mask = edge_mask.cpu()
    if hasattr(edge_index, "numpy"):
        edge_index = edge_index.numpy().tolist()
    if hasattr(edge_mask, "numpy"):
        edge_mask = edge_mask.numpy().tolist()
    if isinstance(edge_index, list) and len(edge_index) == 2:
        row0, row1 = edge_index
    else:
        row0 = [e[0] for e in edge_index]
        row1 = [e[1] for e in edge_index]
    bond_max: Dict[tuple, float] = {}
    for k in range(len(row0)):
        u, v = int(row0[k]), int(row1[k])
        key = (min(u, v), max(u, v))
        val = float(edge_mask[k]) if k < len(edge_mask) else 0.0
        bond_max[key] = max(bond_max.get(key, 0.0), val)
    return bond_max


def _importance_to_color(importance: float) -> tuple:
    """Map importance in [0, 1] to (r, g, b) in 0-1. Low=light grey, high=dark red."""
    if importance <= 0:
        return (0.85, 0.85, 0.85)
    if importance >= 1:
        return (0.8, 0.0, 0.0)
    # Interpolate grey -> red
    r = 0.85 + (0.8 - 0.85) * importance
    g = 0.85 * (1 - importance)
    b = 0.85 * (1 - importance)
    return (r, g, b)


def draw_molecule_with_mask(
    sdf_path: Path,
    edge_index: Union[List[List[int]], Any],
    edge_mask: Union[List[float], Any],
    out_path_png: Path,
) -> bool:
    """
    Load molecule from SDF, compute 2D coords, color bonds by edge_mask (max per bond), save PNG.
    Returns True if drawn successfully, False if SDF missing or draw failed.
    """
    if not sdf_path.exists():
        _LOG.warning("SDF not found: %s", sdf_path)
        return False
    mol = Chem.MolFromMolFile(str(sdf_path), removeHs=False)
    if mol is None:
        _LOG.warning("Failed to parse SDF: %s", sdf_path)
        return False
    n = mol.GetNumAtoms()
    if n == 0:
        out_path_png.parent.mkdir(parents=True, exist_ok=True)
        out_path_png.write_bytes(b"")
        return True

    bond_imp = _bond_importance_map(edge_index, edge_mask)
    rdDepictor.Compute2DCoords(mol)

    highlight_bonds: List[int] = []
    highlight_bond_colors: Dict[int, tuple] = {}
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        key = (min(i, j), max(i, j))
        imp = bond_imp.get(key, 0.0)
        idx = bond.GetIdx()
        highlight_bonds.append(idx)
        highlight_bond_colors[idx] = _importance_to_color(imp)

    w = h = _DRAW_SIZE
    out_path_png.parent.mkdir(parents=True, exist_ok=True)
    try:
        drawer = Draw.rdMolDraw2D.MolDraw2DCairo(w, h)
        # C++ signature: (mol, highlightAtoms, highlightBonds, highlightAtomColors, highlightBondColors, ...)
        drawer.DrawMolecule(
            mol,
            highlightAtoms=[],
            highlightBonds=highlight_bonds,
            highlightAtomColors={},
            highlightBondColors=highlight_bond_colors,
        )
        drawer.FinishDrawing()
        drawer.WriteDrawingText(str(out_path_png))
        return True
    except (AttributeError, OSError) as e:
        _LOG.warning("MolDraw2DCairo failed (%s), fallback to MolToImage", e)
        try:
            img = Draw.MolToImage(mol, size=(w, h))
            img.save(out_path_png)
            return True
        except Exception as e2:
            _LOG.warning("Fallback draw failed: %s", e2)
            return False


def write_explanation_index_html(out_path: Path, report_dict: Dict[str, Any]) -> None:
    """
    Write index.html with summary from explanation_report.json and per-graph cards
    with thumbnail links to graphs/mask_<pdb_id>.png.
    """
    mean_fid_plus = report_dict.get("mean_fidelity_plus", 0.0)
    mean_fid_minus = report_dict.get("mean_fidelity_minus", 0.0)
    num_graphs = report_dict.get("num_graphs", 0)
    per_graph = report_dict.get("per_graph", [])
    source_ts = report_dict.get("source_explanation_timestamp", "")

    style = (
        "body { font-family: sans-serif; max-width: 1200px; margin: 1em auto; padding: 0 1em; } "
        ".grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 1em; } "
        ".card { border: 1px solid #ccc; border-radius: 6px; overflow: hidden; text-align: center; } "
        ".card a { text-decoration: none; color: inherit; display: block; } "
        ".card img { width: 100%; height: auto; max-height: 200px; object-fit: contain; display: block; } "
        ".card .label { padding: 0.25em; font-size: 14px; font-weight: bold; } "
        ".card .meta { font-size: 12px; color: #666; padding: 0 0.25em 0.5em; } "
        "h1 { margin-bottom: 0.25em; } .timestamp { color: #666; font-size: 14px; margin-bottom: 1em; } "
    )
    body = [
        "<h1>Explanation visualizations</h1>",
        f"<p class='timestamp'>Source explanation run: {_html_escape(source_ts)}</p>",
        f"<p><strong>Mean fidelity (fid+)</strong>: {mean_fid_plus:.4f} &nbsp; "
        f"<strong>Mean fidelity (fid−)</strong>: {mean_fid_minus:.4f} &nbsp; "
        f"<strong>Graphs</strong>: {num_graphs}</p>",
        "<div class='grid'>",
    ]
    for e in per_graph:
        graph_id = e.get("graph_id", "?")
        fid_plus = e.get("fidelity_plus", 0.0)
        fid_minus = e.get("fidelity_minus", 0.0)
        auroc = e.get("auroc")
        auroc_str = f"{auroc:.4f}" if auroc is not None else "–"
        img_src = f"graphs/mask_{_html_escape(graph_id)}.png"
        body.append("<div class='card'>")
        body.append(f"  <a href='{img_src}' target='_blank'>")
        body.append(f"    <img src='{img_src}' alt='{_html_escape(graph_id)}' loading='lazy' onerror=\"this.alt='No image'\"/>")
        body.append(f"    <span class='label'>{_html_escape(graph_id)}</span>")
        body.append(
            f"    <span class='meta'>fid+ {fid_plus:.2f} fid− {fid_minus:.2f} auroc {_html_escape(auroc_str)}</span>"
        )
        body.append("  </a>")
        body.append("</div>")
    body.append("</div>")

    html_parts = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head><meta charset='utf-8'/><title>Explanation visualizations</title>",
        f"<style>{style}</style>",
        "</head><body>",
        "\n".join(body),
        "</body></html>",
    ]
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "index.html").write_text("\n".join(html_parts), encoding="utf-8")
