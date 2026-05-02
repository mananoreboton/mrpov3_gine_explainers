"""
SDF-based 2D molecular visualization with explainer importance coloring.
Supports bond coloring (edge_mask), atom coloring (node_mask), or both.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor

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


# ---------------------------------------------------------------------------
# Edge-mask → bond importance
# ---------------------------------------------------------------------------


def _bond_importance_map(
    edge_index: Union[List[List[int]], Any],
    edge_mask: Union[List[float], Any],
) -> Dict[tuple, float]:
    """Build per-bond importance: key (min(u,v), max(u,v)), value = max of the two directed entries."""
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


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------


def _importance_to_bond_color(importance: float) -> tuple:
    """Map importance in [0, 1] to (r, g, b). Low=light grey, high=dark red."""
    if importance <= 0:
        return (0.85, 0.85, 0.85)
    if importance >= 1:
        return (0.8, 0.0, 0.0)
    r = 0.85 + (0.8 - 0.85) * importance
    g = 0.85 * (1 - importance)
    b = 0.85 * (1 - importance)
    return (r, g, b)


def _importance_to_atom_color(importance: float) -> tuple:
    """Map importance in [0, 1] to (r, g, b). Low=light blue, high=dark orange."""
    if importance <= 0:
        return (0.88, 0.92, 1.0)
    if importance >= 1:
        return (0.9, 0.35, 0.0)
    r = 0.88 + (0.9 - 0.88) * importance
    g = 0.92 - (0.92 - 0.35) * importance
    b = 1.0 - 1.0 * importance
    return (r, g, b)


# ---------------------------------------------------------------------------
# Node-mask → atom importance
# ---------------------------------------------------------------------------


def _atom_importance_list(
    node_mask: Union[List[float], Any],
    num_atoms: int,
) -> List[float]:
    """Convert a node_mask (list or tensor) to a list of per-atom importance floats."""
    if hasattr(node_mask, "cpu"):
        node_mask = node_mask.cpu()
    if hasattr(node_mask, "numpy"):
        node_mask = node_mask.numpy().tolist()
    if isinstance(node_mask, list):
        vals = node_mask
    else:
        vals = list(node_mask)
    # Pad or truncate to num_atoms
    while len(vals) < num_atoms:
        vals.append(0.0)
    return [float(v) for v in vals[:num_atoms]]


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def draw_molecule_with_mask(
    sdf_path: Path,
    edge_index: Union[List[List[int]], Any, None] = None,
    edge_mask: Union[List[float], Any, None] = None,
    out_path_png: Optional[Path] = None,
    node_mask: Union[List[float], Any, None] = None,
) -> bool:
    """
    Load molecule from SDF, colour bonds by edge_mask and/or atoms by node_mask,
    save PNG.  Returns True on success.
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
        if out_path_png is not None:
            out_path_png.parent.mkdir(parents=True, exist_ok=True)
            out_path_png.write_bytes(b"")
        return True

    rdDepictor.Compute2DCoords(mol)

    # Bond highlights (edge_mask)
    highlight_bonds: List[int] = []
    highlight_bond_colors: Dict[int, tuple] = {}
    if edge_index is not None and edge_mask is not None:
        bond_imp = _bond_importance_map(edge_index, edge_mask)
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            key = (min(i, j), max(i, j))
            imp = bond_imp.get(key, 0.0)
            idx = bond.GetIdx()
            highlight_bonds.append(idx)
            highlight_bond_colors[idx] = _importance_to_bond_color(imp)

    # Atom highlights (node_mask)
    highlight_atoms: List[int] = []
    highlight_atom_colors: Dict[int, tuple] = {}
    if node_mask is not None:
        scores = _atom_importance_list(node_mask, n)
        for atom_idx, imp in enumerate(scores):
            highlight_atoms.append(atom_idx)
            highlight_atom_colors[atom_idx] = _importance_to_atom_color(imp)

    w = h = _DRAW_SIZE
    if out_path_png is not None:
        out_path_png.parent.mkdir(parents=True, exist_ok=True)
    try:
        drawer = Draw.rdMolDraw2D.MolDraw2DCairo(w, h)
        drawer.DrawMolecule(
            mol,
            highlightAtoms=highlight_atoms,
            highlightBonds=highlight_bonds,
            highlightAtomColors=highlight_atom_colors,
            highlightBondColors=highlight_bond_colors,
        )
        drawer.FinishDrawing()
        if out_path_png is not None:
            drawer.WriteDrawingText(str(out_path_png))
        return True
    except (AttributeError, OSError) as e:
        _LOG.warning("MolDraw2DCairo failed (%s), fallback to MolToImage", e)
        try:
            img = Draw.MolToImage(mol, size=(w, h))
            if out_path_png is not None:
                img.save(out_path_png)
            return True
        except Exception as e2:
            _LOG.warning("Fallback draw failed: %s", e2)
            return False


def draw_molecule_base(sdf_path: Path, out_path_png: Path) -> bool:
    """
    Load molecule from SDF and draw a plain 2D structure (no explainer highlighting).
    """
    return draw_molecule_with_mask(sdf_path, out_path_png=out_path_png)


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------


def write_explanation_index_html(out_path: Path, report_dict: Dict[str, Any]) -> None:
    """Write index.html with summary and per-graph cards."""
    mean_fid_plus = report_dict.get("mean_fidelity_plus", 0.0)
    mean_fid_minus = report_dict.get("mean_fidelity_minus", 0.0)
    mean_char = report_dict.get("mean_pyg_characterization", 0.0)
    mean_fsuf = report_dict.get("mean_paper_sufficiency", 0.0)
    mean_fcom = report_dict.get("mean_paper_comprehensiveness", 0.0)
    mean_ff1 = report_dict.get("mean_paper_f1_fidelity", 0.0)
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
        f"<strong>Mean fidelity (fid&minus;)</strong>: {mean_fid_minus:.4f} &nbsp; "
        f"<strong>Char (PyG)</strong>: {mean_char:.4f} &nbsp; "
        f"<strong>Fsuf (raw)</strong>: {mean_fsuf:.4f} &nbsp; "
        f"<strong>Fcom (raw)</strong>: {mean_fcom:.4f} &nbsp; "
        f"<strong>Ff1 (clamped)</strong>: {mean_ff1:.4f} &nbsp; "
        f"<strong>Graphs</strong>: {num_graphs}</p>",
        "<div class='grid'>",
    ]
    for e in per_graph:
        graph_id = e.get("graph_id", "?")
        fid_plus = e.get("fidelity_plus", 0.0)
        fid_minus = e.get("fidelity_minus", 0.0)
        char = e.get("pyg_characterization", 0.0)
        fsuf = e.get("paper_sufficiency", 0.0)
        fcom = e.get("paper_comprehensiveness", 0.0)
        ff1 = e.get("paper_f1_fidelity", 0.0)
        img_src = f"graphs/mask_{_html_escape(graph_id)}.png"
        body.append("<div class='card'>")
        body.append(f"  <a href='{img_src}' target='_blank'>")
        body.append(f"    <img src='{img_src}' alt='{_html_escape(graph_id)}' loading='lazy' onerror=\"this.alt='No image'\"/>")
        body.append(f"    <span class='label'>{_html_escape(graph_id)}</span>")
        body.append(
            f"    <span class='meta'>"
            f"fid+ {fid_plus:.2f} fid&minus; {fid_minus:.2f} "
            f"char {char:.2f} "
            f"Fsuf(raw) {fsuf:.2f} Fcom(raw) {fcom:.2f} Ff1(clamped) {ff1:.2f}"
            f"</span>"
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


def write_comparison_index_html(
    out_path: Path,
    comparison_data: Dict[str, Any],
) -> None:
    """Write a cross-explainer comparison index.html.

    ``comparison_data`` has shape::

        {
            "explainers": ["GRADEXPINODE", ...],
            "graph_ids": ["7BQY", ...],
            "per_explainer": { "GRADEXPINODE": {"mean_fid_plus": ..., ...}, ... },
            "grid": { "7BQY": { "GRADEXPINODE": {"img": "GRADEXPINODE/graphs/mask_7BQY.png", "fid_plus": ...}, ... }, ... },
        }
    """
    explainers = comparison_data.get("explainers", [])
    graph_ids = comparison_data.get("graph_ids", [])
    per_explainer = comparison_data.get("per_explainer", {})
    grid = comparison_data.get("grid", {})

    style = (
        "body { font-family: sans-serif; margin: 1em; } "
        "table { border-collapse: collapse; width: 100%; } "
        "th, td { border: 1px solid #ccc; padding: 4px; text-align: center; font-size: 13px; } "
        "th { background: #f5f5f5; position: sticky; top: 0; } "
        "img { max-width: 150px; max-height: 150px; object-fit: contain; } "
        ".summary { margin-bottom: 1em; } "
        ".summary td { font-weight: bold; } "
    )

    rows: list[str] = []
    # Summary row
    rows.append("<table class='summary'><tr><th>Explainer</th>")
    for ex in explainers:
        rows.append(f"<th>{_html_escape(ex)}</th>")
    rows.append("</tr><tr><td>Mean fid+</td>")
    for ex in explainers:
        v = per_explainer.get(ex, {}).get("mean_fid_plus", 0.0)
        rows.append(f"<td>{v:.4f}</td>")
    rows.append("</tr><tr><td>Mean fid&minus;</td>")
    for ex in explainers:
        v = per_explainer.get(ex, {}).get("mean_fid_minus", 0.0)
        rows.append(f"<td>{v:.4f}</td>")
    rows.append("</tr><tr><td>Mean char (PyG)</td>")
    for ex in explainers:
        v = per_explainer.get(ex, {}).get("mean_pyg_characterization", 0.0)
        rows.append(f"<td>{v:.4f}</td>")
    rows.append("</tr><tr><td>Mean Fsuf (raw)</td>")
    for ex in explainers:
        v = per_explainer.get(ex, {}).get("mean_paper_sufficiency", 0.0)
        rows.append(f"<td>{v:.4f}</td>")
    rows.append("</tr><tr><td>Mean Fcom (raw)</td>")
    for ex in explainers:
        v = per_explainer.get(ex, {}).get("mean_paper_comprehensiveness", 0.0)
        rows.append(f"<td>{v:.4f}</td>")
    rows.append("</tr><tr><td>Mean Ff1 (clamped)</td>")
    for ex in explainers:
        v = per_explainer.get(ex, {}).get("mean_paper_f1_fidelity", 0.0)
        rows.append(f"<td>{v:.4f}</td>")
    rows.append("</tr></table>")

    # Per-graph grid
    rows.append("<table><tr><th>Graph</th>")
    for ex in explainers:
        rows.append(f"<th>{_html_escape(ex)}</th>")
    rows.append("</tr>")
    for gid in graph_ids:
        rows.append(f"<tr><td>{_html_escape(gid)}</td>")
        for ex in explainers:
            cell = grid.get(gid, {}).get(ex, {})
            img = cell.get("img", "")
            fp = cell.get("fid_plus", 0.0)
            fm = cell.get("fid_minus", 0.0)
            ff1 = cell.get("paper_f1_fidelity", 0.0)
            if img:
                rows.append(
                    f"<td><a href='{_html_escape(img)}' target='_blank'>"
                    f"<img src='{_html_escape(img)}' loading='lazy'/></a>"
                    f"<br/>fid+ {fp:.2f} fid&minus; {fm:.2f}"
                    f"<br/>Ff1 {ff1:.2f}</td>"
                )
            else:
                rows.append("<td>-</td>")
        rows.append("</tr>")
    rows.append("</table>")

    html = "\n".join([
        "<!DOCTYPE html><html lang='en'>",
        "<head><meta charset='utf-8'/><title>Explainer comparison</title>",
        f"<style>{style}</style></head><body>",
        "<h1>Explainer comparison</h1>",
        "\n".join(rows),
        "</body></html>",
    ])
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "comparison.html").write_text(html, encoding="utf-8")
