"""
Visualize ligand graphs from a built PyG dataset (data.pt).

By default walks **CV folds** (from raw ``Splits/``), and within each fold **train →
val → test**, in ``pdb_order.txt`` order when available. Each dataset graph is **drawn
at most once** (first appearance in that iteration); ``index.html`` still lists every
planned **(fold, split)** placement, so the same PDB may appear as multiple thumbnails
linking to the same report when it sits in different folds' partitions.

Optional ``--fold_index`` / ``--fold_indices`` limit which folds appear in the plan.
Optional ``--num-graphs-by-fold`` caps how many **index entries** (and first-time draws)
per **(fold, split)** bucket. ``--indices`` overrides the default plan.

Uses RDKit's 2D drawer (MolDraw2D) for publication-quality graphics. For each
selected graph this script:
- Builds an RDKit molecule from the graph and sets the full 3D conformer (x, y, z).
- Generates optimal 2D coordinates from the 3D structure via
  GenerateDepictionMatching3DStructure(), so the drawing respects 3D layout.
- Bond styles: single = one central line; double = two shifted lines;
  triple = two shifted + central; aromatic = dashed.
- Saves PNG and SVG under results/visualizations/; writes HTML reports with tables.
- ``index.html`` uses a **tab per CV fold**, then **train / val / test** inside each tab.

Usage (examples):
    uv run python visualize_graphs.py
    uv run python visualize_graphs.py --num-graphs-by-fold 4
    uv run python visualize_graphs.py --fold_indices 0 1 --num-graphs-by-fold 8
    uv run python visualize_graphs.py --indices 0 1 2 3
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import BondType as RkBondType
from rdkit.Chem import rdDepictor
from rdkit.Geometry import Point3D

from mprov3_gine_explainer_defaults import (
    BUILT_DATASET_FOLDER_NAME,
    DEFAULT_DATA_ROOT,
    DEFAULT_NUM_FOLDS,
    DEFAULT_RESULTS_ROOT,
    DEFAULT_TEST_SPLIT_FILE,
    DEFAULT_TRAIN_SPLIT_FILE,
    DEFAULT_VAL_SPLIT_FILE,
    RESULTS_DATASETS,
    RESULTS_VISUALIZATIONS,
    resolve_dataset_dir,
    resolve_fold_indices,
)
from dataset import (
    MProV3Dataset,
    ORIGINAL_CATEGORY_FROM_CLASS,
    load_dataset_pdb_order,
    load_splits,
)
from utils import RunLogger, log_overwrite_dir_if_nonempty, html_document, html_escape

# Image size in pixels (RDKit drawer uses this for PNG/SVG canvas).
_DRAW_SIZE = 500

# Default plan: (dataset_idx, fold_index, split). ``--indices`` uses Nones for fold/split.
PlanItem = Tuple[int, Optional[int], Optional[str]]
_SPLIT_ORDER = ("train", "val", "test")


@dataclass(frozen=True)
class BondVisual:
    """Human-readable bond type for reports."""

    label: str


def bond_scalar_to_visual(value: float) -> BondVisual:
    """Map stored scalar bond type to a label (single/double/triple/aromatic)."""
    if np.isclose(value, 1.0):
        return BondVisual(label="single")
    if np.isclose(value, 2.0):
        return BondVisual(label="double")
    if np.isclose(value, 3.0):
        return BondVisual(label="triple")
    if np.isclose(value, 1.5):
        return BondVisual(label="aromatic")
    return BondVisual(label=f"unknown({value:.2f})")


def _bond_scalar_to_rdkit(value: float) -> RkBondType:
    """Map stored bond scalar to RDKit BondType for correct drawing."""
    if np.isclose(value, 2.0):
        return RkBondType.DOUBLE
    if np.isclose(value, 3.0):
        return RkBondType.TRIPLE
    if np.isclose(value, 1.5):
        return RkBondType.AROMATIC
    return RkBondType.SINGLE


def _unique_undirected_edges(
    edge_index: torch.Tensor, edge_attr: torch.Tensor
) -> List[Tuple[int, int, float]]:
    """
    Collapse bidirectional edges into a unique undirected list.

    Returns a list of (u, v, bond_scalar) with u < v.
    """
    if edge_index.numel() == 0:
        return []
    ei = edge_index.cpu().numpy()
    ea = edge_attr.view(-1).cpu().numpy()
    seen = {}
    for col in range(ei.shape[1]):
        u = int(ei[0, col])
        v = int(ei[1, col])
        if u == v:
            continue
        key = (u, v) if u < v else (v, u)
        if key not in seen:
            seen[key] = float(ea[col])
    return [(u, v, s) for (u, v), s in seen.items()]


def _pdb_to_dataset_index(
    ds: MProV3Dataset,
    pdb_order: Optional[Sequence[str]],
) -> dict[str, int]:
    """Map PDB ID → dataset index (``pdb_order`` when present, else first ``pdb_id`` on graph)."""
    n = len(ds)
    out: dict[str, int] = {}
    if pdb_order is not None:
        for i, p in enumerate(pdb_order):
            if i < n and p not in out:
                out[str(p)] = i
    else:
        for i in range(n):
            pid = str(getattr(ds[i], "pdb_id", "") or "")
            if pid and pid not in out:
                out[pid] = i
    return out


def plan_by_fold_and_split(
    ds: MProV3Dataset,
    pdb_order: Optional[Sequence[str]],
    splits_root: Path,
    train_file: str,
    val_file: str,
    test_file: str,
    num_folds: int,
    fold_list: Sequence[int],
) -> List[Tuple[int, int, str]]:
    """
    One index row per (fold, split, pdb) that maps into the dataset.

    Order: folds in ``fold_list``, then train → val → test, then ``pdb_order`` order
    within each split (or split-file order if ``pdb_order`` is missing).
    """
    n = len(ds)
    pdb_to_idx = _pdb_to_dataset_index(ds, pdb_order)
    splits = load_splits(splits_root, train_file, val_file, test_file, num_folds)
    plan: List[Tuple[int, int, str]] = []

    for k in fold_list:
        if k < 0 or k >= len(splits):
            continue
        train_ids, val_ids, test_ids = splits[k][0], splits[k][1], splits[k][2]
        for split_name, id_list in zip(_SPLIT_ORDER, (train_ids, val_ids, test_ids)):
            id_set = set(id_list)
            if pdb_order is not None:
                pdbs_ordered = [p for p in pdb_order if p in id_set]
            else:
                pdbs_ordered = list(id_list)
            seen_pdb: set[str] = set()
            for pdb in pdbs_ordered:
                ps = str(pdb)
                if ps in seen_pdb:
                    continue
                seen_pdb.add(ps)
                idx = pdb_to_idx.get(ps)
                if idx is None or not (0 <= idx < n):
                    continue
                plan.append((idx, k, split_name))
    return plan


def apply_per_fold_split_cap(
    plan: List[Tuple[int, int, str]],
    cap: int,
) -> List[Tuple[int, int, str]]:
    """Keep at most ``cap`` plan rows per (fold, split) bucket, preserving order."""
    counts: dict[Tuple[int, str], int] = defaultdict(int)
    out: List[Tuple[int, int, str]] = []
    for idx, k, s in plan:
        key = (k, s)
        if counts[key] < cap:
            counts[key] += 1
            out.append((idx, k, s))
    return out


def _mol_from_graph(
    atomic_numbers: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    pos_3d: np.ndarray,
) -> Chem.Mol:
    """
    Build an RDKit molecule from the PyG graph with a 3D conformer (x, y, z).

    - pos_3d: (N, 3) array (x, y, z). Used for GenerateDepictionMatching3DStructure.
    """
    anum = atomic_numbers.detach().cpu().numpy()
    n = len(anum)
    mol = Chem.RWMol()
    for i in range(n):
        mol.AddAtom(Chem.Atom(int(anum[i])))
    for u, v, scalar in _unique_undirected_edges(edge_index, edge_attr):
        bt = _bond_scalar_to_rdkit(scalar)
        mol.AddBond(int(u), int(v), bt)
    mol = mol.GetMol()
    if n == 0:
        return mol
    conf = Chem.Conformer(n)
    for i in range(n):
        conf.SetAtomPosition(
            i,
            Point3D(
                float(pos_3d[i, 0]),
                float(pos_3d[i, 1]),
                float(pos_3d[i, 2]),
            ),
        )
    mol.AddConformer(conf, assignId=True)
    return mol


def _generate_2d_from_3d(mol: Chem.Mol) -> None:
    """
    Generate 2D coordinates that mimic the molecule's 3D structure.

    Uses RDKit's GenerateDepictionMatching3DStructure so the final 2D
    depiction preserves spatial relationships from the 3D conformer.
    Modifies mol in place (replaces/adds 2D conformer).
    """
    if mol.GetNumAtoms() == 0:
        return
    try:
        # reference = same molecule with 3D; confId=0 is the conformer we added
        rdDepictor.GenerateDepictionMatching3DStructure(
            mol,
            mol,
            confId=0,
            acceptFailure=True,
            forceRDKit=False,
        )
    except Exception:
        # Fallback: standard 2D coords from topology only (ignores 3D)
        rdDepictor.Compute2DCoords(mol, clearConfs=True)


def draw_graph(
    pos3d: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    atomic_numbers: torch.Tensor,
    out_path_png: Path,
    out_path_svg: Optional[Path] = None,
) -> None:
    """
    Draw a single molecular graph with RDKit (MolDraw2D) and save PNG (and optionally SVG).

    Uses full 3D positions to build the molecule; RDKit generates optimal 2D coordinates
    from the 3D structure (GenerateDepictionMatching3DStructure). Bond drawing:
    single = one central line, double = two shifted lines, triple = two shifted + central,
    aromatic = dashed.
    """
    pos_3d = pos3d.detach().cpu().numpy()  # (N, 3)

    mol = _mol_from_graph(atomic_numbers, edge_index, edge_attr, pos_3d)
    _generate_2d_from_3d(mol)
    if mol.GetNumAtoms() == 0:
        out_path_png.parent.mkdir(parents=True, exist_ok=True)
        out_path_png.write_bytes(b"")
        if out_path_svg:
            out_path_svg.write_text("<!-- empty molecule -->", encoding="utf-8")
        return

    w = h = _DRAW_SIZE
    out_path_png.parent.mkdir(parents=True, exist_ok=True)

    # Prefer MolDraw2DCairo (best quality); fall back to MolToImage if Cairo not built.
    try:
        drawer = Draw.rdMolDraw2D.MolDraw2DCairo(w, h)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        drawer.WriteDrawingText(str(out_path_png))
    except (AttributeError, OSError):
        img = Draw.MolToImage(mol, size=(w, h))
        img.save(out_path_png)

    if out_path_svg is not None:
        drawer_svg = Draw.rdMolDraw2D.MolDraw2DSVG(w, h)
        drawer_svg.DrawMolecule(mol)
        drawer_svg.FinishDrawing()
        out_path_svg.write_text(drawer_svg.GetDrawingText(), encoding="utf-8")


def write_html_report(
    out_dir: Path,
    image_filename: str,
    pdb_id: str,
    category: Optional[int],
    pIC50: Optional[float],
    pos3d: torch.Tensor,
    atomic_numbers: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    svg_filename: Optional[str] = None,
) -> None:
    """
    Write an HTML report for a single graph including:
    - header with PDB ID, category, optional pIC50
    - embedded image
    - node and edge tables
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    pos_np = pos3d.detach().cpu().numpy()
    atomic_np = atomic_numbers.detach().cpu().numpy()
    unique_edges = _unique_undirected_edges(edge_index, edge_attr)

    style = (
        "body { font-family: sans-serif; } "
        "table { border-collapse: collapse; margin-bottom: 1.5em; } "
        "th, td { border: 1px solid #ccc; padding: 4px 8px; font-size: 12px; } "
        "th { background: #f0f0f0; }"
    )
    body: List[str] = [
        f"<h1>MPro ligand graph: {html_escape(pdb_id)}</h1>",
        "<p>",
        f"<strong>PDB ID</strong>: {html_escape(pdb_id)}<br/>",
    ]
    if category is not None:
        body.append(f"<strong>Category</strong>: {category}<br/>")
    if pIC50 is not None:
        body.append(f"<strong>pIC50</strong>: {pIC50:.3f}<br/>")
    body.append("</p>")
    body.append(f"<p><img src='{html_escape(image_filename)}' alt='Graph {pdb_id}'/></p>")
    if svg_filename:
        body.append(f"<p><a href='{html_escape(svg_filename)}'>Vector (SVG)</a></p>")

    body.append("<h2>Nodes (atoms)</h2>")
    body.append("<table>")
    body.append("<tr><th>Index</th><th>Atomic number</th><th>x</th><th>y</th><th>z</th></tr>")
    for idx in range(pos_np.shape[0]):
        x, y, z = pos_np[idx]
        zn = int(atomic_np[idx])
        body.append(
            "<tr>"
            f"<td>{idx}</td>"
            f"<td>{zn}</td>"
            f"<td>{x:.4f}</td>"
            f"<td>{y:.4f}</td>"
            f"<td>{z:.4f}</td>"
            "</tr>"
        )
    body.append("</table>")
    body.append("<h2>Edges (bonds)</h2>")
    body.append("<table>")
    body.append("<tr><th>Source</th><th>Target</th><th>bond_scalar</th><th>bond_type</th></tr>")
    for u, v, scalar in unique_edges:
        visual = bond_scalar_to_visual(scalar)
        body.append(
            "<tr>"
            f"<td>{u}</td>"
            f"<td>{v}</td>"
            f"<td>{scalar:.2f}</td>"
            f"<td>{html_escape(visual.label)}</td>"
            "</tr>"
        )
    body.append("</table>")

    html = html_document(f"MPro ligand graph: {html_escape(pdb_id)}", body, style=style)
    (out_dir / f"{pdb_id}.html").write_text(html, encoding="utf-8")


def _index_card_lines(pdb_id: str, category: Optional[int], pIC50: Optional[float]) -> List[str]:
    safe_id = html_escape(pdb_id)
    img_src = html_escape(f"{pdb_id}.png")
    report_href = html_escape(f"{pdb_id}.html")
    meta_parts = []
    if category is not None:
        meta_parts.append(f"Cat. {category}")
    if pIC50 is not None:
        meta_parts.append(f"pIC50 {pIC50:.2f}")
    meta_str = html_escape(" · ".join(meta_parts)) if meta_parts else ""
    lines = [
        "<div class='card'>",
        f"  <a href='{report_href}'>",
        f"    <img src='{img_src}' alt='{safe_id}' loading='lazy' />",
        f"    <span class='label'>{safe_id}</span>",
    ]
    if meta_str:
        lines.append(f"    <span class='meta'>{meta_str}</span>")
    lines.extend(["  </a>", "</div>"])
    return lines


def write_index_html(
    out_dir: Path,
    entries: List[Tuple[str, Optional[int], Optional[float], Optional[int], Optional[str]]],
) -> None:
    """
    Write an index.html page with thumbnail links to all generated graph reports.

    Default entries are grouped by CV **fold** (one **tab** per fold in ``index.html``),
    then **train / val / test** inside each panel. The same PDB may appear multiple times
    (different fold/split slots) linking to the same files.

    entries: (pdb_id, category, pIC50, fold_index, split) with ``None`` fold/split for
    ``--indices`` runs (single **Other** section).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    style = (
        "body { font-family: sans-serif; max-width: 1200px; margin: 1em auto; padding: 0 1em; } "
        ".grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 1em; } "
        ".card { border: 1px solid #ccc; border-radius: 6px; overflow: hidden; text-align: center; } "
        ".card a { text-decoration: none; color: inherit; display: block; } "
        ".card img { width: 100%; height: auto; display: block; } "
        ".card .label { padding: 0.5em; font-size: 14px; } "
        ".card .meta { font-size: 12px; color: #666; } "
        "h1 { margin-bottom: 0.5em; } "
        "h2.fold-section { margin-top: 1.75em; margin-bottom: 0.75em; padding-bottom: 0.25em; "
        "border-bottom: 1px solid #ddd; font-size: 1.25rem; } "
        "h2.fold-section:first-of-type { margin-top: 0.5em; } "
        "h3.split-section { margin-top: 1.25em; margin-bottom: 0.5em; font-size: 1.05rem; color: #333; } "
        ".timestamp { color: #666; font-size: 14px; margin-bottom: 1em; } "
        ".fold-tablist { display: flex; flex-wrap: wrap; gap: 0.35em; margin: 1em 0 0.5em 0; "
        "padding: 0; list-style: none; border-bottom: 2px solid #ddd; } "
        ".fold-tablist button { font: inherit; padding: 0.5em 1em; margin-bottom: -2px; "
        "border: 2px solid transparent; border-radius: 6px 6px 0 0; background: #f5f5f5; "
        "cursor: pointer; color: #333; } "
        ".fold-tablist button:hover { background: #eee; } "
        ".fold-tablist button[aria-selected='true'] { background: #fff; border-color: #ddd; "
        "border-bottom-color: #fff; font-weight: 600; } "
        ".fold-tabpanel { padding-top: 0.5em; } "
        ".fold-tabpanel[hidden] { display: none !important; } "
    )

    by_fold: dict[int, dict[str, List[Tuple[str, Optional[int], Optional[float]]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    other: List[Tuple[str, Optional[int], Optional[float]]] = []

    for pdb_id, category, pIC50, fold, split in entries:
        if fold is None or split is None:
            other.append((pdb_id, category, pIC50))
        else:
            by_fold[fold][split].append((pdb_id, category, pIC50))

    body: List[str] = [
        "<h1>MPro ligand graphs</h1>",
        f"<p class='timestamp'>Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} — "
        f"{len(entries)} index entries</p>",
        "<p class='meta' style='font-size: 13px; color: #555;'>Same PDB may appear under multiple "
        "fold/split headings; thumbnails link to the same PNG/HTML when only one draw ran.</p>",
    ]

    fold_keys = sorted(by_fold.keys())
    use_tabs = len(fold_keys) > 0

    if use_tabs:
        body.append('<div class="fold-tabs" id="fold-tabs-root">')
        body.append('<div role="tablist" class="fold-tablist" aria-label="CV fold">')
        for i, fold in enumerate(fold_keys):
            split_map = by_fold[fold]
            n_fold = sum(len(v) for v in split_map.values())
            panel_id = f"viz-fold-{fold}"
            sel = "true" if i == 0 else "false"
            body.append(
                f'<button type="button" role="tab" id="tab-{html_escape(panel_id)}" '
                f'aria-controls="{html_escape(panel_id)}" aria-selected="{sel}" '
                f'data-viz-panel="{html_escape(panel_id)}">Fold {fold} ({n_fold})</button>'
            )
        if other:
            opanel = "viz-fold-other"
            body.append(
                f'<button type="button" role="tab" id="tab-{html_escape(opanel)}" '
                f'aria-controls="{html_escape(opanel)}" aria-selected="false" '
                f'data-viz-panel="{html_escape(opanel)}">Other ({len(other)})</button>'
            )
        body.append("</div>")

        for i, fold in enumerate(fold_keys):
            split_map = by_fold[fold]
            n_fold = sum(len(v) for v in split_map.values())
            panel_id = f"viz-fold-{fold}"
            hidden = "" if i == 0 else " hidden"
            labelled_by = f"tab-{html_escape(panel_id)}"
            body.append(
                f'<div class="fold-tabpanel" role="tabpanel" id="{html_escape(panel_id)}" '
                f'aria-labelledby="{labelled_by}"{hidden}>'
            )
            body.append(
                f"<h2 class='fold-section'>Fold {fold} ({n_fold} entries)</h2>"
            )
            for split in _SPLIT_ORDER:
                row = split_map.get(split, [])
                body.append(
                    f"<h3 class='split-section'>{split.capitalize()} ({len(row)})</h3>"
                )
                body.append("<div class='grid'>")
                for pdb_id, category, pIC50 in row:
                    body.extend(_index_card_lines(pdb_id, category, pIC50))
                body.append("</div>")
            body.append("</div>")

        if other:
            panel_id = "viz-fold-other"
            hidden = " hidden" if fold_keys else ""
            labelled_by = f"tab-{html_escape(panel_id)}"
            body.append(
                f'<div class="fold-tabpanel" role="tabpanel" id="{html_escape(panel_id)}" '
                f'aria-labelledby="{labelled_by}"{hidden}>'
            )
            body.append(f"<h2 class='fold-section'>Other ({len(other)} graphs)</h2>")
            body.append(
                "<p class='meta' style='margin: -0.5em 0 1em 0;'>"
                "No fold/split label (e.g. explicit --indices).</p>"
            )
            body.append("<div class='grid'>")
            for pdb_id, category, pIC50 in other:
                body.extend(_index_card_lines(pdb_id, category, pIC50))
            body.append("</div>")
            body.append("</div>")

        body.append("</div>")
        body.append(
            "<script>"
            "(function(){"
            "var root=document.getElementById('fold-tabs-root');"
            "if(!root)return;"
            "var tabs=root.querySelectorAll('[role=tab][data-viz-panel]');"
            "var panels=root.querySelectorAll('[role=tabpanel]');"
            "function show(id){"
            "panels.forEach(function(p){var on=(p.id===id);p.hidden=!on;});"
            "tabs.forEach(function(t){var on=(t.getAttribute('data-viz-panel')===id);"
            "t.setAttribute('aria-selected',on?'true':'false');});"
            "}"
            "tabs.forEach(function(t){"
            "t.addEventListener('click',function(){show(t.getAttribute('data-viz-panel'));});"
            "});"
            "}());"
            "</script>"
        )
    else:
        if other:
            body.append(f"<h2 class='fold-section'>Other ({len(other)} graphs)</h2>")
            body.append(
                "<p class='meta' style='margin: -0.5em 0 1em 0;'>"
                "No fold/split label (e.g. explicit --indices).</p>"
            )
            body.append("<div class='grid'>")
            for pdb_id, category, pIC50 in other:
                body.extend(_index_card_lines(pdb_id, category, pIC50))
            body.append("</div>")

    html = html_document("MPro ligand graphs — index", body, style=style)
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def _indices_plan(
    ds: MProV3Dataset,
    indices: Optional[Sequence[int]],
) -> Optional[List[PlanItem]]:
    """If ``--indices`` is used, return ``(idx, None, None)`` rows; else ``None``."""
    if not indices:
        return None
    n = len(ds)
    return [(i, None, None) for i in indices if 0 <= i < n]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw ligand graphs from a built PyG dataset (data.pt) and save images plus "
            "HTML reports under results/visualizations/."
        )
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default=None,
        help=f"Root for outputs (default: {DEFAULT_RESULTS_ROOT}); uses latest results_root/datasets/<timestamp>/, writes to results_root/visualizations/<timestamp>/.",
    )
    parser.add_argument(
        "--num-graphs-by-fold",
        type=int,
        default=None,
        dest="num_graphs_by_fold",
        metavar="N",
        help=(
            "After building the default plan, keep at most N index rows (and first-time "
            "draws) per (fold, split) bucket: train, val, and test separately per fold. "
            "Ignored when --indices is set. Default: no cap."
        ),
    )
    parser.add_argument(
        "--splits_root",
        type=str,
        default=None,
        help=f"Raw MPro snapshot with Splits/ (default: {DEFAULT_DATA_ROOT}).",
    )
    parser.add_argument(
        "--num_folds",
        type=int,
        default=DEFAULT_NUM_FOLDS,
        help=f"Number of CV folds in the split files (default: {DEFAULT_NUM_FOLDS}).",
    )
    parser.add_argument(
        "--fold_indices",
        type=int,
        nargs="+",
        default=None,
        metavar="K",
        help=(
            "Only include these CV folds in the plan (order preserved; e.g. one fold: --fold_indices 0). "
            "Default: all folds."
        ),
    )
    parser.add_argument(
        "--train_split_file",
        type=str,
        default=DEFAULT_TRAIN_SPLIT_FILE,
        help=f"Train split filename under Splits/ (default: {DEFAULT_TRAIN_SPLIT_FILE}).",
    )
    parser.add_argument(
        "--val_split_file",
        type=str,
        default=DEFAULT_VAL_SPLIT_FILE,
        help=f"Validation split filename under Splits/ (default: {DEFAULT_VAL_SPLIT_FILE}).",
    )
    parser.add_argument(
        "--test_split_file",
        type=str,
        default=DEFAULT_TEST_SPLIT_FILE,
        help=f"Test split filename under Splits/ (default: {DEFAULT_TEST_SPLIT_FILE}).",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=None,
        help="Explicit dataset indices to visualize (overrides default plan, fold filter, and --num-graphs-by-fold).",
    )
    parser.add_argument(
        "--svg",
        action="store_true",
        help="Also save vector SVG files for publication-quality figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root or DEFAULT_RESULTS_ROOT)
    dataset_dir = resolve_dataset_dir(results_root)
    dataset_base = results_root / RESULTS_DATASETS
    dataset_name = BUILT_DATASET_FOLDER_NAME

    ds = MProV3Dataset(root=str(dataset_base), dataset_name=dataset_name)
    pdb_order = load_dataset_pdb_order(dataset_base, dataset_name)

    splits_root = Path(args.splits_root or DEFAULT_DATA_ROOT)
    plan: List[PlanItem] = []
    raw = _indices_plan(ds=ds, indices=args.indices)
    if raw is not None:
        plan = raw
    else:
        fold_list = resolve_fold_indices(
            args.num_folds, fold_indices=args.fold_indices
        )
        split_plan = plan_by_fold_and_split(
            ds,
            pdb_order,
            splits_root,
            args.train_split_file,
            args.val_split_file,
            args.test_split_file,
            args.num_folds,
            fold_list,
        )
        if args.num_graphs_by_fold is not None:
            split_plan = apply_per_fold_split_cap(split_plan, args.num_graphs_by_fold)
        plan = list(split_plan)

    output_dir = results_root / RESULTS_VISUALIZATIONS
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "visualize.log"

    index_entries: List[
        Tuple[str, Optional[int], Optional[float], Optional[int], Optional[str]]
    ] = []

    with RunLogger(log_path) as log:
        log_overwrite_dir_if_nonempty(output_dir, log.log)
        log.log(f"Dataset: {dataset_dir}")
        log.log(f"Output: {output_dir}")
        log.log(
            f"Loaded {len(ds)} graphs; plan has {len(plan)} index row(s) "
            f"(splits_root={splits_root}, fold_indices={args.fold_indices}, "
            f"num_graphs_by_fold={args.num_graphs_by_fold}, indices={args.indices})"
        )

        drawn: set[int] = set()
        for idx, fold_k, split_name in plan:
            g = ds[idx]
            # x: [x, y, z, atomic_number]
            x = g.x
            pos3d = x[:, :3]
            atomic_numbers = x[:, 3].round().to(torch.long)
            edge_index = g.edge_index
            edge_attr = g.edge_attr

            pdb_id = getattr(g, "pdb_id", f"idx_{idx}")
            pdb_id_str = str(pdb_id)
            category_class = None
            if hasattr(g, "category"):
                try:
                    category_class = int(g.category.view(-1)[0].item())
                except Exception:
                    category_class = None
            # Show category in original scale (-1, 0, 1) for display
            category = (
                ORIGINAL_CATEGORY_FROM_CLASS.get(category_class, category_class)
                if category_class is not None
                else None
            )
            pIC50 = None
            if hasattr(g, "pIC50"):
                try:
                    pIC50 = float(g.pIC50.view(-1)[0].item())
                except Exception:
                    pIC50 = None

            img_filename = f"{pdb_id_str}.png"
            img_path = output_dir / img_filename
            svg_path = (output_dir / f"{pdb_id_str}.svg") if args.svg else None

            if idx not in drawn:
                draw_graph(
                    pos3d=pos3d,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    atomic_numbers=atomic_numbers,
                    out_path_png=img_path,
                    out_path_svg=svg_path,
                )

                write_html_report(
                    out_dir=output_dir,
                    image_filename=img_filename,
                    pdb_id=pdb_id_str,
                    category=category,
                    pIC50=pIC50,
                    pos3d=pos3d,
                    atomic_numbers=atomic_numbers,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    svg_filename=f"{pdb_id_str}.svg" if args.svg else None,
                )
                drawn.add(idx)

            index_entries.append((pdb_id_str, category, pIC50, fold_k, split_name))

        write_index_html(output_dir, index_entries)
        log.log(f"Index: {output_dir / 'index.html'}")
        log.log("Done.")


if __name__ == "__main__":
    main()

