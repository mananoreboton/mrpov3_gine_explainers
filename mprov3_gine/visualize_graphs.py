"""
Visualize a subset of ligand graphs from a built PyG dataset (data.pt).

Uses RDKit's 2D drawer (MolDraw2D) for publication-quality graphics. For each
selected graph this script:
- Builds an RDKit molecule from the graph and sets the full 3D conformer (x, y, z).
- Generates optimal 2D coordinates from the 3D structure via
  GenerateDepictionMatching3DStructure(), so the drawing respects 3D layout.
- Bond styles: single = one central line; double = two shifted lines;
  triple = two shifted + central; aromatic = dashed.
- Saves PNG and SVG under report/input/graphs; writes HTML reports with tables.

Usage (examples):
    uv run python visualize_graphs.py
    uv run python visualize_graphs.py --num_graphs 16
    uv run python visualize_graphs.py --pdb_ids 5R83 6LU7
    uv run python visualize_graphs.py --indices 0 1 2 3
"""

from __future__ import annotations

import argparse
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
    DEFAULT_RESULTS_ROOT,
    PYG_DATA_FILENAME,
    RESULTS_DATASETS,
    RESULTS_VISUALIZATIONS,
)
from dataset import MProV3Dataset, ORIGINAL_CATEGORY_FROM_CLASS, load_dataset_pdb_order
from utils import RunLogger, get_latest_timestamp_dir, html_document, html_escape, run_timestamp

# Image size in pixels (RDKit drawer uses this for PNG/SVG canvas).
_DRAW_SIZE = 500


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


def write_index_html(
    out_dir: Path,
    entries: List[Tuple[str, Optional[int], Optional[float]]],
) -> None:
    """
    Write an index.html page with thumbnail links to all generated graph reports.

    entries: list of (pdb_id, category, pIC50) for each graph in this run.
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
        ".timestamp { color: #666; font-size: 14px; margin-bottom: 1em; }"
    )
    body: List[str] = [
        "<h1>MPro ligand graphs</h1>",
        f"<p class='timestamp'>Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} — {len(entries)} graphs</p>",
        "<div class='grid'>",
    ]
    for pdb_id, category, pIC50 in entries:
        safe_id = html_escape(pdb_id)
        img_src = html_escape(f"{pdb_id}.png")
        report_href = html_escape(f"{pdb_id}.html")
        meta_parts = []
        if category is not None:
            meta_parts.append(f"Cat. {category}")
        if pIC50 is not None:
            meta_parts.append(f"pIC50 {pIC50:.2f}")
        meta_str = html_escape(" · ".join(meta_parts)) if meta_parts else ""
        body.append("<div class='card'>")
        body.append(f"  <a href='{report_href}'>")
        body.append(f"    <img src='{img_src}' alt='{safe_id}' loading='lazy' />")
        body.append(f"    <span class='label'>{safe_id}</span>")
        if meta_str:
            body.append(f"    <span class='meta'>{meta_str}</span>")
        body.append("  </a>")
        body.append("</div>")
    body.append("</div>")
    html = html_document("MPro ligand graphs — index", body, style=style)
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def _select_indices_from_args(
    ds: MProV3Dataset,
    pdb_order: Optional[Sequence[str]],
    num_graphs: int,
    indices: Optional[Sequence[int]],
    pdb_ids: Optional[Sequence[str]],
) -> List[int]:
    """Resolve which dataset indices to visualize based on CLI arguments."""
    n = len(ds)
    if indices:
        return [i for i in indices if 0 <= i < n]

    if pdb_ids:
        if pdb_order is None:
            raise ValueError(
                "pdb_ids were provided but pdb_order.txt is missing. "
                "Please ensure the dataset was built with build_dataset.py."
            )
        mapping = {p: idx for idx, p in enumerate(pdb_order)}
        result: List[int] = []
        for p in pdb_ids:
            if p in mapping:
                result.append(mapping[p])
        return result

    # Default: first num_graphs graphs.
    return list(range(min(num_graphs, n)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw a subset of ligand graphs from a built PyG dataset (data.pt) and "
            "save images plus HTML reports to results/visualizations/."
        )
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default=None,
        help=f"Root for outputs (default: {DEFAULT_RESULTS_ROOT}); uses latest results_root/datasets/<timestamp>/, writes to results_root/visualizations/<timestamp>/.",
    )
    parser.add_argument(
        "--num_graphs",
        type=int,
        default=16,
        help="Number of graphs to visualize if --indices/--pdb_ids are not provided (default: 16).",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=None,
        help="Explicit dataset indices to visualize (overrides --num_graphs).",
    )
    parser.add_argument(
        "--pdb_ids",
        type=str,
        nargs="+",
        default=None,
        help="PDB IDs to visualize (requires pdb_order.txt built by build_dataset.py).",
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
    dataset_base = results_root / RESULTS_DATASETS
    latest_dataset = get_latest_timestamp_dir(dataset_base)
    if latest_dataset is None or not (latest_dataset / PYG_DATA_FILENAME).exists():
        raise FileNotFoundError(
            f"No dataset found under {dataset_base}. Run build_dataset.py with --results_root {results_root} first."
        )
    dataset_name = latest_dataset.name

    ds = MProV3Dataset(root=str(dataset_base), dataset_name=dataset_name)
    pdb_order = load_dataset_pdb_order(dataset_base, dataset_name)

    selected_indices = _select_indices_from_args(
        ds=ds,
        pdb_order=pdb_order,
        num_graphs=args.num_graphs,
        indices=args.indices,
        pdb_ids=args.pdb_ids,
    )

    ts = run_timestamp()
    output_dir = results_root / RESULTS_VISUALIZATIONS / ts
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "visualize.log"

    index_entries: List[Tuple[str, Optional[int], Optional[float]]] = []

    with RunLogger(log_path) as log:
        log.log(f"Dataset: {dataset_base / dataset_name} (latest)")
        log.log(f"Output: {output_dir}")
        log.log(f"Loaded {len(ds)} graphs; writing {len(selected_indices)} graphs")

        for idx in selected_indices:
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
            index_entries.append((pdb_id_str, category, pIC50))

        write_index_html(output_dir, index_entries)
        log.log(f"Index: {output_dir / 'index.html'}")
        log.log("Done.")


if __name__ == "__main__":
    main()

