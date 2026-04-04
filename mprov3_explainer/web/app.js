/**
 * Single-page exploration UI: reads report_data.json from the same directory
 * (requires HTTP server). Routes use the hash, e.g. #/classification or
 * #/fold/0/explainer/GNNEXPL/graph/7GCK
 */

(function () {
  "use strict";

  /** @type {object|null} Full payload from report_data.json */
  let data = null;

  /**
   * Escape text for safe insertion into HTML.
   * @param {string} s
   */
  function esc(s) {
    const d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
  }

  /**
   * Current hash route as path segments (no leading # or slashes).
   * @returns {string[]}
   */
  function routeParts() {
    const h = window.location.hash.replace(/^#\/?/, "").trim();
    if (!h) return [];
    return h.split("/").filter(Boolean);
  }

  /**
   * Build top nav links for the active view.
   * @param {string[]} parts
   */
  function renderNav(parts) {
    const nav = document.getElementById("main-nav");
    if (!nav) return;
    const links = [
      ["Home", "#/"],
      ["Train / classification", "#/classification"],
    ];
    let html = "";
    for (const [label, href] of links) {
      html += `<a href="${href}">${esc(label)}</a>`;
    }
    if (data && data.meta && data.meta.fold_indices) {
      for (const fid of data.meta.fold_indices) {
        html += `<a href="#/fold/${esc(fid)}">Fold ${esc(fid)}</a>`;
      }
    }
    if (data && data.meta && data.meta.explainers) {
      for (const ex of data.meta.explainers) {
        html += `<a href="#/explainer/${esc(ex)}">${esc(ex)}</a>`;
      }
    }
    nav.innerHTML = html;
  }

  /**
   * @param {string} foldId
   * @returns {object|undefined}
   */
  function foldBlock(foldId) {
    return data && data.folds && data.folds[foldId];
  }

  /**
   * Find classification row for a PDB inside a fold.
   * @param {string} foldId
   * @param {string} pdbId
   */
  function classRowForPdb(foldId, pdbId) {
    const fb = foldBlock(foldId);
    if (!fb || !fb.classification || !fb.classification.graphs) return null;
    return fb.classification.graphs.find((g) => g.pdb_id === pdbId) || null;
  }

  /** Home: links to major sections */
  function viewHome() {
    const m = data.meta;
    const nf = m.num_folds;
    const ex = (m.explainers || []).join(", ");
    return `
      <div class="card">
        <h2>Overview</h2>
        <p><strong>CV folds (reported)</strong>: ${esc(String(m.fold_indices.length))} fold(s); num_folds in data = ${esc(String(nf))}</p>
        <p><strong>Explainers</strong>: ${esc(ex || "—")}</p>
        <p><strong>Generated</strong>: ${esc(m.generated_at || "—")}</p>
        <ul>
          <li><a href="#/classification">Train / classification results by fold</a></li>
        </ul>
        <h3 class="section-title">Folds</h3>
        <ul>
          ${(m.fold_indices || [])
            .map(
              (fid) =>
                `<li><a href="#/fold/${esc(fid)}">Fold ${esc(fid)} of ${esc(String(nf))}</a> — metrics by explainer, links to each explainer slice</li>`,
            )
            .join("")}
        </ul>
        <h3 class="section-title">Explainers</h3>
        <ul>
          ${(m.explainers || [])
            .map(
              (e) =>
                `<li><a href="#/explainer/${esc(e)}">${esc(e)}</a> — metrics by fold, links to each fold slice</li>`,
            )
            .join("")}
        </ul>
      </div>
      <div class="card">
        <h2>GNN (training / evaluation settings)</h2>
        <pre class="raw-mask" style="max-height:none">${esc(JSON.stringify(m.gnn, null, 2))}</pre>
      </div>`;
  }

  /** Classification: GNN block + per-fold accuracy + graph list */
  function viewClassification() {
    const m = data.meta;
    let body = `
      <div class="card">
        <h2>GNN type and parameters</h2>
        <pre class="raw-mask" style="max-height:none">${esc(JSON.stringify(m.gnn, null, 2))}</pre>
      </div>`;

    for (const fid of m.fold_indices || []) {
      const fb = foldBlock(fid);
      if (!fb) continue;
      const c = fb.classification || {};
      const acc =
        c.accuracy != null ? Number(c.accuracy).toFixed(4) : "—";
      body += `<div class="card"><h2>Fold ${esc(fid)} — classification</h2>`;
      body += `<p><strong>Test accuracy</strong>: ${esc(acc)}</p>`;
      if (c.evaluation_timestamp) {
        body += `<p class="tagline">Eval run: ${esc(c.evaluation_timestamp)}</p>`;
      }
      body += '<div class="grid">';
      for (const g of c.graphs || []) {
        const ok = g.correct ? "correct" : "wrong";
        const lbl = g.correct ? "correct" : "incorrect";
        body += `<div class="thumb">
          <a href="#/fold/${esc(fid)}/graph/${esc(g.pdb_id)}">
            <img src="${esc(g.base_image)}" alt="${esc(g.pdb_id)}" loading="lazy" />
          </a>
          <div class="caption">
            <strong>${esc(g.pdb_id)}</strong><br/>
            real ${esc(String(g.real_category))} → pred ${esc(String(g.predicted_category))}
            <span class="${ok}">${esc(lbl)}</span>
          </div>
        </div>`;
      }
      body += "</div></div>";
    }
    return body;
  }

  /**
   * Summary table: explainer metrics for one fold.
   * @param {string} foldId
   */
  function tableMetricsByExplainer(foldId) {
    const rows = (data.summary_by_fold && data.summary_by_fold[foldId]) || [];
    if (!rows.length) return "<p>No explainer rows for this fold.</p>";
    let h =
      "<table class='metrics-table'><thead><tr><th>Explainer</th><th class='num'>fid+</th><th class='num'>fid−</th><th class='num'>char</th><th class='num'>Fsuf</th><th class='num'>Fcom</th><th class='num'>Ff1</th><th class='num'>n</th></tr></thead><tbody>";
    for (const r of rows) {
      h += `<tr>
        <td><a href="#/fold/${esc(foldId)}/explainer/${esc(r.explainer)}">${esc(r.explainer)}</a></td>
        <td class="num">${esc(fmt(r.mean_fidelity_plus))}</td>
        <td class="num">${esc(fmt(r.mean_fidelity_minus))}</td>
        <td class="num">${esc(fmt(r.mean_pyg_characterization))}</td>
        <td class="num">${esc(fmt(r.mean_paper_sufficiency))}</td>
        <td class="num">${esc(fmt(r.mean_paper_comprehensiveness))}</td>
        <td class="num">${esc(fmt(r.mean_paper_f1_fidelity))}</td>
        <td class="num">${esc(String(r.num_graphs))}</td>
      </tr>`;
    }
    h += "</tbody></table>";
    return h;
  }

  /**
   * Summary table: one explainer across folds.
   * @param {string} explainerId
   */
  function tableMetricsByFold(explainerId) {
    const rows =
      (data.summary_by_explainer && data.summary_by_explainer[explainerId]) || [];
    if (!rows.length) return "<p>No rows for this explainer.</p>";
    let h =
      "<table class='metrics-table'><thead><tr><th>Fold</th><th class='num'>fid+</th><th class='num'>fid−</th><th class='num'>char</th><th class='num'>Fsuf</th><th class='num'>Fcom</th><th class='num'>Ff1</th><th class='num'>n</th></tr></thead><tbody>";
    for (const r of rows) {
      h += `<tr>
        <td><a href="#/fold/${esc(r.fold)}/explainer/${esc(explainerId)}">${esc(r.fold)}</a></td>
        <td class="num">${esc(fmt(r.mean_fidelity_plus))}</td>
        <td class="num">${esc(fmt(r.mean_fidelity_minus))}</td>
        <td class="num">${esc(fmt(r.mean_pyg_characterization))}</td>
        <td class="num">${esc(fmt(r.mean_paper_sufficiency))}</td>
        <td class="num">${esc(fmt(r.mean_paper_comprehensiveness))}</td>
        <td class="num">${esc(fmt(r.mean_paper_f1_fidelity))}</td>
        <td class="num">${esc(String(r.num_graphs))}</td>
      </tr>`;
    }
    h += "</tbody></table>";
    return h;
  }

  function fmt(x) {
    if (x == null || Number.isNaN(x)) return "—";
    return Number(x).toFixed(4);
  }

  /** Fold index: links to explainer slices + summary table */
  function viewFold(foldId) {
    const fb = foldBlock(foldId);
    if (!fb) return `<p>Unknown fold <code>${esc(foldId)}</code></p>`;
    const nf = fb.num_folds != null ? fb.num_folds : data.meta.num_folds;
    let h = `<div class="card"><h2>Fold ${esc(foldId)} of ${esc(String(nf))}</h2>`;
    h += "<h3 class='section-title'>Results per explainer</h3><ul>";
    for (const ex of data.meta.explainers || []) {
      const slice = fb.explainers && fb.explainers[ex];
      if (!slice) continue;
      h += `<li><a href="#/fold/${esc(foldId)}/explainer/${esc(ex)}">${esc(ex)}</a></li>`;
    }
    h += "</ul>";
    h += "<h3 class='section-title'>Metrics by explainer</h3>";
    h += tableMetricsByExplainer(foldId);
    h += "</div>";
    return h;
  }

  /** Explainer index: links to each fold + summary by fold */
  function viewExplainer(explainerId) {
    let h = `<div class="card"><h2>Explainer ${esc(explainerId)}</h2>`;
    h += "<h3 class='section-title'>Results per fold</h3><ul>";
    for (const fid of data.meta.fold_indices || []) {
      const fb = foldBlock(fid);
      if (!fb || !fb.explainers || !fb.explainers[explainerId]) continue;
      h += `<li><a href="#/fold/${esc(fid)}/explainer/${esc(explainerId)}">Fold ${esc(fid)}</a></li>`;
    }
    h += "</ul>";
    h += "<h3 class='section-title'>Metrics by fold</h3>";
    h += tableMetricsByFold(explainerId);
    h += "</div>";
    return h;
  }

  /**
   * Fold + explainer: list graphs with thumbnails linking to graph detail.
   */
  function viewFoldExplainer(foldId, explainerId) {
    const fb = foldBlock(foldId);
    if (!fb || !fb.explainers || !fb.explainers[explainerId]) {
      return `<p>No data for fold ${esc(foldId)} / ${esc(explainerId)}</p>`;
    }
    const graphs = fb.explainers[explainerId].graphs || {};
    const ids = Object.keys(graphs).sort();
    let h = `<div class="card"><h2>Fold ${esc(foldId)} — ${esc(explainerId)}</h2>`;
    h += '<div class="grid">';
    for (const gid of ids) {
      const g = graphs[gid];
      const href = `#/fold/${esc(foldId)}/explainer/${esc(explainerId)}/graph/${esc(gid)}`;
      h += `<div class="thumb">
        <a href="${href}"><img src="${esc(g.mask_image)}" alt="${esc(gid)}" loading="lazy" /></a>
        <div class="caption"><a href="${href}">${esc(gid)}</a><br/>
        fid+ ${esc(fmt(g.fidelity_plus))} / Ff1 ${esc(fmt(g.paper_f1_fidelity))}
        </div>
      </div>`;
    }
    h += "</div></div>";
    return h;
  }

  /**
   * Single graph: PDB, base 2D image, classification, then each explainer mask + raw JSON.
   */
  function viewGraph(foldId, explainerId, graphId) {
    const fb = foldBlock(foldId);
    if (!fb) return `<p>Unknown fold.</p>`;
    const cr = classRowForPdb(foldId, graphId);
    const baseImg = cr && cr.base_image ? cr.base_image : "";

    const ctx =
      explainerId && explainerId.length
        ? `fold ${esc(foldId)}, context ${esc(explainerId)}`
        : `fold ${esc(foldId)}`;
    let h = `<div class="card"><h2>Graph ${esc(graphId)} <span class="tagline">(${ctx})</span></h2>`;

    if (baseImg) {
      h += `<p><img src="${esc(baseImg)}" alt="2D ligand" style="max-width:360px;height:auto;border:1px solid #ccc;border-radius:6px" /></p>`;
    }

    if (cr) {
      const ok = cr.correct ? "correct" : "wrong";
      h += `<p><strong>Classification</strong>: real ${esc(String(cr.real_category))} → pred ${esc(String(cr.predicted_category))} <span class="${ok}">${cr.correct ? "correct" : "incorrect"}</span></p>`;
    } else {
      h += "<p><em>No classification row for this PDB in this fold.</em></p>";
    }

    h += "<h3 class='section-title'>Graph + mask by explainer (this fold)</h3>";

    for (const ex of data.meta.explainers || []) {
      const block = fb.explainers && fb.explainers[ex];
      const g = block && block.graphs && block.graphs[graphId];
      if (!g) continue;
      h += `<div class="graph-detail-explainer"><h4>${esc(ex)}</h4>`;
      h += `<p><img src="${esc(g.mask_image)}" alt="mask ${esc(ex)}" style="max-width:360px;height:auto;border:1px solid #ccc;border-radius:6px" /></p>`;
      h += "<p><strong>Per-graph metrics</strong>: fid+ " + esc(fmt(g.fidelity_plus)) + ", fid− " + esc(fmt(g.fidelity_minus)) + ", Ff1 " + esc(fmt(g.paper_f1_fidelity)) + "</p>";
      h += "<p><strong>Explanation raw values</strong> (edge_index / masks):</p>";
      h += `<pre class="raw-mask">${esc(JSON.stringify(g.mask_raw, null, 2))}</pre>`;
      h += "</div>";
    }

    h += "</div>";
    return h;
  }

  /**
   * Dispatch from hash segments to a view.
   * @param {string[]} parts
   */
  function dispatch(parts) {
    if (parts.length === 0) return viewHome();
    if (parts[0] === "classification") return viewClassification();
    if (parts[0] === "fold" && parts.length === 2) return viewFold(parts[1]);
    if (parts[0] === "explainer" && parts.length === 2) return viewExplainer(parts[1]);
    if (
      parts[0] === "fold" &&
      parts[2] === "explainer" &&
      parts.length === 4
    ) {
      return viewFoldExplainer(parts[1], parts[3]);
    }
    if (
      parts[0] === "explainer" &&
      parts[2] === "fold" &&
      parts.length === 4
    ) {
      return viewFoldExplainer(parts[3], parts[1]);
    }
    if (parts[0] === "fold" && parts[2] === "graph" && parts.length === 4) {
      return viewGraph(parts[1], "", parts[3]);
    }
    if (
      parts[0] === "fold" &&
      parts[2] === "explainer" &&
      parts[4] === "graph" &&
      parts.length === 6
    ) {
      return viewGraph(parts[1], parts[3], parts[5]);
    }
    return `<p>Unrecognized route. <a href="#/">Home</a></p>`;
  }

  /** Re-render main content from current hash */
  function render() {
    const app = document.getElementById("app");
    if (!app || !data) return;
    const parts = routeParts();
    renderNav(parts);
    app.innerHTML = dispatch(parts);
  }

  /** Load JSON then first render */
  async function load() {
    const app = document.getElementById("app");
    try {
      const res = await fetch("report_data.json", { cache: "no-store" });
      if (!res.ok) throw new Error(res.status + " " + res.statusText);
      data = await res.json();
      window.addEventListener("hashchange", render);
      render();
    } catch (e) {
      if (app) {
        app.innerHTML =
          "<p class='wrong'>Could not load report_data.json. Serve this folder over HTTP (e.g. python generate_visualizations.py).</p><pre>" +
          esc(String(e)) +
          "</pre>";
      }
    }
  }

  load();
})();
