"""Three composite, full-width figures for the v2 preprint (Nature-style lettered panels).
Usage: python fig_story2.py <agg_dir> <results_dir> <audit_data_dir> <out_dir>"""
import sys, json, glob, logging, numpy as np, pandas as pd, matplotlib
logging.getLogger("fontTools").setLevel(logging.ERROR); matplotlib.use("Agg")
import matplotlib.pyplot as plt; from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.colors import LinearSegmentedColormap, Normalize; from scipy.stats import gaussian_kde
A, R, D, OUT = sys.argv[1:5]
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
COL = {"eegpt": "#0072B2", "biot": "#E69F00", "labram": "#009E73"}; NAME = {"eegpt": "EEGPT", "biot": "BIOT", "labram": "LaBraM"}
FAIL, GREY, INK = "#D55E00", "#9a9a9a", "#222222"
def ns(v):
    try: return f"{float(v):g}"
    except ValueError: return str(v)
BB = [b for b in ["eegpt", "labram", "biot"] if glob.glob(f"{A}/table_{b}.csv")]
T = {b: pd.read_csv(f"{A}/table_{b}.csv", dtype={"severity": str}) for b in BB}
P = {}
for b in BB:
    try: p = pd.read_csv(f"{R}/per_recording_predictions_{b}_mlp_s7.csv", dtype={"severity": str}); p["severity"] = p.severity.map(ns); P[b] = p
    except FileNotFoundError: pass
def base(b): r = T[b][(T[b].axis=="sampling_rate")&(T[b].severity=="250")&(T[b].arm=="canonicalized")].iloc[0]; return float(r.auroc_mean)
def cell(b, ax, sv, arm):
    g = T[b][(T[b].axis==ax)&(T[b].severity==sv)&(T[b].arm==arm)]
    return (float(g.auroc_mean.iloc[0]), float(g.auroc_min.iloc[0]), float(g.auroc_max.iloc[0])) if len(g) else (np.nan, np.nan, np.nan)
def letter(fig, x, y, s): fig.text(x, y, s, fontsize=9, fontweight="bold", va="top", ha="left")
CELLS = [("sampling_rate", "A1 rate (Hz)", ["128","200","256","512"]), ("calibration", "A2 gain (×)", ["0.5","0.75","1.5","2"]),
         ("power_line", "A3 mains (µV)", ["5","15","30","60"]), ("montage", "A4 montage", ["legacy16","bipolar18"]), ("broadband", "A5 SNR (dB)", ["-5","0","5","10","20"])]
SHORT = lambda v: v.replace("legacy16","leg16").replace("bipolar18","bip18")

# ================================================================= Figure 1: disciplined schematic (two rows, shared geometry)
from matplotlib.patches import Arc, Polygon
INK1, INK2 = "#333333", "#777777"; BLUE, BLUEF = "#0072B2", "#EAF3FA"; ORG, ORGF = "#B26A00", "#FFF6E5"; GRYF = "#F3F3F3"
fig = plt.figure(figsize=(3.5, 1.15), dpi=300)
def mk(rect, y0=0, y1=20): ax = fig.add_axes(rect); ax.set_axis_off(); ax.set_xlim(0, 100); ax.set_ylim(y0, y1); return ax
def rr(ax, x, y, w, h, fc=GRYF, ec=INK2, lw=0.5): ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.9", fc=fc, ec=ec, lw=lw))
def arrow(ax, x0, x1, y): ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=4.5, lw=0.55, color=INK1, shrinkA=0, shrinkB=0))
def recorders(ax, x, y, w, h):
    for k in range(3):
        dx, dy = k*1.6, (2-k)*1.6; rr(ax, x+dx, y+dy, w-3.2, h-3.2, fc="white" if k < 2 else GRYF, ec=INK2, lw=0.45)
    gx = np.linspace(x+3.2+2.0, x+w-2.6, 4); gy = np.linspace(y+2.0, y+h-4.8, 3)
    for gxx in gx:
        for gyy in gy: ax.plot([gxx], [gyy], "o", ms=1.1, color=INK1)
def esf(ax, x, y, w, h, off=None, label=True):
    rr(ax, x, y, w, h, fc=BLUEF, ec=BLUE, lw=0.6)
    for k in range(6):
        yy = y + h - (k+1)*h/7.2; c = FAIL if off == k else BLUE
        ax.plot([x+0.16*w, x+0.90*w], [yy, yy], color=c, lw=1.1 if off == k else 0.8, solid_capstyle="round", ls=(0,(1.2,1.2)) if off == k else "-")
        if label: ax.text(x+0.06*w, yy, f"C{k+1}", fontsize=3.2, color=INK2, va="center", ha="left")
def backbone(ax, x, y, w, h):
    rr(ax, x, y, w, h, fc=GRYF, ec=INK2, lw=0.5)
    for k in range(3): yy = y + h*(0.2 + k*0.25); ax.add_patch(FancyBboxPatch((x+0.14*w, yy), 0.72*w, h*0.14, boxstyle="round,pad=0,rounding_size=0.4", fc="white", ec=INK2, lw=0.4))
    lx, ly = x+w-3.0, y+h-3.0; ax.add_patch(Rectangle((lx-1.0, ly-1.1), 2.0, 1.4, fc=INK1, ec="none")); ax.add_patch(Arc((lx, ly+0.3), 1.3, 1.5, theta1=0, theta2=180, lw=0.5, color=INK1))
def gauge(ax, x, y, r, delta=False):
    ax.add_patch(Arc((x, y), 2*r, 2*r, theta1=0, theta2=180, lw=0.6, color=INK1)); ax.plot([x-r, x+r], [y, y], color=INK1, lw=0.5)
    ang = np.deg2rad(40); ax.plot([x, x+0.85*r*np.cos(ang)], [y, y+0.85*r*np.sin(ang)], color=FAIL, lw=0.7); ax.plot([x], [y], "o", ms=1.4, color=INK1)
    ax.text(x, y-2.6, "ΔAUROC" if delta else "AUROC", fontsize=4.2, ha="center", va="top", color=INK1)
# row a
ax = mk([0.035, 0.52, 0.955, 0.47], y0=-4.5, y1=20); fig.text(0.005, 0.97, "a", fontsize=7, fontweight="bold", va="top")
recorders(ax, 1, 2, 22, 17); ax.text(12, -1.0, "125–512 Hz", fontsize=3.8, ha="center", va="top", color=INK2)
arrow(ax, 24, 29, 10.5); esf(ax, 30, 2, 22, 17); ax.text(41, -1.0, "ESF", fontsize=4.2, ha="center", va="top", color=BLUE, fontweight="bold")
arrow(ax, 53, 58, 10.5); backbone(ax, 59, 2, 22, 17); ax.text(70, -1.0, "frozen backbone", fontsize=3.8, ha="center", va="top", color=INK2)
arrow(ax, 82, 87, 10.5); gauge(ax, 93.5, 8.5, 5.5)
# row b
bx = mk([0.035, 0.02, 0.955, 0.48], y0=-3.5, y1=23.5); fig.text(0.005, 0.49, "b", fontsize=7, fontweight="bold", va="top")
rr(bx, 1, 4.5, 20, 12, fc=ORGF, ec=ORG, lw=0.6); bx.text(11, 12.2, "shift", fontsize=4.2, ha="center", va="center", color=ORG, fontweight="bold"); bx.text(11, 8.0, "A1–A5", fontsize=4.0, ha="center", va="center", color=ORG)
bx.add_patch(FancyArrowPatch((21.2, 12.5), (29.4, 16.0), arrowstyle="-|>", mutation_scale=4.5, lw=0.55, color=INK1)); bx.add_patch(FancyArrowPatch((21.2, 8.5), (29.4, 5.0), arrowstyle="-|>", mutation_scale=4.5, lw=0.55, color=INK1))
esf(bx, 30, 11.2, 22, 8.6, label=False); esf(bx, 30, 0.6, 22, 8.6, off=2, label=False)
bx.text(41, 20.6, "canonicalized", fontsize=3.8, va="bottom", ha="center", color=BLUE); bx.text(41, -0.2, "naive", fontsize=3.8, va="top", ha="center", color=FAIL)
bx.add_patch(FancyArrowPatch((52.4, 15.5), (69.0, 12.0), arrowstyle="-|>", mutation_scale=4.5, lw=0.55, color=INK1)); bx.add_patch(FancyArrowPatch((52.4, 4.9), (69.0, 8.4), arrowstyle="-|>", mutation_scale=4.5, lw=0.55, color=INK1))
backbone(bx, 69.5, 2, 18, 17); arrow(bx, 88, 90.5, 10.5); gauge(bx, 95.5, 8.5, 4.5, delta=True)
fig.savefig(f"{OUT}/fig1_overview.pdf"); plt.close(fig); print("fig1")

# ================================================================= Figure 2: every cell + why (bandwidth)
nb = len(BB)
labels, ys, yi = [], [], 0
for ax, ttl, sevs in CELLS:
    for sv in sevs: labels.append((ax, sv, f"{ttl.split(' (')[0].replace('native rate','rate').replace('gain ×g','gain')} {SHORT(sv)}")); ys.append(yi); yi += 1
    yi += 0.6
fig = plt.figure(figsize=(7.16, 3.6), dpi=300)
gsL = fig.add_gridspec(1, nb, wspace=0.12, left=0.115, right=0.60, top=0.92, bottom=0.17)
gsR = fig.add_gridspec(2, 1, hspace=0.45, left=0.70, right=0.99, top=0.92, bottom=0.17)
dax = [fig.add_subplot(gsL[0, j]) for j in range(nb)]
for j, b in enumerate(BB):
    a = dax[j]; bl = base(b); a.axvspan(bl - 0.01, bl + 0.01, color=COL[b], alpha=0.12, lw=0); a.axvline(bl, color=COL[b], lw=0.7, ls=(0,(3,2)))
    for (ax, sv, lab), y in zip(labels, ys):
        xc, lo, hi = cell(b, ax, sv, "canonicalized"); xn = cell(b, ax, sv, "naive")[0]
        if np.isnan(xc): continue
        if not np.isnan(xn) and abs(xn - xc) > 1e-6: a.annotate("", (xc, y), (xn, y), arrowprops=dict(arrowstyle="-|>", lw=0.8, color=GREY, mutation_scale=6, shrinkA=2, shrinkB=2))
        if not np.isnan(xn): a.plot([xn], [y], "o", ms=3.3, mfc="white", mec=GREY, mew=0.9, zorder=3)
        a.plot([lo, hi], [y, y], color=COL[b], lw=2.2, alpha=0.35, zorder=2); a.plot([xc], [y], "s", ms=3.3, color=COL[b], zorder=4)
    a.set_xlim(0.38, 0.95); a.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9]); a.grid(True, axis="x", lw=0.3, ls=":", color="#bbbbbb"); a.set_title(NAME[b], color=COL[b], fontweight="bold", loc="left", pad=3); a.set_xlabel("AUROC, TUAB eval")
    a.set_yticks(ys); a.tick_params(axis="y", length=0)
    if j == 0: a.set_yticklabels([l for _, _, l in labels])
    else: a.set_yticklabels([])
    a.set_ylim(max(ys) + 0.8, -0.8)
letter(fig, 0.005, 0.99, "a"); fig.text(0.03, 0.99, "Per cell", fontsize=7.5, va="top")
# bandwidth panels
b1 = fig.add_subplot(gsR[0, 0]); b2 = fig.add_subplot(gsR[1, 0], sharex=b1); cuts = [15,20,25,30,40,50,64,80,100]
for b in BB:
    g = T[b][(T[b].axis=="bandwidth")&(T[b].arm=="canonicalized")].copy()
    if len(g)==0: continue
    g["cut"] = g.severity.astype(float); g = g.sort_values("cut")
    b1.fill_between(g.cut, g.auroc_min, g.auroc_max, color=COL[b], alpha=0.25, lw=0); b1.plot(g.cut, g.auroc_mean, "-", marker="s", ms=2.8, lw=1.0, color=COL[b], label=NAME[b]); b1.axhline(base(b), color=COL[b], lw=0.5, ls=(0,(3,2)), alpha=0.8)
    b2.plot(g.cut, g.bal_acc, "-", marker="s", ms=2.8, lw=1.0, color=COL[b], label=f"{NAME[b]} bal. acc."); b2.plot(g.cut, g.ece, ":", marker="^", ms=2.8, lw=1.0, color=COL[b], label=f"{NAME[b]} ECE")
if "biot" in T:
    y = cell("biot", "sampling_rate", "128", "canonicalized")[0]; b1.axhline(y, color=COL["biot"], lw=0.5, ls=(0,(1,1)))
    b1.text(25, y - 0.012, "BIOT, 128 Hz cell", fontsize=6, color=COL["biot"], va="top")
for a in (b1, b2): a.set_xscale("log"); a.set_xticks(cuts); a.set_xticklabels([str(c) for c in cuts], fontsize=6); a.minorticks_off(); a.grid(True, lw=0.3, ls=":", color="#bbbbbb")
b1.set_ylim(0.45, 0.97); b1.set_ylabel("AUROC, canonicalized arm"); b1.legend(loc="lower right", frameon=False, handlelength=1.6); plt.setp(b1.get_xticklabels(), visible=False)
b2.set_ylim(0, 1.3); b2.set_yticks([0, .2, .4, .6, .8, 1.0]); b2.set_ylabel("balanced accuracy / ECE"); b2.set_xlabel("low-pass cutoff (Hz)"); b2.legend(loc="upper center", frameon=False, ncol=2, handlelength=1.5, columnspacing=0.8, bbox_to_anchor=(0.5, 1.02))
letter(fig, 0.635, 0.99, "b"); fig.text(0.66, 0.99, "Low-pass: ranking", fontsize=7.5, va="top")
pos2 = b2.get_position(); letter(fig, 0.635, pos2.y1 + 0.045, "c"); fig.text(0.66, pos2.y1 + 0.045, "Low-pass: calibration", fontsize=7.5, va="top")
fig.text(0.115, 0.012, "open circle: naive arm   square: canonicalized arm   dashed: clean baseline   shaded: ±0.01", fontsize=6.3, va="bottom")
fig.savefig(f"{OUT}/fig2_cells_bandwidth.pdf"); plt.close(fig); print("fig2")

# ================================================================= Figure 3: recordings, legacy recorder, taxonomy (condensed grid)
L = json.load(open(f"{R}/legacy.json")); pair = [b for b in ["eegpt", "biot"] if b in P]
CATS = [("robust_correct","robust-correct","#4d4d4d"), ("shift_recovered","shift-recovered","#a6a6a6"), ("calibration_collapse","calibration collapse","#c7b299"),
        ("axis_specific","axis-specific","#999933"), ("shift_induced_flip","shift-induced flip","#e6a15a"), ("catastrophic_shift","catastrophic shift","#D55E00"), ("robust_wrong","robust-wrong","#e0e0e0")]
trows = []
for b, suf in [("eegpt",""), ("labram","_labram"), ("biot","_biot")]:
    for arm in ["naive","canonicalized"]:
        f = f"{D}/per_recording_failure_profile{suf}_{arm}.csv"
        if glob.glob(f): trows.append((b, arm, pd.read_csv(f)))
fig = plt.figure(figsize=(7.16, 2.9), dpi=300)
gs = fig.add_gridspec(2, 5, width_ratios=[0.78, 0.78, 0.12, 1.25, 1.25], wspace=0.4, hspace=0.6, left=0.06, right=0.985, top=0.90, bottom=0.20)
lim = (-12, 16); xs = np.linspace(-12, 16, 400)
for j, b in enumerate(pair):
    p = P[b]; b0 = p[(p.axis=="sampling_rate")&(p.severity=="250")&(p.arm=="canonicalized")].set_index("rec_id"); b1_ = p[(p.axis=="sampling_rate")&(p.severity=="128")&(p.arm=="canonicalized")].set_index("rec_id")
    i = b0.index.intersection(b1_.index); x, y, lab = b0.loc[i].logit.values, b1_.loc[i].logit.values, b0.loc[i].label.values
    a = fig.add_subplot(gs[0, j]); a.plot(lim, lim, color=GREY, lw=0.6, ls=(0,(3,2)), zorder=1)
    a.scatter(x[lab==1], y[lab==1], s=4, color=COL[b], alpha=0.55, lw=0, label="abnormal", zorder=2); a.scatter(x[lab==0], y[lab==0], s=4, facecolors="white", edgecolors=COL[b], lw=0.45, label="normal", zorder=3)
    a.set_xlim(lim); a.set_ylim(lim); a.set_xticks([-10, 0, 10]); a.set_yticks([-10, 0, 10]); a.set_box_aspect(1)
    a.set_title(f"{NAME[b]}   normals {np.median(x[lab==0]):+.1f} $\\rightarrow$ {np.median(y[lab==0]):+.1f}", loc="left", fontsize=6.3, color=COL[b], pad=3)
    a.set_xlabel("logit, clean", labelpad=1)
    if j == 0: a.set_ylabel("logit, 128 Hz", labelpad=1); a.legend(loc="lower right", frameon=False, handletextpad=0.2, markerscale=1.4, borderpad=0.0, fontsize=6)
    # legacy row
    a = fig.add_subplot(gs[1, j]); kn, ka = gaussian_kde(x[lab==0]), gaussian_kde(x[lab==1])
    a.plot(xs, kn(xs), color=COL[b], lw=0.9); a.fill_between(xs, 0, ka(xs), color=COL[b], alpha=0.5, lw=0); a.plot(xs, ka(xs), color=COL[b], lw=0.9)
    ymax = max(kn(xs).max(), ka(xs).max()) * 1.3
    if b == "biot": k = gaussian_kde(y[lab==0])(xs); a.plot(xs, k, color=FAIL, lw=0.9, ls=(0,(2,1.5))); a.text(0.99, 0.95, "normals\nat 128 Hz", transform=a.transAxes, fontsize=5.6, color=FAIL, ha="right", va="top")
    pn, pa = xs[np.argmax(kn(xs))], xs[np.argmax(ka(xs))]; a.text(pn, kn(pn)[0]*1.04, "normal", fontsize=5.6, ha="center", va="bottom"); a.text(pa + 2.0, ka(pa)[0]*1.02, "abnormal", fontsize=5.6, ha="left", va="bottom")
    lg = np.array(L["backbones"][b]["legacy_logits"]); a.vlines(lg, 0, ymax*0.14, color=INK, lw=1.0); a.text(np.mean(lg), -ymax*0.04, "legacy ×6", fontsize=5.6, ha="center", va="top")
    a.set_ylim(-ymax*0.22, ymax); a.set_yticks([]); a.set_xlim(lim); a.set_xticks([-10, 0, 10]); a.set_xlabel("logit, clean", labelpad=1); a.spines["left"].set_visible(False)
# taxonomy, right half
for k, (cls, ttl) in enumerate([(0, "ground-truth normal"), (1, "ground-truth abnormal")]):
    a = fig.add_subplot(gs[k, 3:])
    for i, (b, arm, p) in enumerate(trows):
        q = p[p.label==cls]; n = len(q); left = 0
        for key, lab_, colr in CATS:
            w = (q.category==key).sum()/n if n else 0
            a.barh(i, w, left=left, color=colr, height=0.72, lw=0.3, ec="white", label=lab_ if (i == 0 and k == 0) else None)
            if key == "catastrophic_shift" and w > 0.08: a.text(left + w/2, i, f"{int((q.category==key).sum())}", ha="center", va="center", fontsize=5.8, color="white", fontweight="bold")
            left += w
    a.set_xlim(0, 1); a.set_xticks([0, 0.5, 1]); a.set_xticklabels(["0", "0.5", "1"]); a.set_title(ttl, loc="left", fontsize=6.3, pad=3); a.tick_params(length=2); a.spines["left"].set_visible(False)
    a.set_yticks(range(len(trows))); a.set_yticklabels([f"{NAME[b]} {'naive' if arm=='naive' else 'canon.'}" for b, arm, _ in trows], fontsize=5.8); a.invert_yaxis()
    for i, (b, arm, _) in enumerate(trows): a.get_yticklabels()[i].set_color(COL[b])
    if k == 1: a.set_xlabel("fraction of recordings", labelpad=1)
    if k == 0: h, l = a.get_legend_handles_labels(); fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.76, 0.0), ncol=4, frameon=False, handlelength=1.0, columnspacing=0.8, fontsize=5.6, borderaxespad=0.0)
letter(fig, 0.005, 0.985, "a"); fig.text(0.03, 0.985, "128 Hz, per recording", fontsize=7, va="top")
letter(fig, 0.005, 0.50, "b"); fig.text(0.03, 0.50, "legacy recorder, 125 Hz", fontsize=7, va="top")
px = fig.axes[-2].get_position().x0 - 0.075; letter(fig, px, 0.985, "c"); fig.text(px + 0.025, 0.985, "failure categories", fontsize=7, va="top")
fig.savefig(f"{OUT}/fig3_recordings.pdf"); plt.close(fig); print("fig3")
