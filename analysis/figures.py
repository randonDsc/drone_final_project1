"""Standalone scientific figures; no fabricated time histories or event data."""
import os
from pathlib import Path
from itertools import combinations

import numpy as np

from .archive_io import SCENARIOS, PROFILES
from .archive_io import json_write


LABELS = dict(hover="Hover", straight="Straight", turns="Turns / waypoints", altitude="Altitude",
              circle="Circle", figure_eight="Figure eight", dijkstra="Dijkstra")
WIND_LABELS = dict(training_range="Training-range wind", unseen_gusts="Unseen gusts", outside_range="Outside-range mean wind")


def create_figures(output, context):
    cache = Path(__file__).parent/".cache"
    os.environ.setdefault("MPLCONFIGDIR", str(cache/"matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    (cache/"matplotlib").mkdir(parents=True, exist_ok=True)
    (cache/"fontconfig").mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False,
                         "savefig.facecolor":"white", "pdf.fonttype":42})
    directory = output/"figures"
    directory.mkdir()
    pairs, primary = context["pairs"], context["primary"]
    seeds = sorted({p["seed"] for p in pairs})
    export_checks = {}

    with plt.rc_context():
        def save(fig, name):
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bounds = fig.get_tightbbox(renderer).padded(.2)
            pixel_bounds = bounds.transformed(fig.dpi_scale_trans)
            labels = [text for ax in fig.axes for text in (ax.xaxis.label, ax.yaxis.label, ax.title)]
            labels += [fig._suptitle] if fig._suptitle is not None else []
            checked = 0
            for label in labels:
                if not label.get_visible() or not label.get_text():
                    continue
                box = label.get_window_extent(renderer)
                if (box.x0<pixel_bounds.x0-1 or box.y0<pixel_bounds.y0-1
                        or box.x1>pixel_bounds.x1+1 or box.y1>pixel_bounds.y1+1):
                    raise ValueError("Figure text outside export bounds: "+name)
                checked += 1
            x_labels = [ax.xaxis.label.get_window_extent(renderer) for ax in fig.axes if ax.xaxis.label.get_text()]
            for left, right in combinations(x_labels, 2):
                if (min(left.x1,right.x1)-max(left.x0,right.x0)>1
                        and min(left.y1,right.y1)-max(left.y0,right.y0)>1):
                    raise ValueError("Overlapping x-axis labels: "+name)
            fig.savefig(directory/(name+".png"), dpi=220, bbox_inches="tight", pad_inches=.2)
            fig.savefig(directory/(name+".pdf"), bbox_inches="tight", pad_inches=.2)
            export_checks[name] = dict(axis_labels_within_export=True, text_elements_checked=checked,
                                       x_axis_labels_separated=True, padding_inches=.2, png_dpi=220)
            plt.close(fig)

        def forest(rows, title, name, sensitivity=False, metric_label="Position RMSE", secondary=False):
            fig, axes = plt.subplots(1,3,figsize=(14,7),sharey=True,layout="constrained")
            for ax, profile in zip(axes, PROFILES):
                for y, scenario in enumerate(SCENARIOS):
                    row = next(r for r in rows if r["scenario"]==scenario and r["wind_profile"]==profile)
                    if row["mean_difference"] is None:
                        ax.text(.5,y,"No eligible pairs",transform=ax.get_yaxis_transform(),ha="center",color="gray")
                        continue
                    mean, low, high = (row[k] for k in ("mean_difference","mean_difference_ci_low","mean_difference_ci_high"))
                    color = "#0072B2" if mean<0 else "#D55E00"
                    if low is not None:
                        ax.plot([low,high],[y,y],color=color,linewidth=2)
                    ax.scatter([mean],[y],edgecolor=color,
                               facecolor=color if not secondary or row["wilcoxon_reject_holm"] else "white",
                               s=35,zorder=3)
                    ax.annotate(f"n={row['n_pairs']}",(mean,y),xytext=(4,6),textcoords="offset points",fontsize=8)
                ax.axvline(0,color="black",linestyle="--",linewidth=1)
                ax.set(title=WIND_LABELS[profile],xlabel=metric_label+"\nDifference: PID+SAC − PID (m)")
                ax.grid(axis="x",alpha=.2)
                ax.margins(x=.17,y=.12)
                ax.set_yticks(range(len(SCENARIOS)),[LABELS[s] for s in SCENARIOS])
            axes[0].invert_yaxis()
            subtitle = ("Nonfailed pairs with equal recorded decision counts; pointwise 95% CIs" if sensitivity else
                        f"{len(seeds)} paired seeds per condition; pointwise 95% BCa intervals")
            fig.suptitle(title+"\n"+subtitle,fontsize=13)
            if secondary:
                handles = [Line2D([],[],marker="o",linestyle="",color="#333333",label="Within-metric Holm p ≤ 0.05"),
                           Line2D([],[],marker="o",linestyle="",markerfacecolor="white",color="#333333",label="Within-metric Holm p > 0.05")]
                fig.legend(handles=handles,loc="outside lower center",ncol=2,frameon=False,fontsize=9)
            save(fig,name)

        forest(primary,"Position tracking effect by condition","rmse_effects")

        fig, ax = plt.subplots(figsize=(12,11),layout="constrained")
        matrix = np.array([[next(p["sac"]["position_rmse"]-p["pid"]["position_rmse"] for p in pairs
                                 if (p["scenario"],p["wind_profile"],p["seed"])==(s,w,seed))
                            for seed in seeds] for s in SCENARIOS for w in PROFILES])
        scale = float(np.max(abs(matrix)))
        im = ax.imshow(matrix,cmap="RdBu_r",norm=TwoSlopeNorm(0,vmin=-scale,vmax=scale),aspect="auto")
        for y,(s,w) in enumerate((s,w) for s in SCENARIOS for w in PROFILES):
            for x,seed in enumerate(seeds):
                p = next(p for p in pairs if (p["scenario"],p["wind_profile"],p["seed"])==(s,w,seed))
                if p["pid"]["failed"] or p["sac"]["failed"]:
                    ax.text(x,y,"×",ha="center",va="center",fontsize=15,color="black")
        ax.set_xticks(range(len(seeds)),[str(s) for s in seeds],rotation=45,ha="right")
        ax.set_yticks(range(len(matrix)),[LABELS[s]+" / "+w for s in SCENARIOS for w in PROFILES])
        ax.set(xlabel="Environment seed",title="Per-seed position RMSE difference (SAC − PID)\nBlue favors SAC; × marks a failure in either member")
        fig.colorbar(im,ax=ax,label="Difference in flight-level RMSE (m)",shrink=.7)
        save(fig,"seed_differences")

        fig, axes = plt.subplots(1,3,figsize=(14,5.8),layout="constrained")
        for ax, profile in zip(axes, PROFILES):
            group = [p for p in pairs if p["wind_profile"]==profile]
            maximum = max(max(p[c]["position_rmse"] for c in ("pid","sac")) for p in group)*1.07
            ax.plot([0,maximum],[0,maximum],"--",color="gray",linewidth=1)
            for p in group:
                color, marker = ("#D55E00","^") if p["scenario"]=="dijkstra" else ("#0072B2","o")
                failed = p["pid"]["failed"] or p["sac"]["failed"]
                ax.scatter(p["pid"]["position_rmse"],p["sac"]["position_rmse"],marker=marker,
                           facecolors="none" if failed else color,edgecolors=color,s=36,alpha=.8)
            ax.set(xlim=(0,maximum),ylim=(0,maximum),xlabel="PID RMSE (m)",ylabel="PID+SAC RMSE (m)",title=WIND_LABELS[profile])
            ax.set_aspect("equal",adjustable="box")
        handles=[Line2D([],[],color="#0072B2",marker="o",linestyle="",label="Open-air case"),
                 Line2D([],[],color="#D55E00",marker="^",linestyle="",label="Dijkstra case"),
                 Line2D([],[],color="gray",marker="o",markerfacecolor="none",linestyle="",label="Either member failed")]
        axes[0].legend(handles=handles,loc="upper left",fontsize=8)
        fig.suptitle("Position RMSE by paired flight\nPoints below the diagonal favor PID+SAC",fontsize=13)
        save(fig,"paired_scatter")

        fig, axes = plt.subplots(2,2,figsize=(12,10),layout="constrained")
        for y,outcome in enumerate(("completed","failed")):
            for x,controller in enumerate(("pid","sac")):
                ax=axes[y,x]
                counts=np.array([[sum(p[controller][outcome] for p in pairs if p["scenario"]==s and p["wind_profile"]==w)
                                  for w in PROFILES] for s in SCENARIOS])
                ax.imshow(counts/len(seeds),vmin=0,vmax=1,cmap="Greens" if outcome=="completed" else "Reds",aspect="auto")
                for i in range(len(SCENARIOS)):
                    for j in range(3):
                        ax.text(j,i,f"{counts[i,j]}/{len(seeds)}",ha="center",va="center",color="white" if counts[i,j]>len(seeds)*.6 else "black")
                ax.set_xticks(range(3),["Training range","Unseen gusts","Outside range"],rotation=15)
                ax.set_yticks(range(len(SCENARIOS)),[LABELS[s] for s in SCENARIOS])
                ax.set_title(("PID" if controller=="pid" else "PID+SAC")+": "+outcome)
        fig.suptitle("Flight completion and failure by condition",fontsize=13)
        save(fig,"outcomes")

        fig, ax=plt.subplots(figsize=(12,6),layout="constrained")
        names={"open_air_all_profiles":"Open air: all profiles", "dijkstra_all_profiles":"Dijkstra: all profiles",
               "open_air_training_range":"Open air: training range", "open_air_unseen_gusts":"Open air: unseen gusts",
               "open_air_outside_range":"Open air: outside range"}
        for y,r in enumerate(context["overview"]):
            ax.plot([r["mean_difference_ci_low"],r["mean_difference_ci_high"]],[y,y],linewidth=2,color="#0072B2")
            ax.scatter(r["mean_difference"],y,color="#0072B2")
        ax.set_yticks(range(len(context["overview"])),[names[r["benchmark"]]+f" ({r['paired_cases']} cases)" for r in context["overview"]])
        ax.invert_yaxis(); ax.axvline(0,color="black",linestyle="--")
        ax.set(xlabel="Mean flight-level RMSE difference: PID+SAC − PID (m)",
               title=f"Benchmark effects across {len(seeds)} environment seed blocks\nPointwise 95% BCa intervals")
        ax.grid(axis="x",alpha=.2)
        save(fig,"seed_cluster_overview")

        fig, axes=plt.subplots(1,2,figsize=(12,6),layout="constrained")
        for ax,field,title in zip(axes,["different_decision_counts","both_nonfailed_equal_counts"],
                                 ["Unequal decision counts","Nonfailed, equal-count pairs"]):
            counts=np.array([[next(r[field] for r in context["lengths"] if r["scenario"]==s and r["wind_profile"]==w)
                              for w in PROFILES] for s in SCENARIOS])
            ax.imshow(counts,vmin=0,vmax=len(seeds),cmap="Blues",aspect="auto")
            for i in range(len(SCENARIOS)):
                for j in range(3):
                    ax.text(j,i,f"{counts[i,j]}/{len(seeds)}",ha="center",va="center",color="white" if counts[i,j]>len(seeds)*.6 else "black")
            ax.set_xticks(range(3),["Training range","Unseen gusts","Outside range"],rotation=15)
            ax.set_yticks(range(len(SCENARIOS)),[LABELS[s] for s in SCENARIOS]); ax.set_title(title)
        fig.suptitle("Paired-flight observation lengths",fontsize=13)
        save(fig,"observed_lengths")
        forest(context["sensitivity"],"RMSE among the surviving, equal-count pairs","rmse_sensitivity",sensitivity=True)

        for metric, label in (("cross_track_rmse", "Cross-track RMSE"),
                              ("max_deviation", "Maximum position error"),
                              ("cross_track_max", "Maximum cross-track error")):
            group = [row for row in context["secondary"] if row["metric"]==metric]
            forest(group, label+" by condition", "secondary_"+metric, metric_label=label, secondary=True)

        fig, axes = plt.subplots(1,2,figsize=(12,5.8),layout="constrained")
        for ax, controller in zip(axes, ("pid", "sac")):
            values = np.array([[np.mean([p[controller]["saturation_rate"] for p in pairs
                                        if p["scenario"]==s and p["wind_profile"]==w]) for w in PROFILES]
                               for s in SCENARIOS])
            ax.imshow(values,vmin=0,vmax=1,cmap="Greys",aspect="auto")
            for i in range(len(SCENARIOS)):
                for j in range(len(PROFILES)):
                    ax.text(j,i,f"{100*values[i,j]:.1f}%",ha="center",va="center")
            ax.set_xticks(range(3),["Training\nrange", "Unseen\ngusts", "Outside\nrange"])
            ax.set_yticks(range(len(SCENARIOS)),[LABELS[s] for s in SCENARIOS])
            ax.set_title("PID" if controller=="pid" else "PID+SAC")
        fig.suptitle("Recorded motor saturation\nMean flight-level fraction of saturated physics steps",fontsize=13)
        save(fig,"secondary_saturation_rate")

    json_write(output/"figure_export_checks.json", export_checks)
