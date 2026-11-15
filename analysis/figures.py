"""Standalone scientific figures; no fabricated time histories or event data."""
import os
from pathlib import Path

import numpy as np

from .archive_io import SCENARIOS, PROFILES


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

    with plt.rc_context():
        def save(fig, name):
            fig.savefig(directory/(name+".png"), dpi=180)
            fig.savefig(directory/(name+".pdf"))
            plt.close(fig)

        def forest(rows, title, name, sensitivity=False):
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
                    ax.scatter([mean],[y],color=color,s=35,zorder=3)
                    ax.annotate(f"n={row['n_pairs']}",(mean,y),xytext=(4,6),textcoords="offset points",fontsize=8)
                ax.axvline(0,color="black",linestyle="--",linewidth=1)
                ax.set(title=WIND_LABELS[profile],xlabel="Position RMSE difference: SAC − PID (m)")
                ax.grid(axis="x",alpha=.2)
                ax.margins(x=.17,y=.12)
                ax.set_yticks(range(len(SCENARIOS)),[LABELS[s] for s in SCENARIOS])
            axes[0].invert_yaxis()
            subtitle = ("Conditional sensitivity: both nonfailed, same recorded length; selection can bias comparisons" if sensitivity else
                        f"{len(seeds)} matched seed pairs per condition; bars are pointwise 95% BCa intervals")
            fig.suptitle(title+"\n"+subtitle,fontsize=13)
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

        fig, axes = plt.subplots(1,3,figsize=(14,5),layout="constrained")
        for ax, profile in zip(axes, PROFILES):
            group = [p for p in pairs if p["wind_profile"]==profile]
            maximum = max(max(p[c]["position_rmse"] for c in ("pid","sac")) for p in group)*1.07
            ax.plot([0,maximum],[0,maximum],"--",color="gray",linewidth=1)
            for p in group:
                color, marker = ("#D55E00","^") if p["scenario"]=="dijkstra" else ("#0072B2","o")
                failed = p["pid"]["failed"] or p["sac"]["failed"]
                ax.scatter(p["pid"]["position_rmse"],p["sac"]["position_rmse"],marker=marker,
                           facecolors="none" if failed else color,edgecolors=color,s=36,alpha=.8)
            ax.set(xlim=(0,maximum),ylim=(0,maximum),xlabel="PID position RMSE (m)",ylabel="PID+SAC position RMSE (m)",title=WIND_LABELS[profile])
            ax.set_aspect("equal",adjustable="box")
        handles=[Line2D([],[],color="#0072B2",marker="o",linestyle="",label="Open-air case"),
                 Line2D([],[],color="#D55E00",marker="^",linestyle="",label="Dijkstra case"),
                 Line2D([],[],color="gray",marker="o",markerfacecolor="none",linestyle="",label="Either member failed")]
        axes[0].legend(handles=handles,loc="upper left",fontsize=8)
        fig.suptitle("Paired flight comparisons — below the diagonal favors SAC\nDescriptive points; repeated seed identifiers across conditions remain dependent",fontsize=12)
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
                ax.set_title(("PID" if controller=="pid" else "PID+SAC")+": reported "+outcome)
        fig.suptitle("Outcomes by condition\nNo failure does not guarantee completion; all recorded flights are retained",fontsize=13)
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
        ax.set(xlabel="Difference in mean flight-level position RMSE: SAC − PID (m)",
               title=f"Overall benchmark effects with seed dependence preserved\nEach estimate resamples {len(seeds)} whole seed blocks; pointwise 95% BCa intervals")
        ax.grid(axis="x",alpha=.2)
        save(fig,"seed_cluster_overview")

        fig, axes=plt.subplots(1,2,figsize=(12,6),layout="constrained")
        for ax,field,title in zip(axes,["different_decision_counts","both_nonfailed_equal_counts"],
                                 ["Pairs with different recorded lengths","Both nonfailed, same recorded length"]):
            counts=np.array([[next(r[field] for r in context["lengths"] if r["scenario"]==s and r["wind_profile"]==w)
                              for w in PROFILES] for s in SCENARIOS])
            ax.imshow(counts,vmin=0,vmax=len(seeds),cmap="Blues",aspect="auto")
            for i in range(len(SCENARIOS)):
                for j in range(3):
                    ax.text(j,i,f"{counts[i,j]}/{len(seeds)}",ha="center",va="center",color="white" if counts[i,j]>len(seeds)*.6 else "black")
            ax.set_xticks(range(3),["Training range","Unseen gusts","Outside range"],rotation=15)
            ax.set_yticks(range(len(SCENARIOS)),[LABELS[s] for s in SCENARIOS]); ax.set_title(title)
        fig.suptitle("Observed-length diagnostic\nDecision counts are a duration proxy, not precise failure timestamps",fontsize=13)
        save(fig,"observed_lengths")
        forest(context["sensitivity"],"RMSE among the surviving, equal-count pairs","rmse_sensitivity",sensitivity=True)
