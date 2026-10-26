import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.mapping import profile_color


def preview():
    for method in ["kmeans","dbscan"]:
        geometry=json.loads((ROOT/f"maps/{method}_grid_geometry.geojson").read_text())
        shapes={f["id"]:f["geometry"]["coordinates"][0] for f in geometry["features"]}
        train=pd.read_csv(ROOT/f"models/{method}/training_labels.csv")
        test=pd.read_csv(ROOT/"reports/testing_2026/labels_all_2026.csv")
        test["profile"]=test[f"{method}_profile"]
        fig,axes=plt.subplots(2,1,figsize=(15,9))
        for ax,frame,period in [(axes[0],train,"2025Q4"),(axes[1],test,"2026Q3")]:
            frame=frame.loc[frame["period"].eq(period)]
            ax.add_collection(PolyCollection([shapes[g] for g in frame["grid_id"]],
                facecolors=[profile_color(p) for p in frame["profile"]],edgecolors="#777777",linewidths=.12))
            ax.set(xlim=(95,141),ylim=(-11,6),xlabel="Longitude",ylabel="Latitude",
                   title=f"{method.upper()} | {period} | profil aktivitas katalog")
            ax.set_aspect("equal")
            ax.legend(handles=[Patch(facecolor=profile_color(p),label=p) for p in sorted(frame["profile"].unique())],
                      loc="upper left",bbox_to_anchor=(1,1),fontsize=8)
        fig.tight_layout()
        fig.savefig(ROOT/f"reports/figures/{method}_map_preview.png",dpi=140)
        plt.close(fig)
    print("Pratinjau peta statis tersimpan.")


if __name__=="__main__":
    preview()
