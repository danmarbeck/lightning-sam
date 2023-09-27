import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

METHOD_PRINT_NAMES = {
        "points": "Points",
        "": "Bounding Box",
        "fixations": "Fixations",
        "gaze": "Blurred Gaze",
        "hm_fixations": "HM Fixations",
        "hm_fixations_duration": "HM Fix. + Dur.",
        "hm_gaze": "HM Gaze",
    }


def convert_tb_data(root_dir, sort_by=None):
    """Convert local TensorBoard data into Pandas DataFrame.

    Function takes the root directory path and recursively parses
    all events data.
    If the `sort_by` value is provided then it will use that column
    to sort values; typically `wall_time` or `step`.

    *Note* that the whole data is converted into a DataFrame.
    Depending on the data size this might take a while. If it takes
    too long then narrow it to some sub-directories.

    Paramters:
        root_dir: (str) path to root dir with tensorboard data.
        sort_by: (optional str) column name to sort by.

    Returns:
        pandas.DataFrame with [wall_time, name, step, value] columns.

    """
    import os
    import pandas as pd
    from tensorflow.python.summary.summary_iterator import summary_iterator

    def convert_tfevent(filepath, root_dir):
        return pd.DataFrame([
            parse_tfevent(e, filepath, root_dir) for e in summary_iterator(filepath) if len(e.summary.value)
        ])

    def parse_tfevent(tfevent, filepath, root_dir):
        return dict(
            wall_time=tfevent.wall_time,
            name=tfevent.summary.value[0].tag,
            step=tfevent.step,
            value=float(tfevent.summary.value[0].simple_value),
            exp_name=re.sub("/version_\d","", str(Path(filepath).parent).strip(root_dir))
        )

    columns_order = ['exp_name', 'wall_time', 'name', 'step', 'value']

    out = []
    for (root, _, filenames) in os.walk(root_dir):
        for filename in filenames:
            if "events.out.tfevents" not in filename:
                continue
            file_full_path = os.path.join(root, filename)
            out.append(convert_tfevent(file_full_path, root_dir))

    # Concatenate (and sort) all partial individual dataframes
    all_df = pd.concat(out)[columns_order]
    if sort_by is not None:
        all_df = all_df.sort_values(sort_by)

    return all_df.reset_index(drop=True)


def plot_lr_scan(args):

    out_path = Path("/home/daniel/PycharmProjects/lightning-sam/plotting")
    out_path.mkdir(exist_ok=True, parents=True)



    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Times New Roman"
    })

    dir_path = "/home/daniel/PycharmProjects/lightning-sam/out/training/"
    exp_name = "lr_scan_full/"

    df = convert_tb_data(f"{dir_path}/{exp_name}")

    learning_rate_strs = list(sorted(set([s.split("_")[-1] for s in list(df["exp_name"])]), key=lambda str: float(str)))
    method_strs = list(set([re.sub('_?pascal_lr_\de-\d', "", re.sub('palma_3090_base_config_precomputed_', "", name)) for name in list(set(df["exp_name"]))]))
    method_strs = list(sorted(method_strs, key=lambda x: list(METHOD_PRINT_NAMES.keys()).index(x)))

    df_f1_values = df[df["name"] == "f1_avg"]
    df_iou_values = df[df["name"] == "iou_val"]

    fig, ax = plt.subplots(figsize=(8,4.5))

    f1_results = {}

    for method_str in method_strs:

        if method_str == "":
            method_values_f1 = df_f1_values[["precomputed_pascal" in name for name in df_f1_values["exp_name"]]]
            method_values_iou = df_f1_values[["precomputed_pascal" in name for name in df_iou_values["exp_name"]]]
        else:
            method_values_f1 = df_f1_values[[f"precomputed_{method_str}_pascal" in name for name in df_f1_values["exp_name"]]]
            method_values_iou = df_f1_values[[f"precomputed_{method_str}_pascal" in name for name in df_iou_values["exp_name"]]]

        end_results_f1 = [method_values_f1.loc[method_values_f1[[lr_str in name for name in method_values_f1["exp_name"]]]["step"].idxmax()]["value"] for lr_str in learning_rate_strs]
        end_results_iou = [method_values_iou.loc[method_values_iou[[lr_str in name for name in method_values_iou["exp_name"]]]["step"].idxmax()]["value"] for lr_str in learning_rate_strs]

        ax.plot(list(map(float, learning_rate_strs)), end_results_f1, label=METHOD_PRINT_NAMES[method_str])

        f1_results[method_str] = end_results_f1

    ax.vlines(list(map(float, learning_rate_strs))[np.argmax(f1_results["fixations"])],
              np.max(f1_results["hm_gaze"]),
              np.max(f1_results["fixations"]), colors="gray", linestyles="--")
    ax.vlines(list(map(float, learning_rate_strs))[np.argmax(f1_results["points"])],
              np.max(f1_results[""]),
              np.max(f1_results["points"]), colors="gray", linestyles="--")

    ax.ticklabel_format(style="sci", scilimits=(-2, 2))
    ax.set_xscale("log")
    ax.set_ylim(0.85, 0.92)
    ax.grid(which="both", alpha=0.6)
    ax.set_xlabel(r"Learning Rate $\eta$")
    ax.set_ylabel("Dice Score (Validation)")
    ax.legend()

    fig.savefig(Path(out_path, "lr_scan.pdf"))

    print(df.columns)


if __name__ == '__main__':
    args = None

    plot_lr_scan(args)
