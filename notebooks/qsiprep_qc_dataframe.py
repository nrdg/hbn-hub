import argparse
import functools
import json
import numpy as np
import operator
import pandas as pd
import s3fs
import seaborn as sns


def create_qsiprep_qc_dataframe(qc_file, exclusion_criteria="default", plot_pairs=False):
    if qc_file.startswith("s3://"):
        fs = s3fs.S3FileSystem()
        with fs.open(qc_file) as fp:
            qc_json = json.load(fp)
    else:
        with open(qc_file, "r") as fp:
            qc_json = json.load(fp)
    
    df_qc = pd.DataFrame(qc_json["subjects"])

    if exclusion_criteria is None:
        return df_qc
    
    if exclusion_criteria == "default":
        exclusion_criteria = {
            "raw_neighbor_corr": (0.05, 1.0),
            "t1_neighbor_corr": (0.05, 1.0),
            "mean_fd": (0.0, 0.95),
            "t1_dice_distance": (0.0, 0.95),
            "raw_num_bad_slices": (0.0, 0.95),
        }
    
    masks = {}
    for qc_metric, (p_lo, p_hi) in exclusion_criteria.items():
        masks[qc_metric] = np.logical_and(
            df_qc[qc_metric] >= df_qc[qc_metric].quantile(p_lo),
            df_qc[qc_metric] <= df_qc[qc_metric].quantile(p_hi)
        )
        
        df_qc[qc_metric + "_pass"] = masks[qc_metric]
    
    union = functools.reduce(operator.mul, list(masks.values()), 1)
    df_qc["qc_pass"] = union

    if plot_pairs:
        sns.pairplot(df_qc, vars=list(masks.keys()), hue="qc_pass")
    
    return df_qc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a qsiprep group-level dwiqc report, optionally identifying participants based on exclusion criteria"
    )
    parser.add_argument("qc_path", metavar="qc_path", type=str, help="the path to the input QC file. Can be an Amazon S3 URI.")
    parser.add_argument("output", metavar="output", type=str, help="path to output file")
    parser.add_argument(
        "-e", "--exclude", action="append", nargs=3, metavar=("metric", "min", "max"),
        help="exclusion criteria passed as three arguments: metric name, minimum acceptable value, maximum acceptable value"
    )
    
    args = parser.parse_args()

    if args.exclude:
        exclude = {metric: (float(low), float(high)) for metric, high, low in args.exclude}
    else:
        exclude = "default"

    df = create_qsiprep_qc_dataframe(args.qc_path, exclusion_criteria=exclude, plot_pairs=False)
    df.to_csv(args.output)