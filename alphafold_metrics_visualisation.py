#!/usr/bin/env python3

"""
Created on 31 Jul. 2024
"""

__author__ = "Nicolas JEANNE"
__copyright__ = "GNU General Public License"
__email__ = "jeanne.n@chu-toulouse.fr"
__version__ = "1.0.0"


import argparse
import json
import logging
import os
import pickle
import re
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def create_log(path, level):
    """Create the log as a text file and as a stream.

    :param path: the path of the log.
    :type path: str
    :param level: the level og the log.
    :type level: str
    :return: the logging:
    :rtype: logging
    """

    log_level_dict = {"DEBUG": logging.DEBUG,
                      "INFO": logging.INFO,
                      "WARNING": logging.WARNING,
                      "ERROR": logging.ERROR,
                      "CRITICAL": logging.CRITICAL}

    if level is None:
        log_level = log_level_dict["INFO"]
    else:
        log_level = log_level_dict[level]

    if os.path.exists(path):
        os.remove(path)

    logging.basicConfig(format="%(asctime)s %(levelname)s:\t%(message)s",
                        datefmt="%Y/%m/%d %H:%M:%S",
                        level=log_level,
                        handlers=[logging.FileHandler(path), logging.StreamHandler()])
    return logging


def get_models_data(input_dir):
    """
    Extract the models' data.

    :param input_dir: the path to the Alphafold output directory.
    :type input_dir: str
    :return: the models' extracted data.
    rtype: dict
    """
    pattern_result_multimer = re.compile("result_model_\\d_multimer_.+\\.pkl")
    is_multimer = False
    for alphafold_file in os.listdir(input_dir):
        match = pattern_result_multimer.search(alphafold_file)
        if match:
            is_multimer = True
            break
    logging.info(f"{'Multimer' if is_multimer else 'Monomer'} Alphafold run detected.")

    data = []
    for f in range(1, 6):
        path = None
        if is_multimer:
            for g in range(5):
                path = os.path.join(args.input, f"result_model_{f}_multimer_v2_ptm_pred_{g}.pkl")
        else:
            # path = os.path.join(args.input, f"result_model_{f}_pred_0.pkl")
            path = os.path.join(args.input, f"result_model_{f}.pkl")
        data.append(pickle.load(open(path, "rb")))
    return data


def plot_msa_with_coverage(msa_data, out_dir, run_id, out_format):
    """
    Plot the Alphafold Multiple Sequence Alignment with the coverage information.

    :param msa_data: the MSA data.
    :type msa_data: numpy.ndarray
    :param out_dir: the path of the output directory.
    :type out_dir: str
    :param run_id: the alphafold run name.
    :type run_id: str
    :param out_format: toe output out_format.
    :type out_format: str
    """
    seq_id = (np.array(msa_data[0] == msa_data).mean(-1))
    seq_id_sort = seq_id.argsort()
    non_gaps = (msa_data != 21).astype(float)
    non_gaps[non_gaps == 0] = np.nan
    final = non_gaps[seq_id_sort] * seq_id[seq_id_sort, None]

    plt.subplots()
    plt.title(f"Sequence coverage ({run_id})")
    plt.imshow(final, interpolation="nearest", aspect="auto", cmap="rainbow_r", vmin=0, vmax=1, origin="lower")
    plt.plot((msa_data != 21).sum(0), color="black")
    plt.xlim(-0.5, msa_data.shape[1] - 0.5)
    plt.ylim(-0.5, msa_data.shape[0] - 0.5)
    plt.colorbar(label="Sequence identity to query")
    plt.xlabel("Positions")
    plt.ylabel("Sequences")
    path = os.path.join(out_dir, f"msa_coverage_{run_id}.{out_format}")
    plt.savefig(path)
    logging.info(f"MSA coverage plot: {path}")


def get_pae_plddt(data_models):
    """
    Extract the pLDDT scores and PAE scores (if any) for each model.

    :param data_models: the models' data.
    :type data_models: dict
    :return: the pLDDT and PAE (if any) data.
    :rtype: dict
    """
    data = {}
    add_pae = True
    if "predicted_aligned_error" not in data_models[0].keys():
        add_pae = False
        logging.warning("No Predicted Alignement Error (PAE) values in the Alphafold models, the PAE plots will not be "
                        "produced.")
    for idx, val in enumerate(data_models):
        if add_pae:
            data[f"model_{idx + 1}"] = {"plddt": val["plddt"], "pae": val["predicted_aligned_error"]}
        else:
            data[f"model_{idx + 1}"] = {"plddt": val["plddt"]}
    return data


def plot_plddt(data, out_dir, run_id, out_format):
    """
    Plot the Alphafold predicted Local Distance Difference Test (pLDDT) scores.

    :param data: the models' data.
    :type data: dict
    :param out_dir: the path of the output directory.
    :type out_dir: str
    :param run_id: the alphafold run name.
    :type run_id: str
    :param out_format: the output out_format.
    :type out_format: str
    """
    plt.clf()
    plt.subplots()
    plt.title(f"Predicted LDDT per position ({run_id})")
    s = 0
    for model_name, value in data.items():
        plddt_value = round(list(ranking_dict["plddts"].values())[s], 6)
        plt.plot(value["plddt"], label=f"{model_name} pLDDT: {plddt_value}")
        s += 1
    plt.legend()
    plt.ylim(0, 100)
    plt.ylabel("Predicted LDDT")
    plt.xlabel("Positions")
    path = os.path.join(out_dir, f"pLDDT_coverage_{run_id}.{out_format}")
    plt.savefig(path)
    logging.info(f"pLDDT plot: {path}")


def plot_pae(data, out_dir, run_id, out_format):
    """
    Plot the Alphafold Predicted Alignment Error (PAE) scores.

    :param data: the models' data.
    :type data: dict
    :param out_dir: the path of the output directory.
    :type out_dir: str
    :param run_id: the alphafold run name.
    :type run_id: str
    :param out_format: the output out_format.
    :type out_format: str
    """
    plt.clf()
    plt.subplots()
    plt.title(f"Predicted Alignment Error ({run_id})")
    num_models = len(data)
    plt.figure(figsize=(3 * num_models, 2), dpi=100)
    for n, (model_name, value) in enumerate(data.items()):
        plt.subplot(1, num_models, n + 1)
        plt.imshow(value["pae"], label=model_name, cmap="bwr", vmin=0, vmax=30)
        plt.colorbar()

    path = os.path.join(out_dir, f"PAE_{run_id}.{out_format}")
    plt.savefig(path)
    logging.info(f"PAE plot: {path}")


if __name__ == "__main__":
    descr = f"""
    {os.path.basename(__file__)} v. {__version__}

    Created by {__author__}.
    Contact: {__email__}
    {__copyright__}

    Distributed on an "AS IS" basis without warranties or conditions of any kind, either express or implied.

    Create the metrics visualisation plots for the 5 Alphafold predicted models:
        - Multiple Sequence Alignment (MSA) with coverage.
        - predicted Local Difference Distances Test (pLDDT) scores.
        - Predicted Alignment Error (PAE).
    
    Inspired by: https://blog.biostrand.ai/explained-how-to-plot-the-prediction-quality-metrics-with-alphafold2
    """
    parser = argparse.ArgumentParser(description=descr, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--out", required=True, type=str, help="the path to the output directory.")
    parser.add_argument("-y", "--format", required=False, default="svg",
                        choices=["eps", "jpg", "jpeg", "pdf", "pgf", "png", "ps", "raw", "svg", "svgz", "tif", "tiff"],
                        help="the output plots out_format: 'eps': 'Encapsulated Postscript', "
                             "'jpg': 'Joint Photographic Experts Group', 'jpeg': 'Joint Photographic Experts Group', "
                             "'pdf': 'Portable Document Format', 'pgf': 'PGF code for LaTeX', "
                             "'png': 'Portable Network Graphics', 'ps': 'Postscript', 'raw': 'Raw RGBA bitmap', "
                             "'rgba': 'Raw RGBA bitmap', 'svg': 'Scalable Vector Graphics', "
                             "'svgz': 'Scalable Vector Graphics', 'tif': 'Tagged Image File Format', "
                             "'tiff': 'Tagged Image File Format'. Default is 'svg'.")
    parser.add_argument("-l", "--log", required=False, type=str,
                        help="the path for the log file. If this option is skipped, the log file is created in the "
                             "output directory.")
    parser.add_argument("--log-level", required=False, type=str,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="set the log level. If the option is skipped, log level is INFO.")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("input", type=str,
                        help="the path to the Alphafold prediction directory.")
    args = parser.parse_args()

    # create output directory if necessary
    os.makedirs(args.out, exist_ok=True)
    # create the logger
    if args.log:
        log_path = args.log
    else:
        log_path = os.path.join(args.out, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
    create_log(log_path, args.log_level)

    logging.info(f"version: {__version__}")
    logging.info(f"CMD: {' '.join(sys.argv)}")

    # extract the data
    name = os.path.basename(os.path.normpath(args.input))
    logging.info(f"Alphafold metrics visualisation for {name}.")
    ranking_dict = json.load(open(os.path.join(args.input, "ranking_debug.json"), "r"))
    feature_dict = pickle.load(open(os.path.join(args.input, "features.pkl"), "rb"))
    model_dicts = get_models_data(args.input)

    # plot the MSA with coverage
    plot_msa_with_coverage(feature_dict["msa"], args.out, name, args.format)

    # get the pLLDT and PAE values per model
    pae_plddt_per_model = get_pae_plddt(model_dicts)

    # plot the pLDDT per position
    plot_plddt(pae_plddt_per_model, args.out, name, args.format)

    # plot the PAE if any PAE data
    if "pae" in pae_plddt_per_model[f"model_1"]:
        plot_pae(pae_plddt_per_model, args.out, name, args.format)
