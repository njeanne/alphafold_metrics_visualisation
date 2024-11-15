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

from dna_features_viewer import GraphicFeature, GraphicRecord
import matplotlib
matplotlib.use('Agg')
from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def get_data_ranking(path, alphafold_version):
    """
    Get the models ranking depending on the Alphafold version.
    pLDDT score for Alphafold2 and Ranking Score for Alphafold3.

    :param path: the path of the input file or directory depending on the alphafold version.
    :type path: str
    :param alphafold_version: the version of Alphafold.
    :type alphafold_version: str
    :return: the ranking data.
    :rtype: dict
    """
    data = {}
    no_match_any_file = True
    if alphafold_version == "alphafold2":
        data = json.load(open(os.path.join(path, "ranking_debug.json"), "r"))["plddts"]
    else:
        pattern_summary_confidences = re.compile(".+_summary_confidences_([0-4])\\.json")
        for alphafold3_file in os.listdir(path):
            match = pattern_summary_confidences.search(alphafold3_file)
            if match:
                no_match_any_file = False
                model_nb = match.group(1)
                data[f"model_{model_nb}"] = json.load(open(os.path.join(path, alphafold3_file), "r"))["ranking_score"]

        if no_match_any_file:
            logging.error(
                f"No match with an Alphafold3 result summary confidence file found, check if the input directory is an "
                f"Alphafold3 result directory: {args.input}")
            sys.exit(1)
    # order from the lowest to the highest to print the model with the highest value on top of the others
    data = dict(sorted(data.items(), key=lambda item: item[1], reverse=False))

    return data


def get_models_data_alphafold2(input_dir):
    """
    Extract the Alphafold2 models' data.

    :param input_dir: the path to the Alphafold2 output directory.
    :type input_dir: str
    :return: the models' extracted data.
    rtype: dict
    """
    pattern_result_multimer = re.compile("result_model_\\d_multimer_.+\\.pkl")
    is_multimer = False
    for alphafold2_file in os.listdir(input_dir):
        match = pattern_result_multimer.search(alphafold2_file)
        if match:
            is_multimer = True
            break
    logging.info(f"{'Multimer' if is_multimer else 'Monomer'} Alphafold2 run detected.")

    data = {}
    for model_nb in range(1, 6):
        path = None
        if is_multimer:
            for g in range(5):
                path = os.path.join(args.input, f"result_model_{model_nb}_multimer_v2_ptm_pred_{g}.pkl")
        else:
            path = os.path.join(args.input, f"result_model_{model_nb}_ptm_pred_0.pkl")
            if not os.path.exists(path):
                path = os.path.join(args.input, f"result_model_{model_nb}.pkl")
        data[model_nb] = pickle.load(open(path, "rb"))

    return data


def get_models_data_alphafold3(input_dir):
    """
    Extract the Alphafold3 models' data.

    :param input_dir: the path to the Alphafold3 output directory.
    :type input_dir: str
    :return: the models' extracted data.
    rtype: dict
    """
    data = {}
    no_match_any_file = True
    pattern_full_data = re.compile(".+_full_data_([0-4])\\.json")
    for alphafold3_file in os.listdir(input_dir):
        match = pattern_full_data.search(alphafold3_file)
        if match:
            no_match_any_file = False
            model_nb = match.group(1)
            data[model_nb] = json.load(open(os.path.join(input_dir, alphafold3_file), "r"))

    if no_match_any_file:
        logging.error(f"No match with an Alphafold3 full data result file found, check if the input directory is an "
                      f"Alphafold3 result directory: {args.input}")
        sys.exit(1)

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
    plt.suptitle(f"Sequence coverage:\n{run_id}", fontsize="large", fontweight="bold")
    plt.imshow(final, interpolation="nearest", aspect="auto", cmap="rainbow_r", vmin=0, vmax=1, origin="lower")
    plt.plot((msa_data != 21).sum(0), color="black")
    plt.xlim(-0.5, msa_data.shape[1] - 0.5)
    plt.ylim(-0.5, msa_data.shape[0] - 0.5)
    plt.colorbar(label="Sequence identity to query")
    plt.xlabel("Positions")
    plt.ylabel("Sequences")
    plt.tight_layout()
    path = os.path.join(out_dir, f"msa_coverage_{run_id}.{out_format}")
    plt.savefig(path)
    logging.info(f"MSA coverage plot: {path}")


def get_pae_plddt(data_models, alphafold_version):
    """
    Extract the pLDDT scores and PAE scores (if any for Alphafold2) for each model.

    :param data_models: the models' data.
    :type data_models: dict
    :param alphafold_version: the version of Alphafold.
    :type alphafold_version: str
    :return: the pLDDT and PAE (if any) data.
    :rtype: dict
    """
    data = {}
    add_pae = True
    data_first_model_keys = data_models[list(data_models.keys())[0]].keys()
    if alphafold_version == "alphafold2" and "predicted_aligned_error" not in data_first_model_keys:
        add_pae = False
        logging.warning("No Predicted Alignement Error (PAE) values in the Alphafold models, the PAE plots will not be "
                        "produced.")
    plddt_key = "atom_plddts"
    pae_key = "pae"
    if alphafold_version == "alphafold2":
        plddt_key = "plddt"
        pae_key = "predicted_aligned_error"
    for model_nb, val in data_models.items():
        data[f"model_{model_nb}"] = {"plddt": val[plddt_key]}
        if add_pae:
            data[f"model_{model_nb}"]["pae"] = val[pae_key]

    return data


def plot_plddt(data, data_ranking, out_dir, run_id, out_format, alphafold_version):
    """
    Plot the Alphafold predicted Local Distance Difference Test (pLDDT) scores.

    :param data: the models' data.
    :type data: dict
    :param data_ranking: the models' ranking.
    :type data_ranking: dict
    :param out_dir: the path of the output directory.
    :type out_dir: str
    :param run_id: the alphafold run name.
    :type run_id: str
    :param out_format: the output format.
    :type out_format: str
    :param alphafold_version: the version of Alphafold.
    :type alphafold_version: str
    """
    plt.clf()
    plt.subplots()
    plt.suptitle("Predicted LDDT per position", fontsize="large", fontweight="bold")
    plt.title(run_id, fontsize="large", fontweight="bold")
    model_index = 0
    pattern_model_alphafold2 = re.compile("(model_\\d)_ptm_pred_\\d")
    for model_name in data_ranking.keys():
        if alphafold_version == "alphafold2":
            match = pattern_model_alphafold2.search(model_name)
            if match:
                model_name = match.group(1)
            else:
                logging.error(f"No match for the model name \"{model_name}\" with the pattern "
                              f"{pattern_model_alphafold2.pattern}")
                sys.exit(1)
            plddt_value = round(list(data_ranking.values())[model_index], 6)
            plt.plot(data[model_name]["plddt"], label=f"{model_name.replace('_', ' ')} pLDDT: {plddt_value}")
        else:
            ranking_score = list(data_ranking.values())[model_index]
            plt.plot(data[model_name]["plddt"], label=f"{model_name} Ranking Score: {ranking_score}")
        model_index += 1
    plt.legend()
    plt.ylim(0, 100)
    plt.ylabel("Predicted LDDT")
    plt.xlabel("Positions")
    path = os.path.join(out_dir, f"pLDDT_{run_id}.{out_format}")
    plt.savefig(path)
    logging.info(f"pLDDT plot: {path}")


def plot_pae(data, out_dir, run_id, out_format, domains_path):
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
    :param domains_path: the domains' coordinates and info file path.
    :type domains_path: str
    """
    models = sorted(list(data.keys()))
    domains = None
    if domains_path is not None:
        try:
            domains = pd.read_csv(args.domains)
        except FileNotFoundError as exc:
            logging.error(exc)
            sys.exit(1)

    for model in models:
        plt.clf()
        if domains is not None:
            # Add the domains' plot
            fig, (ax0, ax1) = plt.subplots(2, 1, height_ratios=[5, 1], sharex=True)
            features = []
            row = None
            for _, row in domains.iterrows():
                features.append(GraphicFeature(start=row["start"], end=row["end"], strand=+1, color=row["color"],
                                               label=row["domain"]))
            record = GraphicRecord(sequence_length=row["end"] + 1, features=features, plots_indexing="genbank")
            record.plot(ax=ax1)
        else:
            fig, ax0 = plt.subplots(1, 1)

        fig.suptitle(f"Predicted Alignment Error {model.replace('_', ' ')}:\n{run_id}", fontsize="large",
                     fontweight="bold")
        heatmap = ax0.imshow(data[model]["pae"], label=model, cmap="bone", vmin=0, vmax=30)
        fig.colorbar(heatmap, ax=ax0, label="Expected Position Error (\u212B)")
        ax0.set_xlabel("Scored Residue")
        ax0.set_ylabel("Aligned Residue")
        ax0.set_ylim(len(data[model]["pae"]) + 1, 1)

        plt.tight_layout()
        if domains is not None:
            # shrink the domains' X axis to the heatmap X axis size
            ax0_positions, ax1_positions = ax0.get_position(), ax1.get_position()
            ax1.set_position([ax0_positions.x0, ax1_positions.y0, ax0_positions.width, ax1_positions.height])

        path = os.path.join(out_dir, f"PAE_{model.replace('_', '-')}_{run_id}.{out_format}")
        plt.savefig(path)
        logging.info(f"PAE plot {model.replace('_', ' ')}: {path}")


if __name__ == "__main__":
    descr = f"""
    {os.path.basename(__file__)} v. {__version__}

    Created by {__author__}.
    Contact: {__email__}
    {__copyright__}

    Distributed on an "AS IS" basis without warranties or conditions of any kind, either express or implied.

    Create the metrics visualisation plots for the 5 Alphafold predicted models:
        - Multiple Sequence Alignment (MSA) with coverage (Alphafold2 data only).
        - predicted Local Difference Distances Test (pLDDT) scores.
        - Predicted Alignment Error (PAE).
    
    Inspired and redesigned from the work of Jasper Zuallaert: https://github.com/jasperzuallaert/VIBFold/blob/main/visualize_alphafold_results.py
    """
    parent_parser = argparse.ArgumentParser()
    parent_parser.add_argument("-o", "--out", required=True, type=str, help="the path to the output directory.")
    parent_parser.add_argument("-d", "--domains", required=False, type=str,
                               help="the path to the CSV domains file. The domains file is used in the plot to display "
                                    "a map of the domains. If the mask do not cover all the domains in the file, the "
                                    "domains argument should not be used. the domains file is a comma separated file, "
                                    "the first column is the annotation name, the 2nd is the residue start coordinate, "
                                    "the 3rd is the residue end coordinate, the last one is the color to apply in "
                                    "hexadecimal format. The coordinate are 1-indexed.")
    parent_parser.add_argument("-y", "--format", required=False, default="svg",
                        choices=["eps", "jpg", "jpeg", "pdf", "pgf", "png", "ps", "raw", "svg", "svgz", "tif", "tiff"],
                        help="the output plots out_format: 'eps': 'Encapsulated Postscript', "
                             "'jpg': 'Joint Photographic Experts Group', 'jpeg': 'Joint Photographic Experts Group', "
                             "'pdf': 'Portable Document Format', 'pgf': 'PGF code for LaTeX', "
                             "'png': 'Portable Network Graphics', 'ps': 'Postscript', 'raw': 'Raw RGBA bitmap', "
                             "'rgba': 'Raw RGBA bitmap', 'svg': 'Scalable Vector Graphics', "
                             "'svgz': 'Scalable Vector Graphics', 'tif': 'Tagged Image File Format', "
                             "'tiff': 'Tagged Image File Format'. Default is 'svg'.")
    parent_parser.add_argument("-l", "--log", required=False, type=str,
                        help="the path for the log file. If this option is skipped, the log file is created in the "
                             "output directory.")
    parent_parser.add_argument("--log-level", required=False, type=str,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="set the log level. If the option is skipped, log level is INFO.")
    parent_parser.add_argument("--version", action="version", version=__version__)

    # add subparser for the different Alphafold versions
    parser = argparse.ArgumentParser(description=descr, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(title="Alphafold version used", dest="alphafold_version",
                                       help="the Alphafold version used.")

    # Alphafold2
    parser_alphafold2 = subparsers.add_parser("alphafold2", parents=[parent_parser], add_help=False,
                                              help="use the Alphafold2 outputs.")
    parser_alphafold2.add_argument("input", type=str,
                                   help="the path to the Alphafold2 modeling directory.")

    # Alphafold3
    parser_alphafold3 = subparsers.add_parser("alphafold3", parents=[parent_parser], add_help=False,
                                              help="use the Alphafold3 outputs.")
    parser_alphafold3.add_argument("input", type=str,
                                  help="the path to the Alphafold3 modeling directory.")

    args = parser.parse_args()

    rcParams["figure.figsize"] = 15, 12

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
    logging.info(f"{args.alphafold_version.capitalize()} metrics visualisation for {name}.")

    model_dicts = None
    ranking_dict = get_data_ranking(args.input, args.alphafold_version)

    if args.alphafold_version == "alphafold2":
        feature_dict = pickle.load(open(os.path.join(args.input, "features.pkl"), "rb"))
        model_dicts = get_models_data_alphafold2(args.input)
        # plot the MSA with coverage
        plot_msa_with_coverage(feature_dict["msa"], args.out, name, args.format)
    elif args.alphafold_version == "alphafold3":
        model_dicts = get_models_data_alphafold3(args.input)

    # get the pLLDT and PAE values per model
    pae_plddt_per_model = get_pae_plddt(model_dicts, args.alphafold_version)

    # plot the pLDDT per position
    plot_plddt(pae_plddt_per_model, ranking_dict, args.out, name, args.format, args.alphafold_version)

    # plot the PAE if any PAE data
    if "pae" in pae_plddt_per_model[f"model_1"]:
        plot_pae(pae_plddt_per_model, args.out, name, args.format, args.domains)
