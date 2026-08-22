import argparse


import torch
import time
from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# custom
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet

import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r
import datasets.aptos
import datasets.eyepacs
import datasets.messidor
import datasets.messidor_2
import datasets.kather
import datasets.digestpath
import datasets.pannuke
import datasets.covid
import datasets.rsna18
import datasets.FourtyX
import datasets.HundredX
import datasets.TwoHundredX
import datasets.FourHundredX

import trainers.coop
import trainers.cocoop
import trainers.zsclip
import trainers.maple
import trainers.independentVL
import trainers.vpt
import trainers.promptsrc
import trainers.kgcoop
import trainers.prograd
import trainers.dapt
import trainers.hrmcoop
import trainers.hrmmaple
import trainers.highencodermaple
import trainers.mmrl
import trainers.mmrlpp
import trainers.hicropl
import trainers.hicroplreason

def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = False #""  # initialization words #set false for Kgcoop
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    # === NEW: HRMCoOp config ===
    cfg.TRAINER.HRMCOOP = CN()
    cfg.TRAINER.HRMCOOP.N_CTX = 16
    cfg.TRAINER.HRMCOOP.CSC = False
    cfg.TRAINER.HRMCOOP.CTX_INIT = ""     # or some phrase if you want init words
    cfg.TRAINER.HRMCOOP.PREC = "fp16"     # fp16, fp32, amp
    cfg.TRAINER.HRMCOOP.CLASS_TOKEN_POSITION = "end"


    # HRM-style parameters (simple defaults)
    cfg.TRAINER.HRMCOOP.H_CYCLES = 2      # number of high-level cycles
    cfg.TRAINER.HRMCOOP.L_CYCLES = 2      # number of low-level cycles per H-step
    cfg.TRAINER.HRMCOOP.N_HEADS  = 4
    cfg.TRAINER.HRMCOOP.MLP_RATIO = 4.0 
    
    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp

    # Config for MaPLe
    cfg.TRAINER.MAPLE = CN()
    cfg.TRAINER.MAPLE.N_CTX = 2  # number of context vectors
    cfg.TRAINER.MAPLE.CTX_INIT = "a photo of the cool"  # initialization words
    cfg.TRAINER.MAPLE.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.MAPLE.PROMPT_DEPTH = 9 # Max 12, minimum 0, for 1 it will act as shallow MaPLe (J=1)
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    cfg.TRAINER.MAPLE.MARGIN_ALPHA = 0.1   # default
    cfg.TRAINER.MAPLE.MARGIN_BETA  = 0.01  # default
    # weights for the two regularizers (match paper defaults)
    cfg.TRAINER.MAPLE.MARGIN_LAMBDA = 1.0
    cfg.TRAINER.MAPLE.MOM_LAMBDA    = 5.0

    # logging controls
    cfg.TRAINER.MAPLE.SCALE_LOG_ENABLE = True
    cfg.TRAINER.MAPLE.SCALE_LOG_EVERY  = 20     # log every N iterations
    cfg.TRAINER.MAPLE.SCALE_LOG_GRADS  = True   # enable gradient-norm logging
    cfg.TRAINER.MAPLE.SCALE_LOG_FILE   = "scale_grad_log.csv"
    # --- add under cfg.TRAINER.MAPLE ---
    cfg.TRAINER.MAPLE.PLOT_ANGDIST = False     # enable/disable plots
    cfg.TRAINER.MAPLE.ANGDIST_MAX_BATCHES = 50 # vision: how many test batches to average
    cfg.TRAINER.MAPLE.ANGDIST_MAX_CLASSES = 0  # text: 0 = all classes, else cap for speed

    # Config for HiCroPL
    cfg.TRAINER.HICROPL = CN()
    cfg.TRAINER.HICROPL.N_CTX = 2 # number of context vectors
    cfg.TRAINER.HICROPL.CROSS_LAYER = 6 # cross layer
    cfg.TRAINER.HICROPL.CTX_INIT = "a photo of a" # initialization words (only for language prompts)
    cfg.TRAINER.HICROPL.PREC = "fp32"
    cfg.TRAINER.HICROPL.PROMPT_DEPTH = 9  # Max 12, minimum 0, for 1 it will act as shallow HICROPL (J=1)
    cfg.TRAINER.HICROPL.TEACHER_NAME = "ViT-L/14"
    cfg.TRAINER.HICROPL.LAMBD = 12.0
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    # layerwise cross-modal cosine alignment
    cfg.TRAINER.HICROPL.ALIGN_START = 2      # 0-based layer index
    cfg.TRAINER.HICROPL.ALIGN_END = 8        # inclusive
    cfg.TRAINER.HICROPL.ALIGN_LAMBDA = 1.0
    cfg.TRAINER.HICROPL.ALIGN_T2V_HIDDEN = 1024
    cfg.TRAINER.HICROPL.ALIGN_V2T_HIDDEN = 1024
    cfg.TRAINER.HICROPL.ALIGN_DROPOUT = 0.0

    # Config for HiCroPL
    cfg.TRAINER.HICROPLReason = CN()
    cfg.TRAINER.HICROPLReason.TRM_START_LAYER = 8 # human layer 3
    cfg.TRAINER.HICROPLReason.TRM_END_LAYER = 11    # human layer 9
    cfg.TRAINER.HICROPLReason.TRM_STEPS = 2
    cfg.TRAINER.HICROPLReason.N_CTX = 2 # number of context vectors
    cfg.TRAINER.HICROPLReason.CROSS_LAYER = 6 # cross layer
    cfg.TRAINER.HICROPLReason.CTX_INIT = "a photo of a" # initialization words (only for language prompts)
    cfg.TRAINER.HICROPLReason.PREC = "fp32"
    cfg.TRAINER.HICROPLReason.PROMPT_DEPTH = 9  # Max 12, minimum 0, for 1 it will act as shallow HICROPL (J=1)
    cfg.TRAINER.HICROPLReason.TEACHER_NAME = "ViT-L/14"
    cfg.TRAINER.HICROPLReason.LAMBD = 12.0
    cfg.TRAINER.HICROPLReason.ALIGN_START = 8
    cfg.TRAINER.HICROPLReason.ALIGN_END = 11
    cfg.TRAINER.HICROPLReason.ALIGN_LAMBDA = 15.0
    cfg.TRAINER.HICROPLReason.ALIGN_T2V_HIDDEN = 1024
    cfg.TRAINER.HICROPLReason.ALIGN_V2T_HIDDEN = 1024
    cfg.TRAINER.HICROPLReason.AD_LAMBDA = 0.5
    cfg.TRAINER.HICROPLReason.AD_TEXT_WEIGHT = 1.0
    cfg.TRAINER.HICROPLReason.AD_VISION_WEIGHT = 1.0
    cfg.TRAINER.HICROPLReason.ALIGN_DROPOUT = 0.0
    # ---------------------------------------------------------
    # Class-Normalized Orthogonal Probe Distillation
    # ---------------------------------------------------------
    cfg.TRAINER.HICROPLReason.PROBE_ENABLE = False  # set True to enable the probe loss

    # Main weight for the full probe loss
    cfg.TRAINER.HICROPLReason.PROBE_LAMBDA = 6.0

    # Internal weights
    cfg.TRAINER.HICROPLReason.PROBE_ANCHOR_WEIGHT = 1.0
    cfg.TRAINER.HICROPLReason.PROBE_PAIR_WEIGHT = 0.5
    cfg.TRAINER.HICROPLReason.PROBE_ORTH_WEIGHT = 0.05
    cfg.TRAINER.HICROPLReason.DAPT_SAVE_PROTOTYPES = True
    cfg.TRAINER.HICROPLReason.DAPT_INTRA_ENABLE = False
    # For orthogonal separation.
    # 0.05 is safer than exact zero, especially for many classes.
    cfg.TRAINER.HICROPLReason.PROBE_ORTH_MARGIN = 0.05

    # Stable initialization from frozen text features
    cfg.TRAINER.HICROPLReason.PROBE_INIT_STD = 0.001
    # ---------------------------------------------------------
    # Prompt-only shared common transformer
    # This does NOT modify pretrained CLIP hidden tokens.
    # It only refines learnable text/vision prompts before injection.
    # ---------------------------------------------------------
    cfg.TRAINER.HICROPLReason.COMMON_PROMPT_ENABLE = True

    # These are the CLIP layer positions whose prompts should be refined.
    # If COMMON_PROMPT_AFTER_LAYER = False:
    #   layer 3 means prompt index 3 is refined and injected at block i=3.
    # If COMMON_PROMPT_AFTER_LAYER = True:
    #   layer 3 means prompt index 4 is refined, equivalent to inserting after layer 3.
    cfg.TRAINER.HICROPLReason.COMMON_PROMPT_LAYERS = [0, 1,2, 3, 4, 5, 6, 7, 8]

    # Recommended for your current request:
    # False = refine prompts before they are injected into layer 3, 6, 10.
    # True  = equivalent to your previous "after 3rd, 6th, 10th layer" hidden-state injection.
    cfg.TRAINER.HICROPLReason.COMMON_PROMPT_AFTER_LAYER = False

    # Common latent space for both text prompts and vision prompts
    cfg.TRAINER.HICROPLReason.COMMON_PROMPT_DIM = 512
    cfg.TRAINER.HICROPLReason.COMMON_PROMPT_HEADS = 8
    cfg.TRAINER.HICROPLReason.COMMON_PROMPT_DEPTH = 1
    cfg.TRAINER.HICROPLReason.COMMON_PROMPT_DROPOUT = 0.0
    cfg.TRAINER.HICROPLReason.COMMON_PROMPT_MIXER_DEPTH = 1
    cfg.TRAINER.HICROPLReason.COMMON_PROMPT_TOKEN_HIDDEN_MULT = 2.0
    cfg.TRAINER.HICROPLReason.COMMON_PROMPT_CHANNEL_HIDDEN_MULT = 4.0

    # Stable residual prompt update
    cfg.TRAINER.HICROPLReason.COMMON_PROMPT_RESIDUAL_SCALE = 1.0
    cfg.TRAINER.HICROPLReason.COMMON_PROMPT_GATE_INIT = -3.0
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    #config for hrmmaple
    cfg.TRAINER.HRMMAPLE = CN()
    cfg.TRAINER.HRMMAPLE.USE_HRM = True
    cfg.TRAINER.HRMMAPLE.USE_VHRM = True
    cfg.TRAINER.HRMMAPLE.H_CYCLES = 2
    cfg.TRAINER.HRMMAPLE.L_CYCLES = 2
    cfg.TRAINER.HRMMAPLE.N_HEADS = 4
    cfg.TRAINER.HRMMAPLE.MLP_RATIO = 4.0
    cfg.TRAINER.HRMMAPLE.V_N_HEADS=8
    cfg.TRAINER.HRMMAPLE.V_MLP_RATIO = 4.0
    cfg.TRAINER.HRMMAPLE.V_H_CYCLES = 2
    cfg.TRAINER.HRMMAPLE.V_L_CYCLES = 2

    cfg.TRAINER.HRMMAPLE.USE_EMA_PROTO = True
    cfg.TRAINER.HRMMAPLE.PROTO_MOMENTUM = 0.90
    cfg.TRAINER.HRMMAPLE.NORM_PROTO = True
    cfg.TRAINER.HRMMAPLE.PREC = "fp16" #"amp"
    cfg.TRAINER.HRMMAPLE.ACT_ENABLE = True
    cfg.TRAINER.HRMMAPLE.ACT_MAX_STEPS = 16
    cfg.TRAINER.HRMMAPLE.ACT_HALT_THRESHOLD = 0.5  # inference threshold
    cfg.TRAINER.HRMMAPLE.ACT_LAMBDA_Q = 1.0        # weight for Q losses
    cfg.TRAINER.HRMMAPLE.ACT_LAMBDA_PONDER = 0.01  # per-step compute penalty (encourages early halt)

    cfg.TRAINER.HRMMAPLE.ACT_EPSILON = 0.05        # optional exploration during training

    #config for HighEncoderMaPLe
    cfg.TRAINER.HighEncoderMaPLe = CN()
    cfg.TRAINER.HighEncoderMaPLe.USE_HRM = True
    cfg.TRAINER.HighEncoderMaPLe.USE_VHRM = True
    cfg.TRAINER.HighEncoderMaPLe.H_CYCLES = 2
    cfg.TRAINER.HighEncoderMaPLe.L_CYCLES = 2
    cfg.TRAINER.HighEncoderMaPLe.N_HEADS = 4
    cfg.TRAINER.HighEncoderMaPLe.MLP_RATIO = 4.0
    cfg.TRAINER.HighEncoderMaPLe.V_N_HEADS=8
    cfg.TRAINER.HighEncoderMaPLe.V_MLP_RATIO = 4.0
    cfg.TRAINER.HighEncoderMaPLe.V_H_CYCLES = 2
    cfg.TRAINER.HighEncoderMaPLe.V_L_CYCLES = 2
    cfg.TRAINER.HighEncoderMaPLe.USE_EMA_PROTO = True
    cfg.TRAINER.HighEncoderMaPLe.PROTO_MOMENTUM = 0.90
    cfg.TRAINER.HighEncoderMaPLe.NORM_PROTO = True
    cfg.TRAINER.HighEncoderMaPLe.PREC = "fp16" #"amp"
    cfg.TRAINER.HighEncoderMaPLe.ACT_ENABLE = True
    cfg.TRAINER.HighEncoderMaPLe.ACT_MAX_STEPS = 16
    cfg.TRAINER.HighEncoderMaPLe.ACT_HALT_THRESHOLD = 0.5  # inference threshold
    cfg.TRAINER.HighEncoderMaPLe.ACT_LAMBDA_Q = 1.0        # weight for Q losses
    cfg.TRAINER.HighEncoderMaPLe.ACT_LAMBDA_PONDER = 0.01  # per-step compute penalty (encourages early halt)
    cfg.TRAINER.HighEncoderMaPLe.ACT_EPSILON = 0.05        # optional exploration during training

    # Config for independent Vision Language prompting (independent-vlp)
    cfg.TRAINER.IVLP = CN()
    cfg.TRAINER.IVLP.N_CTX_VISION = 2  # number of context vectors at the vision branch
    cfg.TRAINER.IVLP.N_CTX_TEXT = 2  # number of context vectors at the language branch
    cfg.TRAINER.IVLP.CTX_INIT = "an example of"  # initialization words (only for language prompts)
    cfg.TRAINER.IVLP.PREC = "fp16"  # fp16, fp32, amp
    # If both variables below are set to 0, 0, will the config will degenerate to COOP model
    cfg.TRAINER.IVLP.PROMPT_DEPTH_VISION = 9 # Max 12, minimum 0, for 0 it will act as shallow MaPLe (J=1)
    cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT = 9  # Max 12, minimum 0, for 0 it will act as shallow MaPLe (J=1)
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    # Config for only vision side prompting
    cfg.TRAINER.VPT = CN()
    cfg.TRAINER.VPT.N_CTX_VISION = 2  # number of context vectors at the vision branch
    cfg.TRAINER.VPT.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.VPT.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.VPT.PROMPT_DEPTH_VISION = 1  # if set to 1, will represent shallow vision prompting only
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    # Config for PromptSRC
    cfg.TRAINER.PROMPTSRC = CN()
    cfg.TRAINER.PROMPTSRC.N_CTX_VISION = 4  # number of context vectors at the vision branch
    cfg.TRAINER.PROMPTSRC.N_CTX_TEXT = 4  # number of context vectors at the language branch
    cfg.TRAINER.PROMPTSRC.CTX_INIT = "a picture of a"  # initialization words
    cfg.TRAINER.PROMPTSRC.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.PROMPTSRC.PROMPT_DEPTH_VISION = 9  # Max 12, minimum 0, for 0 it will be using shallow IVLP prompting (J=1)
    cfg.TRAINER.PROMPTSRC.PROMPT_DEPTH_TEXT = 9  # Max 12, minimum 0, for 0 it will be using shallow IVLP prompting (J=1)
    cfg.TRAINER.PROMPTSRC.TEXT_LOSS_WEIGHT = 25
    cfg.TRAINER.PROMPTSRC.IMAGE_LOSS_WEIGHT = 10
    cfg.TRAINER.PROMPTSRC.GPA_MEAN = 15
    cfg.TRAINER.PROMPTSRC.GPA_STD = 1
    # ---- Angular-distance plots (PromptSRC) ----
    cfg.TRAINER.PROMPTSRC.PLOT_ANGDIST = False         # set True to enable
    cfg.TRAINER.PROMPTSRC.ANGDIST_MAX_BATCHES = 50     # vision: limit batches for speed (None = full loader)
    cfg.TRAINER.PROMPTSRC.ANGDIST_IN_DEGREES = True    # True: degrees, False: radians
    cfg.TRAINER.PROMPTSRC.ANGDIST_SUBDIR = "angdist"   # where to save inside output-dir
    cfg.TRAINER.PROMPTSRC.ANGDIST_SAVE_CSV = True
    cfg.TRAINER.PROMPTSRC.ANGDIST_EPS = 1e-6            # numerical stability for acos/clam
    cfg.TRAINER.PROMPTSRC.ANGDIST_SAVE_PNG = True
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new


    cfg.TRAINER.DAPT = CN()
    cfg.TRAINER.DAPT.VIS_NUM_TOKENS = 16
    cfg.TRAINER.DAPT.VIS_DROPOUT = 0.0
    cfg.TRAINER.DAPT.VIS_BETA = 0.1
    cfg.TRAINER.DAPT.TXT_NUM_TOKENS = 16 
    cfg.TRAINER.DAPT.TXT_RBF_T = 2.0
    cfg.TRAINER.DAPT.TXT_BETA = 0.1

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    cfg.TRAINER.DAPT.PROTOTYPE_GEN = False



    cfg.LOSS = CN()
    cfg.LOSS.GM = False
    cfg.LOSS.NAME = ""
    cfg.LOSS.ALPHA = 0.
    cfg.LOSS.T = 1.
    cfg.LOSS.LAMBDA = 1.
    cfg.OPTIM.EPS = 1e-3
    cfg.TEST.PLOT_ANGDIST = False          # set True when you want plots
    cfg.TEST.ANGDIST_MAX_BATCHES = -1      # -1 = all test batches (safe default)
    # Calibration metric settings
    cfg.TEST.CALIBRATION_BINS = 20
    cfg.TEST.SAVE_CLASSWISE_CALIBRATION = True

def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg

def report_test_gflops(trainer):
    """
    Profile and print a canonical GFLOPs result.
    """
    profile = trainer.profile_test_gflops()

    print(
        f"[EFFICIENCY] GFLOPs (test): "
        f"{profile['gflops_test']:.6f}"
    )

    print(
        "[EFFICIENCY] GFLOPs protocol: "
        f"batch={profile['batch_size']}, "
        f"input={profile['height']}x{profile['width']}, "
        f"classes={profile['num_classes']}"
    )

    unsupported_ops = profile["unsupported_ops"]

    if unsupported_ops:
        print(
            "[EFFICIENCY] Unsupported FLOP operators: "
            f"{unsupported_ops}"
        )

    return profile

def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)
    # if cfg.TRAINER.DAPT.PROTOTYPE_GEN == False:
    #if args.eval_only:
    #    trainer.load_model(args.model_dir, epoch=args.load_epoch)
    #    trainer.test()
    #    return
    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)

        # Run normal evaluation first so evaluator metrics are printed first.
        trainer.test()

        # Print GFLOPs after evaluator output. This is important because
        # parse_test_res.py --test-log starts parsing after evaluator output.
        if args.profile_gflops:
            report_test_gflops(trainer)

        return

    """if not args.no_train:
        start = time.time()
        trainer.train()
        torch.cuda.synchronize()
        end = time.time()
        total_time = end - start
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)  # in GB
        else:
            peak_mem = 0.0
        print(f"[PROFILE] Total train time: {total_time:.1f}s "
          f"({total_time/60:.2f} min), peak GPU memory: {peak_mem:.2f} GB")"""
    if not args.no_train:
        cuda_enabled = torch.cuda.is_available() and cfg.USE_CUDA

        if cuda_enabled:
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        trainer.train()

        if cuda_enabled:
            torch.cuda.synchronize()

        elapsed_seconds = time.perf_counter() - start_time
        elapsed_minutes = elapsed_seconds / 60.0

        if cuda_enabled:
            peak_mem_gb = (
                torch.cuda.max_memory_allocated() / (1024 ** 3)
            )
        else:
            peak_mem_gb = 0.0

        # Keep your existing human-readable profile line.
        print(
            f"[PROFILE] Total train time: "
            f"{elapsed_seconds:.1f}s "
            f"({elapsed_minutes:.2f} min), "
            f"peak GPU memory: {peak_mem_gb:.2f} GB"
        )

        # Add a stable line specifically intended for the parser.
        print(
            f"[EFFICIENCY] Train time (min): "
            f"{elapsed_minutes:.6f}"
        )

        # Run GFLOP analysis after timing so profiling overhead is not
        # included in the reported training duration.
        if args.profile_gflops:
            report_test_gflops(trainer)         

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    parser.add_argument(
        "--profile-gflops",
        action="store_true",
        help="profile and print test-time GFLOPs using batch size 1",
    )
    args = parser.parse_args()
    main(args)
