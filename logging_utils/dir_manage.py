import pathlib
import os
import logging

def _run_dir_exists(run_base_dir):
    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"

    return log_base_dir.exists() or ckpt_base_dir.exists()

def get_directories(args):
    run_base_dir = pathlib.Path(
        f"results/{args.method}/{args.coreset_size}/{args.outer_lr}"
    )
    args.rep_count = "/"
    if _run_dir_exists(run_base_dir):
        rep_count = 0
        while _run_dir_exists(run_base_dir / str(rep_count)):
            rep_count += 1
        args.rep_count = "/" + str(rep_count)
        run_base_dir = run_base_dir / str(rep_count)

    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"

    dirs = [run_base_dir, log_base_dir, ckpt_base_dir]
    for dir in dirs:
        if not dir.exists():
            os.makedirs(dir)

    logging.basicConfig(filename=os.path.join(log_base_dir, "loginfo.log"), filemode='w', format='%(asctime)s : %(levelname)s  %(message)s', datefmt='%Y-%m-%d %A %H:%M:%S' , level=logging.INFO)
    for arg, value in sorted(vars(args).items()):
        logging.info("Argument %s: %r", arg, value)

    (run_base_dir / "settings.txt").write_text(str(args))

    return run_base_dir, ckpt_base_dir, log_base_dir