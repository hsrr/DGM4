import argparse
import copy
import json
import os
import socket
import subprocess
import sys


def find_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def run_cmd(cmd, cwd):
    print("Running:", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with code {result.returncode}: {' '.join(cmd)}")


def parse_last_result(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Result file not found: {path}")
    last = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                last = line
    if last is None:
        raise ValueError(f"Result file is empty: {path}")
    data = json.loads(last)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/map-vepfs/liniuniu/hesirui/DGM4")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--exp_id", required=True, help="log_num used in train.py")
    parser.add_argument("--checkpoint", default="ALBEF_4M.pth")
    parser.add_argument("--train_config", default="configs/train.yaml")
    parser.add_argument("--test_config", default="configs/test.yaml")
    parser.add_argument("--test_epoch", default="best")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_gpu_train", type=int, default=1)
    parser.add_argument("--num_gpu_test", type=int, default=1)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--test_sources", default="guardian,usa_today,washington_post,bbc")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    train_json = os.path.join(args.data_root, "metadata", "train.json")
    val_json = os.path.join(args.data_root, "metadata", "val.json")
    test_json = os.path.join(args.data_root, "metadata", "test.json")

    if not args.skip_train:
        train_port = find_free_port()
        train_cmd = [
            sys.executable,
            "train.py",
            "--config", args.train_config,
            "--output_dir", args.output_dir,
            "--checkpoint", args.checkpoint,
            "--launcher", "pytorch",
            "--rank", "0",
            "--log_num", args.exp_id,
            "--dist-url", f"tcp://127.0.0.1:{train_port}",
            "--world_size", str(args.num_gpu_train),
            "--model_save_epoch", "100",
            "--device", args.device,
            "--data_root", args.data_root,
            "--train_file", train_json,
            "--val_file", val_json,
            "--train_sources", "guardian",
            "--val_sources", "guardian",
        ]
        run_cmd(train_cmd, cwd=repo_root)

    summary = {}
    for source in [s.strip() for s in args.test_sources.split(",") if s.strip()]:
        test_port = find_free_port()
        test_cmd = [
            sys.executable,
            "test.py",
            "--config", args.test_config,
            "--output_dir", args.output_dir,
            "--launcher", "pytorch",
            "--rank", "0",
            "--log_num", args.exp_id,
            "--dist-url", f"tcp://127.0.0.1:{test_port}",
            "--world_size", str(args.num_gpu_test),
            "--test_epoch", args.test_epoch,
            "--device", args.device,
            "--data_root", args.data_root,
            "--train_file", train_json,
            "--val_file", test_json,
            "--val_sources", source,
        ]
        run_cmd(test_cmd, cwd=repo_root)

        eval_dir = os.path.join(args.output_dir, args.exp_id, "evaluation")
        result_file = os.path.join(eval_dir, f"results_{source}.txt")
        data = parse_last_result(result_file)
        # Flatten keys from {"val_AUC_cls": "..."} etc.
        summary[source] = {
            "AUC_cls": data.get("val_AUC_cls"),
            "ACC_cls": data.get("val_ACC_cls"),
            "MAP": data.get("val_MAP"),
            "CF1": data.get("val_CF1"),
        }

    out_dir = os.path.join(args.output_dir, args.exp_id, "evaluation")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "summary_4dgm.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n==== Guardian train -> 4-domain test summary ====")
    for source, metrics in summary.items():
        print(
            f"{source:16s} "
            f"AUC={metrics['AUC_cls']} "
            f"ACC={metrics['ACC_cls']} "
            f"mAP={metrics['MAP']} "
            f"CF1={metrics['CF1']}"
        )
    print(f"\nSaved summary to: {out_path}")


if __name__ == "__main__":
    main()
