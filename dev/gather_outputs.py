

import os
from pathlib import Path

def run_gather_batch(name,num,ddpm_name,subname):
    ddpm_base = ddpm_name.split("/")[-1]
    dest_root = Path("diff_output") / ddpm_base / subname / name / "gather"
    if not dest_root.exists():
        dest_root.mkdir(parents=True)
    for i in range(num):
        exp_name = name + "_%02d" % i
        save_root = Path("diff_output") / ddpm_base / subname / exp_name
        save_root_i = save_root / "images"
        # print(save_root_i)
        # continue
        fns = [fn for fn in save_root_i.iterdir() if fn.suffix == ".png"]
        for src_fn in fns:
            prefix = "%02d_"%i
            dest_fn = dest_root / (prefix + src_fn.name)
            src_fn = src_fn.resolve()
            # print(src_fn,dest_fn)
            # break
            os.symlink(str(src_fn),str(dest_fn))

def main():
    B = 500
    N = int((1e3-1)//B+1)
    # N = int((1e4-1)//B+1)

    # -- v0 --
    # ddpm_name = "google/ddpm-cifar10-32"
    # subname = "batched_v0"
    # run_gather_batch("cond_dev",N,ddpm_name,subname)
    # run_gather_batch("stand_dev",N,ddpm_name,subname)

    ddpm_name = "google/cifar10_v0"
    subname = "batched_v1"
    run_gather_batch("cond_dev",N,ddpm_name,subname)
    run_gather_batch("stand_dev",N,ddpm_name,subname)

if __name__ == "__main__":
    main()
