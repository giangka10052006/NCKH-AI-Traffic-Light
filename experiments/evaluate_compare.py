# Đánh giá/so sánh kết quả
"""
So sanh dinh luong: PPO (RL) vs den tin hieu co dinh (fixed-time), cung mot
kich ban xe (cung sumo_seed) de dam bao cong bang. Chay khong GUI de nhanh.

Dung cho phan "danh gia hieu qua" trong bao cao NCKH: xuat trung binh
thoi gian cho he thong, hang doi trung binh, toc do trung binh cho ca 2
phuong an dieu khien.
"""

import argparse
import os

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from sumo_rl import SumoEnvironment


def run_fixed_time(net_file, route_file, num_seconds, seed):
    """Chay voi fixed_ts=True: SUMO se dung dung tlLogic goc trong net file
    (chu ky den co dinh) thay vi de agent dieu khien -> dung lam baseline."""
    env = SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        out_csv_name="outputs/demo_moi/fixed_time",
        single_agent=True,
        use_gui=False,
        num_seconds=num_seconds,
        fixed_ts=True,
        sumo_seed=seed,
        sumo_warnings=False,
    )
    records = []
    env.reset()
    done = False
    # fixed_ts=True: action bi bo qua, truyen action bat ky hop le
    while not done:
        _, _, terminated, truncated, info = env.step({})
        done = terminated or truncated
        records.append(info)
    env.close()
    return pd.DataFrame(records)


def run_ppo(net_file, route_file, num_seconds, seed, model_path, vecnorm_path):
    def _make():
        return SumoEnvironment(
            net_file=net_file,
            route_file=route_file,
            out_csv_name="outputs/demo_moi/ppo_eval",
            single_agent=True,
            use_gui=False,
            num_seconds=num_seconds,
            delta_time=5,
            yellow_time=3,
            min_green=8,
            max_green=40,
            reward_fn="diff-waiting-time",
            sumo_seed=seed,
            sumo_warnings=False,
        )

    env = DummyVecEnv([_make])
    if os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False

    model = PPO.load(model_path)
    obs = env.reset()
    records = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done_arr, info = env.step(action)
        done = bool(done_arr[0])
        records.append(info[0])
    env.close()
    return pd.DataFrame(records)


def summarize(df, label):
    print(f"\n--- {label} ---")
    print(f"Thoi gian cho trung binh he thong : {df['system_mean_waiting_time'].mean():.2f} s")
    print(f"So xe dung cho trung binh          : {df['system_total_stopped'].mean():.2f} xe")
    print(f"Toc do trung binh he thong         : {df['system_mean_speed'].mean():.2f} m/s")
    return {
        "phuong_an": label,
        "avg_waiting_time_s": df["system_mean_waiting_time"].mean(),
        "avg_queue": df["system_total_stopped"].mean(),
        "avg_speed_mps": df["system_mean_speed"].mean(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--net-file", default="RL.net.xml")
    parser.add_argument("--route-file", default="RL.rou.xml")
    parser.add_argument("--num-seconds", type=int, default=3600)
    parser.add_argument("--seed", default=42)
    parser.add_argument("--model", default="models/ppo_brain")
    parser.add_argument("--vecnorm", default="models/vecnormalize.pkl")
    args = parser.parse_args()

    os.makedirs("outputs/demo_moi", exist_ok=True)

    print("Dang chay kich ban FIXED-TIME (baseline)...")
    df_fixed = run_fixed_time(args.net_file, args.route_file, args.num_seconds, args.seed)

    print("Dang chay kich ban PPO (RL)...")
    df_ppo = run_ppo(
        args.net_file, args.route_file, args.num_seconds, args.seed, args.model, args.vecnorm
    )

    row_fixed = summarize(df_fixed, "Den co dinh (baseline)")
    row_ppo = summarize(df_ppo, "PPO (RL)")

    improve = (
        (row_fixed["avg_waiting_time_s"] - row_ppo["avg_waiting_time_s"])
        / max(row_fixed["avg_waiting_time_s"], 1e-6)
        * 100
    )
    print(f"\n>>> PPO giam thoi gian cho trung binh {improve:.1f}% so voi den co dinh")

    out = pd.DataFrame([row_fixed, row_ppo])
    out_path = "outputs/demo_moi/so_sanh_ppo_vs_fixed.csv"
    out.to_csv(out_path, index=False)
    print(f"Da luu bang so sanh vao: {out_path}")


if __name__ == "__main__":
    main()