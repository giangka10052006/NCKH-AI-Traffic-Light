# Chạy PPO đã huấn luyện để kiểm tra

import argparse
import os
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from sumo_rl import SumoEnvironment
REPO_ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--net-file", default=str(REPO_ROOT / "sumo" / "RL.net.xml"))
    parser.add_argument(
        "--route-file", default=str(REPO_ROOT / "sumo" / "RL.rou.xml"))
    parser.add_argument("--num-seconds", type=int, default=3600)
    parser.add_argument(
        "--model", default=str(REPO_ROOT / "models" / "ppo_brain"))
    parser.add_argument(
        "--vecnorm", default=str(REPO_ROOT / "models" / "vecnormalize.pkl"))
    args = parser.parse_args()
    outputs_dir = REPO_ROOT / "outputs"
    os.makedirs(outputs_dir / "demo_moi", exist_ok=True)

    def _make():
        return SumoEnvironment(
            net_file=args.net_file,
            route_file=args.route_file,
            out_csv_name=str(
                outputs_dir / "demo_moi" / "ppo_test"
            ),
            single_agent=True,
            use_gui=True,
            num_seconds=args.num_seconds,
            delta_time=5,
            yellow_time=3,
            min_green=8,
            max_green=40,
            reward_fn="diff-waiting-time",
            sumo_warnings=False,
        )

    env = DummyVecEnv([_make])
    if os.path.exists(args.vecnorm):
        env = VecNormalize.load(args.vecnorm, env)
        env.training = False
        env.norm_reward = False
    else:
        print(f"Khong tim thay {args.vecnorm}, chay khong chuan hoa obs.")

    print("Dang tai model...")
    model = PPO.load(args.model)

    obs = env.reset()
    waiting_times, queues = [], []
    done = False
    step = 0
    print("AI da san sang dieu khien den giao thong!")

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done_arr, info = env.step(action)
        done = bool(done_arr[0])
        step_info = info[0]
        if "system_total_waiting_time" in step_info:
            waiting_times.append(step_info["system_total_waiting_time"])
        if "system_total_stopped" in step_info:
            queues.append(step_info["system_total_stopped"])
        step += 1

    env.close()

    print("\n=== KET QUA DEMO ===")
    print(f"So buoc mo phong: {step}")
    if waiting_times:
        print(f"Thoi gian cho trung binh he thong: {np.mean(waiting_times):.2f} s")
    if queues:
        print(f"So xe dung cho trung binh (hang doi): {np.mean(queues):.2f} xe")
    print("Chi tiet day du (theo tung buoc) da luu vao outputs/demo_moi/ppo_test*.csv")


if __name__ == "__main__":
    main()