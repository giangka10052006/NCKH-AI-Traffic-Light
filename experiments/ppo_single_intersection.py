"""
Huan luyen PPO dieu khien den tin hieu tai nut giao J1 (RL.net.xml).

CAC LOI DA SUA SO VOI BAN GOC:
1. tlLogic trong RL.net.xml co lan ky tu 'y' trong tung pha -> sumo-rl khong
   nhan dien duoc pha xanh nao ca (green_phases = []), khien action_space =
   Discrete(0) va agent khong the dieu khien den. Da sua lai file net
   (xem RL_net_fixed.xml) voi 4 pha xanh "sach", sumo-rl se tu sinh pha vang
   chuyen tiep dua tren yellow_time.
2. reward_fn mac dinh "diff-waiting-time" chi phan anh thoi gian cho, khong
   truc tiep phat hien un tac (so xe xep hang). Bo sung reward ket hop
   waiting-time + queue de agent uu tien giai toa un tac.
3. learning_rate=0.001 cua PPO la qua cao cho bai toan nay (thuong gay hoi tu
   khong on dinh) -> giam ve 3e-4 va bo sung day du hyperparameter
   (n_steps, batch_size, gae_lambda, ent_coef...).
4. Khong chuan hoa observation/reward -> them VecNormalize giup PPO hoc on
   dinh hon voi state gom nhieu dai gia tri khac nhau (one-hot pha, mat do,
   hang doi).
5. Khong co checkpoint/log -> them CheckpointCallback + TensorBoard log de
   theo doi qua trinh hoc (rat can cho bao cao NCKH).
"""

# Import môi trường SUMO-RL để PPO tương tác với mô phỏng giao thông.
from sumo_rl import SumoEnvironment
# Import thuật toán PPO từ Stable-Baselines3.
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
import argparse  # dùng để nhận tham số từ dòng lệnh
import os
import sys
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# -------------- OLD
# def combined_congestion_reward(ts):
#     """Reward ket hop: giam thoi gian cho tich luy VA giam so xe xep hang.

#     - diff-waiting-time: giu nguyen y tuong goc cua sumo-rl (thuong on dinh,
#       da duoc kiem chung trong nhieu paper).
#     - phat them (penalty) theo tong so xe dang xep hang tai nut giao, de
#       agent hoc uu tien "giai phong" nut giao thay vi chi toi thieu hoa
#       waiting-time cong don.
#     """
#     if not hasattr(ts, 'last_measure'):
#         ts.last_measure = 0.0
#     ts_wait = sum(ts.get_accumulated_waiting_time_per_lane()) / 100.0
#     diff_wait_reward = ts.last_measure - ts_wait
#     ts.last_measure = ts_wait

#     queue = ts.get_total_queued()
#     queue_penalty = 0.2 * queue

#     return diff_wait_reward - queue_penalty

# ---------------------- NEW
# def combined_congestion_reward(ts):
#     """
#     Reward gồm 4 thành phần:
#     1. Giảm accumulated waiting time.
#     2. Giảm tổng số xe đang xếp hàng.
#     3. Phạt lane đông nhất.
#     4. Phạt sự mất cân bằng giữa các lane.
#     """

#     # ----- 1. diff waiting time (reward gốc SUMO-RL) -----
#     if not hasattr(ts, "last_measure"):
#         ts.last_measure = 0.0

#     ts_wait = sum(ts.get_accumulated_waiting_time_per_lane()) / 100.0

#     diff_wait_reward = ts.last_measure - ts_wait
#     ts.last_measure = ts_wait

#     # ----- 2. Queue -----
#     total_queue = ts.get_total_queued()

#     # ----- 3. Queue từng lane -----
#     lane_queues = np.array(ts.get_lanes_queue())

#     max_queue = np.max(lane_queues)

#     # ----- 4. Độ mất cân bằng -----
#     imbalance = np.std(lane_queues)

#     reward = (
#         diff_wait_reward
#         - 0.08 * total_queue
#         - 0.80 * max_queue
#         - 0.30 * imbalance
#     )

#     return reward

# ======================================


def combined_congestion_reward(ts):
    if not hasattr(ts, "last_measure"):
        ts.last_measure = 0.0

    # Reward gốc
    ts_wait = sum(ts.get_accumulated_waiting_time_per_lane()) / 100.0
    diff_wait_reward = ts.last_measure - ts_wait
    ts.last_measure = ts_wait

    # Queue của từng lane
    lane_queue = ts.get_lanes_queue()

    # Tổng queue
    total_queue = sum(lane_queue)

    # Lane tắc nhất
    max_queue = max(lane_queue)

    # Phạt mạnh lane tắc nhất
    reward = (
        diff_wait_reward
        - 0.5 * total_queue
        - 0.8 * max_queue
    )

    return reward


def make_env(net_file, route_file, out_csv, num_seconds, use_gui, sumo_seed):
    return SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        out_csv_name=out_csv,
        single_agent=True,
        use_gui=use_gui,
        num_seconds=num_seconds,
        delta_time=5,       # thoi gian (s) giua 2 lan agent ra quyet dinh
        yellow_time=3,      # den vang thuc te 3s
        min_green=8,        # tranh dao pha lien tuc, thieu thuc te
        max_green=40,
        reward_fn=combined_congestion_reward,
        sumo_seed=sumo_seed,
        sumo_warnings=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--net-file", default=str(REPO_ROOT / "sumo" / "RL.net.xml"))
    parser.add_argument(
        "--route-file", default=str(REPO_ROOT / "sumo" / "RL.rou.xml"))
    parser.add_argument("--num-seconds", type=int, default=3600)
    parser.add_argument("--total-timesteps", type=int, default=300_000)
    parser.add_argument("--sumo-seed", default="random")
    args = parser.parse_args()

    models_dir = REPO_ROOT / "models"
    outputs_dir = REPO_ROOT / "outputs"
    logs_dir = REPO_ROOT / "logs"

    os.makedirs(outputs_dir / "demo_moi", exist_ok=True)
    os.makedirs(models_dir / "checkpoints", exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    env = DummyVecEnv(
        [
            lambda: make_env(
                net_file=args.net_file,
                route_file=args.route_file,
                out_csv=str(outputs_dir / "demo_moi" / "ppo"),
                num_seconds=args.num_seconds,
                use_gui=False,
                sumo_seed=args.sumo_seed,
            )
        ]
    )
    # Chuan hoa observation (khong chuan hoa reward de giu nguyen ty le vat ly
    # cua no khi phan tich ket qua sau nay).
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=720,          # = num_seconds / delta_time (1 episode / update)
        batch_size=60,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,        # khuyen khich kham pha, tranh hoi tu som ve 1 pha co dinh
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=str(logs_dir / "ppo_traffic"),
        verbose=1,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=10_000,
        save_path=str(models_dir / "checkpoints"),
        name_prefix="ppo_traffic",
    )

    print("Bat dau huan luyen PPO dieu khien den tin hieu...")
    model.learn(total_timesteps=args.total_timesteps, callback=checkpoint_cb)

    model.save(str(models_dir / "ppo_brain"))
    env.save(str(models_dir / "vecnormalize.pkl"))
    print(f"Da huan luyen xong. Model: {models_dir / 'ppo_brain.zip'}")
    print(f"Thong ke chuan hoa: {models_dir / 'vecnormalize.pkl'}")
    print(
        f"Xem tiến trình học: "
        f"tensorboard --logdir {logs_dir / 'ppo_traffic'}"
    )

    env.close()


if __name__ == "__main__":
    main()
