# Chạy SUMO-GUI từ cấu hình SUMO
import argparse
import os
import sys
from pathlib import Path

# Đảm bảo thư mục gốc của dự án nằm trong biến môi trường để tìm thấy thư viện sumo_rl
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# === CÁC THƯ VIỆN CỐT LÕI ĐÃ ĐƯỢC BỔ SUNG ===
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sumo_rl import SumoEnvironment
# ===========================================

def main():
    # 1. Cấu hình nhận lệnh từ Terminal
    parser = argparse.ArgumentParser()
    parser.add_argument("--net-file", default="RL.net.xml")
    parser.add_argument("--route-file", default="RL.rou.xml")
    parser.add_argument("--out-csv", default="ppo_gui_test")
    args = parser.parse_args()

    # 2. Xử lý đường dẫn tương đối tự động
    # Lấy tự động thư mục chứa chính file Python này (tức là thư mục 'experiments')
    experiments_dir = Path(__file__).resolve().parent
    
    # Từ thư mục gốc đó, trỏ tới các file cấu hình tương ứng
    net_file_path = experiments_dir / args.net_file
    route_file_path = experiments_dir / args.route_file
    
    net_file = str(net_file_path)
    route_file = str(route_file_path)
    
    print(f"Net file: {net_file}")
    print(f"Route file: {route_file}")

    # Tạo thư mục lưu kết quả tương đối
    outputs_dir = experiments_dir / "outputs" / "demo_moi"
    os.makedirs(outputs_dir, exist_ok=True)

    def _make():
        return SumoEnvironment(
            net_file=net_file,
            route_file=route_file,
            out_csv_name=str(outputs_dir / args.out_csv),
            single_agent=True,
            use_gui=True,
            num_seconds=9999,
            delta_time=5,
            yellow_time=3,
            min_green=8,
            max_green=40,
            reward_fn="diff-waiting-time", 
            sumo_warnings=False,
        )

    env = DummyVecEnv([_make])
    
    # 3. Trỏ tới model bằng đường dẫn tương đối
    models_dir = experiments_dir / "models"
    vecnorm_path = models_dir / "vecnormalize.pkl"
    model_path = models_dir / "ppo_brain.zip"

    if vecnorm_path.exists():
        env = VecNormalize.load(str(vecnorm_path), env)
        env.training = False
        env.norm_reward = False
    else:
        print(f"Không tìm thấy {vecnorm_path}, chạy mà không chuẩn hóa obs")

    print(f"Đang tải mô hình: {model_path}")
    model = PPO.load(str(model_path))

    obs = env.reset()
    done = False
    step = 0
    episode = 1
    print("AI đã sẵn sàng điều khiển đèn giao thông. Cửa sổ SUMO GUI đã mở...")
    print("Bạn có thể dùng các nút Play/Pause/Step trên cửa sổ GUI để thao tác trực tiếp. Nhấn Ctrl+C trên Terminal để dừng.")

    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, info = env.step(action)
            done = bool(done_arr[0])
            step += 1
            
            if step % 50 == 0:
                print(f"Đã chạy {step} bước ở tập (episode) {episode}")

            # --- ĐOẠN CODE KIỂM TRA LƯỢNG XE ---
            # Trích xuất môi trường SUMO gốc từ lớp bọc DummyVecEnv
            base_env = env.envs[0].unwrapped
            
            # Nếu số lượng xe trên đường + xe chưa sinh ra == 0
            if base_env.sumo.simulation.getMinExpectedNumber() == 0:
                print(f"Tuyệt vời! Toàn bộ xe đã thoát hết ở bước {step}.")
                print(f"Tập {episode} hoàn thành xuất sắc. Đang chuẩn bị chạy lại...")
                obs = env.reset()
                done = False
                step = 0
                episode += 1
                continue # Bỏ qua các lệnh bên dưới, bắt đầu tập mới
            # ------------------------------------------

            if done:
                print(f"Tập {episode} kết thúc do hết thời gian. Đang chuẩn bị chạy lại...")
                obs = env.reset()
                done = False
                step = 0
                episode += 1
                
    except KeyboardInterrupt:
        print("\nĐã dừng mô phỏng bằng Ctrl+C")

if __name__ == "__main__":
    main()