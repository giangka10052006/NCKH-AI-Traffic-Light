import os
import sys
from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy

if __name__ == "__main__":
    # BƯỚC MỚI: Ép máy tính tự động tạo thư mục nếu chưa có để tránh mất file
    os.makedirs("outputs/nga_tu_so", exist_ok=True)

    # 1. Khởi tạo môi trường Ngã Tư Sở
    env = SumoEnvironment(
        net_file="sumo_rl/nets/nga-tu-so/osm.net.xml",
        route_file="sumo_rl/nets/nga-tu-so/osm.passenger.trips.xml",
        out_csv_name="outputs/nga_tu_so/ql",
        use_gui=True, # Bật True để xem 3D, đổi thành False nếu muốn AI chạy ngầm cực nhanh
        num_seconds=3600,
        ts_ids=["cluster_1314209709_5683715152_5716408771_5716408772_#4more"],
        single_agent=True,
    )

    ts_id = list(env.ts_ids)[0]
    reset_result = env.reset()
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result

    exploration = EpsilonGreedy(initial_epsilon=0.05, min_epsilon=0.005, decay=1.0)

    # 2. Thiết lập AI
    agent = QLAgent(
        starting_state=env.encode(obs, ts_id),
        state_space=env.observation_space,
        action_space=env.action_space,
        alpha=0.1,
        gamma=0.99,
        exploration_strategy=exploration,
    )

    done = False
    print(f"\nBắt đầu huấn luyện AI tại cụm đèn: {ts_id}")

    # 3. Bắt đầu vòng lặp huấn luyện
    step_count = 0
    while not done:
        action = agent.act()

        step_result = env.step(action)
        if len(step_result) == 5:
            next_obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            next_obs, reward, done, info = step_result

        agent.learn(next_state=env.encode(next_obs, ts_id), reward=reward)

        # In tiến độ ra Terminal để dễ theo dõi
        step_count += 1
        if step_count % 100 == 0:
            print(f"Đang học... Đã mô phỏng được {step_count} bước giao thông")

    # 4. Lưu và đóng
    env.close()
    print("\n✅ Mô phỏng hoàn tất! Dữ liệu đã được lưu CHẮC CHẮN vào thư mục outputs/nga_tu_so/")