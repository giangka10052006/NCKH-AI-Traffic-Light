# Kiểm tra giao tiếp Python ↔ SUMO qua TraCI
import traci # Thư viện lõi để Python điều khiển SUMO
import time

# --- YÊU CẦU 1: KẾT NỐI PYTHON VỚI SUMO ---
# Chỉnh sửa đường dẫn tới 2 file bản đồ của ngã tư đơn giản
sumoCmd = [
    "sumo-gui", # Dùng sumo-gui để hiện 3D, nếu muốn chạy ngầm nhanh thì đổi thành "sumo"
    "-n", "sumo_rl/nets/single-intersection/single-intersection.net.xml", 
    "-r", "sumo_rl/nets/single-intersection/single-intersection.rou.xml"
]

# Khởi động kết nối
traci.start(sumoCmd)
print("Đã kết nối thành công với SUMO!")

step = 0
# Chạy vòng lặp mô phỏng
step = 0
# Chạy vòng lặp mô phỏng
while step < 1000:
    traci.simulationStep() 
    
    tl_id = "t"       
    edge_tay = "w_t" # Đại diện trục Đông - Tây
    edge_bac = "n_t" # Đại diện trục Bắc - Nam

    try:
        # Lấy TỔNG SỐ XE đang có trên cả 2 trục
        xe_truc_ngang = traci.edge.getLastStepVehicleNumber(edge_tay)
        xe_truc_doc = traci.edge.getLastStepVehicleNumber(edge_bac)
        
        # In ra màn hình để theo dõi biến động
        print(f"Giây {step} | Trục Ngang: {xe_truc_ngang} xe --- Trục Dọc: {xe_truc_doc} xe")

        # THUẬT TOÁN CÂN BẰNG TẢI
        # Dùng phép chia lấy dư (%), cứ đúng 15 giây hệ thống mới được phép xem xét đổi đèn 1 lần
        if step % 15 == 0:
            if xe_truc_ngang >= xe_truc_doc:
                print(">>> QUYẾT ĐỊNH: Trục Ngang đông hơn. Cấp XANH 15 giây!")
                traci.trafficlight.setPhase(tl_id, 2) # Xanh cho Đông-Tây
            else:
                print(">>> QUYẾT ĐỊNH: Trục Dọc đông hơn. Cấp XANH 15 giây!")
                traci.trafficlight.setPhase(tl_id, 0) # Xanh cho Bắc-Nam
                
    except traci.exceptions.TraCIException as e:
        print("Lỗi ID:", e)
        break

    step += 1
    time.sleep(0.1) 

traci.close()