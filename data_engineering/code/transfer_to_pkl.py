import pandas as pd

print("--- Bắt đầu quá trình đóng gói sang file .pkl ---")

# 1. Chuyển đổi file ratings_cleaned
try:
    df_clean = pd.read_csv('ml-latest-small/ratings_cleaned.csv')
    df_clean.to_pickle('ml-latest-small/ratings_cleaned.pkl')
    print("✓ Đã biến đổi thành công 'ratings_cleaned.pkl'")
except FileNotFoundError:
    print("x Không tìm thấy file ratings_cleaned.csv")

# 2. Chuyển đổi file ratings_dense (Dữ liệu đã giảm thưa thớt)
try:
    df_dense = pd.read_csv('ml-latest-small/ratings_dense.csv')
    df_dense.to_pickle('ml-latest-small/ratings_dense.pkl')
    print("✓ Đã biến đổi thành công 'ratings_dense.pkl'")
except FileNotFoundError:
    print("x Không tìm thấy file ratings_dense.csv")

# 3. Chuyển đổi file ma trận 2 chiều user_item_matrix
try:
    # Đối với ma trận, ta nên giữ userId làm index bằng cách chọn index_col=0
    df_matrix = pd.read_csv('ml-latest-small/user_item_matrix.csv', index_col=0)
    df_matrix.to_pickle('ml-latest-small/user_item_matrix.pkl')
    print("✓ Đã biến đổi thành công 'user_item_matrix.pkl'")
except FileNotFoundError:
    print("x Không tìm thấy file user_item_matrix.csv")

print("\n--- Hoàn thành! Kiểm tra danh sách file bên trái của bạn nhé ---")