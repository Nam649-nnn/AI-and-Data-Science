import pandas as pd

# 1. Đọc file dữ liệu (Ví dụ dùng file đã xử lý độ thưa thớt ở bước trước)
df = pd.read_csv('ml-latest-small/ratings_dense.csv')
print(f"Kích thước dữ liệu bảng ban đầu: {df.shape}")

# 2. Sử dụng .pivot_table() để biến đổi thành ma trận 2 chiều
# Rows (index) là userId, Columns là movieId, Values là rating
user_item_matrix = df.pivot_table(index='userId', columns='movieId', values='rating')

# Lưu ý: Các ô mà user chưa đánh giá phim sẽ bị chuyển thành NaN (Trống)
# Bạn có thể giữ nguyên NaN (nếu dùng thư viện như Surprise) 
# Hoặc điền số 0 vào các ô trống đó bằng lệnh .fillna(0) tùy vào thuật toán:
user_item_matrix_filled = user_item_matrix.fillna(0)

# 3. Kiểm tra kết quả ma trận
print("\n--- Biến đổi ma trận thành công! ---")
print(f"Kích thước ma trận (Số User x Số Phim): {user_item_matrix_filled.shape}")
print("\nHiển thị thử một góc ma trận (10 dòng đầu, 10 cột đầu):")
print(user_item_matrix_filled.iloc[:10, :10])

# (Tùy chọn) Lưu ma trận này ra file nếu cần
user_item_matrix_filled.to_csv('ml-latest-small/user_item_matrix.csv')