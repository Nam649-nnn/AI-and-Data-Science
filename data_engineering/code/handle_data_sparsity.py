import pandas as pd

# 1. Đọc dữ liệu (sử dụng file đã làm sạch từ bước trước hoặc file gốc)
df = pd.read_csv('ml-latest-small/ratings_cleaned.csv')
print(f"Dữ liệu ban đầu: {df.shape[0]} dòng, {df['userId'].nunique()} users, {df['movieId'].nunique()} movies.")

# =====================================================================
# VÒNG LẶP LỌC K-CORE (Đảm bảo thỏa mãn đồng thời cả 2 điều kiện)
# =====================================================================
# Lưu ý: Khi xóa một user, số lượt đánh giá của một số bộ phim sẽ bị giảm xuống, 
# và ngược lại. Do đó, ta nên chạy lặp một vài lần để dữ liệu hội tụ hoàn toàn.

for i in range(3): # Chạy lặp 3 lần là đủ để đưa về trạng thái ổn định
    # Bước A: Đếm số phim mỗi user đã đánh giá và lọc các user >= 20 bài đánh giá
    user_counts = df['userId'].value_counts()
    active_users = user_counts[user_counts >= 20].index
    df = df[df['userId'].isin(active_users)]
    
    # Bước B: Đếm số lần mỗi phim được đánh giá và lọc các phim >= 10 bài đánh giá
    movie_counts = df['movieId'].value_counts()
    popular_movies = movie_counts[movie_counts >= 10].index
    df = df[df['movieId'].isin(popular_movies)]

# =====================================================================
# KẾT QUẢ SAU KHI XỬ LÝ SPARSITY
# =====================================================================
print("\n--- Sau khi giảm độ thưa thớt (User >= 20, Movie >= 10) ---")
print(f"Dữ liệu còn lại: {df.shape[0]} dòng")
print(f"Số lượng User còn lại: {df['userId'].nunique()}")
print(f"Số lượng Movie còn lại: {df['movieId'].nunique()}")

# Tính toán lại độ thưa thớt (Sparsity) mới của ma trận
num_users = df['userId'].nunique()
num_movies = df['movieId'].nunique()
total_possible_ratings = num_users * num_movies
sparsity = (1 - (len(df) / total_possible_ratings)) * 100

print(f"Độ thưa thớt hiện tại của ma trận: {sparsity:.2f}%")

# Lưu lại file dữ liệu chất lượng cao này
df.to_csv('ml-latest-small/ratings_dense.csv', index=False)
print("Đã lưu thành công file 'ml-latest-small/ratings_dense.csv'!")