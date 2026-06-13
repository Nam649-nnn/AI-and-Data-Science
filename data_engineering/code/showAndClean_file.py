import pandas as pd

df_ratings = pd.read_csv('ml-latest-small/ratings.csv')


so_luong_user = df_ratings['userId'].nunique()

print(f"Số lượng user duy nhất trong bộ dữ liệu là: {so_luong_user}")

print(df_ratings.head())
print(f"Tổng số lượt đánh giá phim: {len(df_ratings)}")

print(df_ratings['userId'].value_counts())

df_titles = pd.read_csv('ml-latest-small/movies.csv')

so_luong_phim = df_titles['title'].nunique()

print(f"Số lượng phim duy nhất trong bộ dữ liệu là: {so_luong_phim}")

print(df_titles.head())
print(f"Tổng số phim trong bộ dữ liệu: {len(df_titles)}")

print(df_ratings['movieId'].value_counts())


print(f"Số lượng dòng ban đầu: {len(df_ratings)}")

# BƯỚC 1: XỬ LÝ DỮ LIỆU BỊ KHUYẾT (MISSING VALUES)
# Nếu một dòng bị thiếu userId hoặc movieId hoặc rating, dòng đó vô giá trị.
# Ta sẽ xóa thẳng tay những dòng bị khuyết này.
df_ratings = df_ratings.dropna(subset=['userId', 'movieId', 'rating'])


# BƯỚC 2: LOẠI BỎ DỮ LIỆU TRÙNG LẶP (DUPLICATES)
# Nếu một user đánh giá trùng một bộ phim nhiều lần, ta chỉ giữ lại lượt đánh giá cuối cùng
df_ratings = df_ratings.drop_duplicates(subset=['userId', 'movieId'], keep='last')


# BƯỚC 3: SỬA ĐỔI KIỂU DỮ LIỆU CHUẨN (DATA TYPES)
df_ratings['userId'] = df_ratings['userId'].astype(int)
df_ratings['movieId'] = df_ratings['movieId'].astype(int)
df_ratings['rating'] = df_ratings['rating'].astype(float)


# ==========================================
# BƯỚC 4: LỌC DỮ LIỆU HỢP LỆ (OUTLIERS / LOGICAL ERRORS)
# ==========================================
# Chỉ giữ lại các dòng có điểm rating từ 0.5 đến 5.0
dieu_kien_hop_le = (df_ratings['rating'] >= 0.5) & (df_ratings['rating'] <= 5.0)
df_ratings = df_ratings[dieu_kien_hop_le]


# ==========================================
# KẾT QUẢ SAU KHI LÀM SẠCH
# ==========================================
print("\n--- Hoàn tất làm sạch dữ liệu! ---")
print(f"Số lượng dòng còn lại: {len(df_ratings)}")

# Xuất dữ liệu đã làm sạch ra một file mới để dùng cho các bước sau
df_ratings.to_csv('ml-latest-small/ratings_cleaned.csv', index=False)
print("Đã lưu file dữ liệu sạch thành 'ml-latest-small/ratings_cleaned.csv'")