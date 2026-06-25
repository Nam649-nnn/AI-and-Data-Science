import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, KNNWithMeans, SVD
from surprise.model_selection import train_test_split as surprise_split

# =====================================================================
# SECTION  2: BASELINE MODEL & EVALUATION
# =====================================================================

class PopularityBaselineRecommender:
    def __init__(self):
        self.top_movies = None
        self.global_mean = 0
        
    def fit(self, train_df):
        self.global_mean = train_df['rating'].mean()
        
        # Thống kê lượt đánh giá (độ phổ biến) và điểm trung bình của từng phim
        movie_stats = train_df.groupby('movieId').agg(
            rating_mean=('rating', 'mean'),
            rating_count=('rating', 'count')
        ).reset_index()
        
        # Sắp xếp giảm dần theo số lượng đánh giá (độ phổ biến)
        self.top_movies = movie_stats.sort_values(by='rating_count', ascending=False)
        
    def predict_rating(self, user_id, movie_id):
        movie_idx = self.top_movies[self.top_movies['movieId'] == movie_id]
        if not movie_idx.empty:
            return movie_idx['rating_mean'].values[0]
        return self.global_mean

    def recommend_top_n(self, n=10):
        return self.top_movies['movieId'].head(n).tolist()


def evaluate_predictions(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {"RMSE": round(rmse, 4), "MAE": round(mae, 4)}


# =====================================================================
# SECTION  3: ADVANCED MODEL (COLLABORATIVE FILTERING)
# =====================================================================

class AdvancedCollaborativeFiltering:
    def __init__(self):
        self.user_sim_df = None
        self.item_sim_df = None
        self.surprise_model = None
        
    def calculate_cosine_similarity(self, user_item_matrix):
        matrix_filled = user_item_matrix.fillna(0)
        
        # Tính tương đồng Item-Item (Movie-Movie)
        item_sim = cosine_similarity(matrix_filled.T)
        self.item_sim_df = pd.DataFrame(item_sim, index=user_item_matrix.columns, columns=user_item_matrix.columns)
        
        # Tính tương đồng User-User
        user_sim = cosine_similarity(matrix_filled)
        self.user_sim_df = pd.DataFrame(user_sim, index=user_item_matrix.index, columns=user_item_matrix.index)
        
        return self.user_sim_df, self.item_sim_df

    def train_surprise_model(self, df, algorithm='knn'):
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
        
        trainset, testset = surprise_split(data, test_size=0.2, random_state=42)
        
        if algorithm == 'knn':
            sim_options = {'name': 'cosine', 'user_based': False} # Item-based CF
            self.surprise_model = KNNWithMeans(sim_options=sim_options)
        elif algorithm == 'svd':
            self.surprise_model = SVD(random_state=42)
            
        self.surprise_model.fit(trainset)
        return self.surprise_model, testset

    #  HÀM CHUYÊN BIỆT THEO Ý TƯỞNG : TƯƠNG ĐỒNG + PHỔ BIẾN
    def recommend_similar_and_popular(self, current_movie_id, baseline_model, top_k=20, final_n=5):
        """
        Tìm các phim có nội dung/gu tương đồng nhất nhưng đồng thời phải PHỔ BIẾN nhất thị trường.
        """
        # Nếu phim hiện tại nằm ngoài hệ thống ma trận (Phim quá mới) -> trả về luôn top phim hot của hệ thống
        if current_movie_id not in self.item_sim_df.index:
            return baseline_model.recommend_top_n(n=final_n)
            
        # Bước 1: Lấy điểm tương đồng Cosine của phim hiện tại với các phim khác
        similar_scores = self.item_sim_df[current_movie_id].drop(current_movie_id)
        
        # Lọc ra top K phim giống nhất (Ví dụ lấy hẳn top 20 phim cùng gu)
        top_similar_movies = similar_scores.sort_values(ascending=False).head(top_k).reset_index()
        top_similar_movies.columns = ['movieId', 'similarity_score']
        
        # Bước 2: Lấy dữ liệu độ phổ biến (lượt đánh giá) từ mô hình Baseline
        movie_popularity = baseline_model.top_movies[['movieId', 'rating_count']]
        
        # Gộp bảng phim tương đồng với bảng thống kê lượt xem thị trường
        merged_df = pd.merge(top_similar_movies, movie_popularity, on='movieId', how='inner')
        
        # Sắp xếp lại danh sách: Ưu tiên phim có LƯỢT XEM KHỦNG nhất lên đầu
        final_recommendations = merged_df.sort_values(by='rating_count', ascending=False)
        
        # Trả về danh sách N phim cuối cùng xuất sắc nhất để đưa lên giao diện
        return final_recommendations['movieId'].head(final_n).tolist()


# =====================================================================
# KHU VỰC CHẠY THẬT VỚI FILE .PKL CỦA THÀNH VIÊN 1
# =====================================================================
if __name__ == "__main__":
    print("=== TIẾN TRÌNH LUỒNG XỬ LÝ CỦA THÀNH VIÊN 2 & THÀNH VIÊN 3 ===")
    
    folder_path = 'data_engineering/result/' # Thay bằng thư mục chứa file .pkl thật 
    
    try:
        df_dense = pd.read_pickle(f'{folder_path}ratings_dense.pkl')
        user_item_matrix = pd.read_pickle(f'{folder_path}user_item_matrix.pkl')
        print(f"✓ Nạp file .pkl thành công từ '{folder_path}'!")
    except FileNotFoundError:
        print("🚨 LỖI: Hãy kiểm tra lại biến folder_path.")
        exit()

    # 1. Huấn luyện Baseline
    baseline = PopularityBaselineRecommender()
    baseline.fit(df_dense)
    
    # 2. Huấn luyện Advanced Model & Tính toán Cosine
    advanced = AdvancedCollaborativeFiltering()
    advanced.calculate_cosine_similarity(user_item_matrix)
    
    # 3. CHẠY THỬ NGHIỆM TÍNH NĂNG THEO Ý TƯỞNG 
    # Giả sử người dùng đang click xem bộ phim có mã ID = 1 (Phim Toy Story)
    movie_id_dang_xem = 1
    
    phim_goi_y = advanced.recommend_similar_and_popular(
        current_movie_id=movie_id_dang_xem, 
        baseline_model=baseline, 
        top_k=20, 
        final_n=5
    )
    
    print(f"\n🎬 Người dùng đang xem phim có ID: {movie_id_dang_xem}")
    print(f"🔥 Hệ thống  gợi ý 5 phim VỪA TƯƠNG ĐỒNG VỪA PHỔ BIẾN NHẤT: {phim_goi_y}")
