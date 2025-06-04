"""
Amazon レビュー分析プロジェクト - データ収集・前処理モジュール

このファイルの目的：
1. Amazon商品レビューのサンプルデータを生成する
2. データをクリーニング・前処理する  
3. 感情分析用のラベル（positive/negative/neutral）を付ける
4. CSVファイルで保存・読み込みできるようにする
5. データの統計情報を分析・表示する

"""

import kaggle
import pandas as pd         
import numpy as np   
import os             
from typing import Dict, List, Optional 
from tqdm import tqdm  
import json          
from datetime import datetime, timedelta


class AmazonReviewCollector:
    """
    Amazon レビューデータ収集・前処理クラス
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        クラスの初期化（インスタンス作成時に最初に実行される）
        
        Args:
            data_dir (str): データファイルを保存するディレクトリ名
                          デフォルトは "data" フォルダ
        """
        
        # データ保存ディレクトリの設定
        self.data_dir = data_dir
        self.ensure_data_dir()  # フォルダが存在しない場合は作成
        
        # 商品カテゴリの定義（分析対象を明確に）
        self.target_categories = {
            "electronics": ["充電器", "イヤホン", "ケーブル", "バッテリー"],
            "books": ["プログラミング本", "ビジネス書", "自己啓発本"],
            "daily_goods": ["キッチン用品", "掃除用具", "収納ボックス"]
        }
        
        # 好評レビューのパターン（高評価の場合に使用）
        self.positive_patterns = [
            "とても良い商品です。品質が高く、長く使えそうです。",
            "期待以上の性能でした。買って良かったです。",
            "デザインが素晴らしく、使いやすいです。",
            "コストパフォーマンスが最高です。おすすめします。",
            "配送も早く、梱包も丁寧でした。また利用したいです。",
            "思っていたより良い品質で満足しています。",
            "値段の割に機能が充実していて驚きました。",
            "使いやすくて、毎日愛用しています。"
        ]
        
        # 定評レビューのパターン（低評価の場合に使用）
        self.negative_patterns = [
            "品質が期待より低く、すぐに壊れました。",
            "説明と実際の商品が違いました。残念です。",
            "価格に見合わない品質だと思います。",
            "使いにくく、期待していたより不便です。",
            "充電速度が遅く、時間がかかりすぎます。",
            "音質が悪く、ノイズが気になります。",
            "届いた商品に傷がありました。",
            "サイズが思っていたより小さくて使いづらいです。"
        ]
        
        # 普通レビューのパターン
        self.neutral_patterns = [
            "普通の商品だと思います。可もなく不可もなく。",
            "値段相応の品質です。特に問題はありません。",
            "期待していた通りの商品でした。",
            "使えますが、特別良いというわけではありません。",
            "標準的な商品です。必要最低限の機能はあります。"
        ]
    
    def ensure_data_dir(self):
        """
        データディレクトリの存在確認・作成
        
        プログラムが実行される前に、データ保存用のフォルダが
        存在するかチェックし、なければ作成
        """
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)  # フォルダを作成
            print(f"データディレクトリを作成しました: {self.data_dir}")
    
    def setup_kaggle_api(self):
        """
        Kaggle API の設定
        kaggle.json ファイルを適切な場所に配置
        """
        import os
        import shutil
        
        # Kaggle 認証ディレクトリの準備
        kaggle_dir = os.path.expanduser("~/.kaggle")
        if not os.path.exists(kaggle_dir):
            os.makedirs(kaggle_dir)
        
        # kaggle.json ファイルのコピー（プロジェクトルートから）
        if os.path.exists("kaggle.json"):
            shutil.copy("kaggle.json", os.path.join(kaggle_dir, "kaggle.json"))
            os.chmod(os.path.join(kaggle_dir, "kaggle.json"), 0o600)
            print("Kaggle API認証設定完了")
        else:
            print("kaggle.json が見つかりません")
    
    def load_real_kaggle_dataset(self, dataset_name: str = "snap/amazon-fine-food-reviews") -> pd.DataFrame:
        """
        実際のKaggleデータセットを取得・処理
        
        Args:
            dataset_name (str): Kaggleデータセット名
            
        Returns:
            pd.DataFrame: 実データ
        """
        try:
            print(f"Kaggleから実データを取得中: {dataset_name}")
            
            # Kaggle API設定
            self.setup_kaggle_api()
            
            # データセットダウンロード
            download_path = os.path.join(self.data_dir, "kaggle_raw")
            os.makedirs(download_path, exist_ok=True)
            
            # kaggle datasets download
            kaggle.api.dataset_download_files(
                dataset_name, 
                path=download_path, 
                unzip=True
            )
            
            # CSVファイルを探して読み込み
            csv_files = [f for f in os.listdir(download_path) if f.endswith('.csv')]
            
            if not csv_files:
                raise FileNotFoundError("CSVファイルが見つかりません")
            
            # 最初のCSVファイルを読み込み
            csv_path = os.path.join(download_path, csv_files[0])
            df_raw = pd.read_csv(csv_path)
            
            print(f"実データ取得完了: {len(df_raw)}件")
            print(f"   ファイル: {csv_files[0]}")
            print(f"   列: {list(df_raw.columns)}")
            
            # Amazon レビューフォーマットに変換
            df_converted = self._convert_kaggle_to_amazon_format(df_raw)
            
            return df_converted
            
        except Exception as e:
            print(f"Kaggle API エラー: {e}")
            print("サンプルデータで代替します...")
            return self._generate_sample_data()
    
    
    def _convert_kaggle_to_amazon_format(self, df_kaggle: pd.DataFrame) -> pd.DataFrame:
        """
        Kaggleデータを統一フォーマットに変換
        
        Args:
            df_kaggle (pd.DataFrame): Kaggle生データ
            
        Returns:
            pd.DataFrame: 統一フォーマットデータ
        """
        print("データフォーマット変換中...")
        
        # 列名のマッピング（データセットに応じて調整）
        column_mapping = {
            # Amazon Fine Food Reviews の場合
            'Id': 'review_id',
            'ProductId': 'product_id', 
            'UserId': 'user_id',
            'Score': 'rating',
            'Summary': 'review_title',
            'Text': 'review_text',
            'Time': 'review_date'
        }
        
        # 列名変更
        df_converted = df_kaggle.rename(columns=column_mapping)
        
        # 必要な列の確認・追加
        required_columns = ['product_id', 'rating', 'review_text']
        missing_columns = [col for col in required_columns if col not in df_converted.columns]
        
        if missing_columns:
            print(f"⚠️ 不足列を補完: {missing_columns}")
            
            # 不足列の補完
            if 'product_id' in missing_columns:
                df_converted['product_id'] = df_converted.index.map(lambda x: f"B{x:06d}")
            if 'rating' in missing_columns:
                df_converted['rating'] = 5  # デフォルト値
            if 'review_text' in missing_columns:
                df_converted['review_text'] = "No review text available"
        
        # 商品名・カテゴリの推定（実データに基づく）
        df_converted['product_name'] = self._estimate_product_names(df_converted)
        df_converted['product_category'] = self._estimate_categories(df_converted)
        
        # 日付の処理
        if 'review_date' in df_converted.columns:
            # Unix時間を正しく変換
            df_converted['review_date'] = pd.to_datetime(df_converted['review_date'], unit='s', errors='coerce')
        # unit='s' を指定してUnix秒を正しく変換
        elif 'Time' in df_converted.columns:
            # 元のTime列からの変換
            df_converted['review_date'] = pd.to_datetime(df_converted['Time'], unit='s', errors='coerce')
        else:
            # ランダムな日付生成
            df_converted['review_date'] = pd.date_range(
                start='2023-01-01', 
                end='2024-12-31', 
                periods=len(df_converted)
            )
        
        # メタデータの追加
        df_converted['helpful_votes'] = np.random.randint(0, 20, len(df_converted))
        df_converted['total_votes'] = df_converted['helpful_votes'] + np.random.randint(0, 10, len(df_converted))
        df_converted['verified_purchase'] = np.random.choice([True, False], len(df_converted), p=[0.8, 0.2])
        
        # データ型の調整
        df_converted['rating'] = pd.to_numeric(df_converted['rating'], errors='coerce')
        df_converted = df_converted.dropna(subset=['rating'])
        df_converted['rating'] = df_converted['rating'].astype(int)
        
        # 評価の範囲調整（1-5に正規化）
        df_converted['rating'] = df_converted['rating'].clip(1, 5)
        
        print(f"フォーマット変換完了: {len(df_converted)}件")
        return df_converted
    
    def _estimate_product_names(self, df: pd.DataFrame) -> pd.Series:
        """実データから商品名を推定（改善版）"""
        
        def extract_product_name(text, summary="", product_id=""):
            if pd.isna(text):
                text = ""
            if pd.isna(summary):
                summary = ""
                
            text_lower = str(text).lower()
            summary_lower = str(summary).lower()
            
            # 1. サマリーから商品名抽出（より具体的）
            if summary and len(summary) > 3:
                # サマリーは商品名に近い場合が多い
                return str(summary)[:50]  # 長すぎる場合は短縮
            
            # 2. レビューテキストからキーワード抽出
            product_keywords = {
                'coffee': ['coffee', 'espresso', 'latte', 'cappuccino'],
                'tea': ['tea', 'green tea', 'black tea', 'herbal'],
                'snack': ['chips', 'crackers', 'nuts', 'cookies'],
                'candy': ['candy', 'chocolate', 'gum', 'sweet'],
                'food': ['sauce', 'pasta', 'rice', 'cereal']
            }
            
            for category, keywords in product_keywords.items():
                for keyword in keywords:
                    if keyword in text_lower:
                        return f"{keyword.title()}"
            
            # 3. ProductIDの活用
            if product_id:
                return f"Product {product_id}"
            
            return "Unknown Food Product"
        
        # 複数列を活用
        if 'review_text' in df.columns:
            return df.apply(lambda row: extract_product_name(
                row.get('review_text', ''),
                row.get('review_title', ''),
                row.get('product_id', '')
            ), axis=1)
        else:
            return pd.Series(["Unknown Product"] * len(df))

    def _estimate_categories(self, df: pd.DataFrame) -> pd.Series:
        """実データからカテゴリを推定"""
        
        def categorize_review(text, summary=""):
            if pd.isna(text):
                text = ""
            if pd.isna(summary):
                summary = ""
            
            text_lower = str(text).lower() + " " + str(summary).lower()
            
            # より詳細なキーワード分類
            if any(word in text_lower for word in ['coffee', 'tea', 'beverage', 'drink', 'juice', 'soda']):
                return 'beverages'
            elif any(word in text_lower for word in ['snack', 'chip', 'candy', 'cookie', 'cracker', 'nuts']):
                return 'snacks'
            elif any(word in text_lower for word in ['food', 'sauce', 'spice', 'pasta', 'rice', 'meat']):
                return 'food'
            else:
                # デフォルトは最も一般的なカテゴリ
                return 'food'
        
        # レビューテキストとサマリーの両方を使用
        if 'review_text' in df.columns and 'review_title' in df.columns:
            return df.apply(lambda row: categorize_review(row['review_text'], row['review_title']), axis=1)
        elif 'review_text' in df.columns:
            return df['review_text'].apply(lambda x: categorize_review(x))
        else:
            # フォールバック：ランダムだが警告
            print("⚠️ テキストデータなし：カテゴリをランダム推定")
            categories = ['food', 'beverages', 'snacks']
            return pd.Series(np.random.choice(categories, len(df)))
        
    def load_kaggle_dataset(self, dataset_name: str = "amazon_reviews", use_real_data: bool = True) -> pd.DataFrame:
        """
        データセット読み込み（更新版）
        
        Args:
            dataset_name (str): データセット名
            use_real_data (bool): 実データを使用するか
            
        Returns:
            pd.DataFrame: レビューデータ
        """
        if use_real_data:
            try:
                # 実データ取得を試行
                return self.load_real_kaggle_dataset("snap/amazon-fine-food-reviews")
            except Exception as e:
                print(f"実データ取得失敗: {e}")
                print("サンプルデータで代替します...")
                return self._generate_sample_data()
        else:
            # サンプルデータ使用
            print("サンプルデータを生成中...")
            return self._generate_sample_data()

    def _generate_sample_data(self, n_samples: int = 3000) -> pd.DataFrame:
        """
        現実的なサンプルデータ生成
        
        Args:
            n_samples (int): 生成するレビュー数（デフォルト3000件）
            
        Returns:
            pd.DataFrame: 生成されたレビューデータ
        """
        print(f"{n_samples}件のサンプルデータを生成中...")
        
        # 乱数のシード設定
        np.random.seed(42)
        
        # 商品マスターデータの作成
        products = []
        for category, items in self.target_categories.items():
            for item in items:
                products.append({
                    "name": item,        # 商品名
                    "category": category # カテゴリ
                })
        
        # メインのデータ生成ループ
        data = []
        
        # tqdm: プログレスバー表示
        for i in tqdm(range(n_samples), desc="データ生成中"):
            
            # ステップ1: ランダムに商品を選択
            product = np.random.choice(products)
            
            # ステップ2: 現実的な評価分布で評価を生成
            # 実際のAmazonは高評価が多いので、それを反映
            rating = np.random.choice(
                [1, 2, 3, 4, 5],                    # 評価値
                p=[0.05, 0.10, 0.15, 0.35, 0.35]   # 各評価の出現確率
            )
            # 5点が35%、4点が35%、3点が15%、2点が10%、1点が5%
            
            # ステップ3: 評価に応じたレビューテキスト生成
            review_text = self._generate_review_text(rating, product["name"])
            
            # ステップ4: リアルな日付生成
            start_date = datetime.now() - timedelta(days=365)
            random_days = np.random.randint(0, 365)
            review_date = start_date + timedelta(days=random_days)
            
            # ステップ5: データレコード作成
            record = {
                # 商品情報
                "product_id": f"B{i:06d}",                    # 商品ID（B000001形式）
                "product_name": product["name"],              # 商品名
                "product_category": product["category"],      # カテゴリ
                
                # レビュー情報  
                "review_id": f"R{i:08d}",                    # レビューID
                "rating": rating,                            # 評価（1-5）
                "review_text": review_text,                  # レビュー本文
                
                # メタデータ（追加情報）
                "helpful_votes": np.random.randint(0, 50),   # 「役に立った」票数
                "total_votes": np.random.randint(0, 100),    # 総投票数
                "verified_purchase": np.random.choice(       # 購入確認済みか
                    [True, False], p=[0.85, 0.15]           # 85%が確認済み
                ),
                "review_date": review_date                   # レビュー日付
            }
            
            # リストにレコードを追加
            data.append(record)
        
        # リストをDataFrameに変換
        df = pd.DataFrame(data)
        print(f"サンプルデータ生成完了: {len(df)}件")
        return df
    
    def _generate_review_text(self, rating: int, product_name: str) -> str:
        """
        評価に基づくレビューテキスト生成
        
        Args:
            rating (int): 評価（1-5）
            product_name (str): 商品名
            
        Returns:
            str: 生成されたレビューテキスト
        """
        
        if rating >= 4:
            # 高評価（4-5点）の場合
            base_text = np.random.choice(self.positive_patterns)
            
            # 商品カテゴリに応じた具体的なコメント追加
            if "充電器" in product_name:
                specifics = ["充電速度が早いです。", "ケーブルも丈夫そうです。", "コンパクトで持ち運びやすいです。"]
            elif "イヤホン" in product_name:
                specifics = ["音質がクリアです。", "フィット感が良いです。", "長時間使っても疲れません。"]
            elif "本" in product_name:
                specifics = ["内容が分かりやすいです。", "実践的で役立ちます。", "読みやすい構成です。"]
            else:
                specifics = ["機能が充実しています。", "デザインが気に入りました。"]
            
            # 基本文 + 具体的コメント
            return base_text + " " + np.random.choice(specifics)
            
        elif rating <= 2:
            # 低評価（1-2点）の場合
            base_text = np.random.choice(self.negative_patterns)
            
            # 商品カテゴリに応じた具体的な問題点追加
            if "充電器" in product_name:
                specifics = ["充電できませんでした。", "ケーブルがすぐ断線しました。", "発熱が気になります。"]
            elif "イヤホン" in product_name:
                specifics = ["音が途切れます。", "耳にフィットしません。", "音質が悪すぎます。"]
            elif "本" in product_name:
                specifics = ["内容が薄いです。", "期待していた内容と違いました。", "誤字が多いです。"]
            else:
                specifics = ["機能が不十分です。", "すぐに故障しました。"]
            
            return base_text + " " + np.random.choice(specifics)
            
        else:
            # 中評価（3点）の場合
            return np.random.choice(self.neutral_patterns)
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        データの前処理・クリーニング
        
        Args:
            df (pd.DataFrame): 生データ
            
        Returns:
            pd.DataFrame: 前処理済みデータ
        """
        print("データ前処理を開始...")
        df_processed = df.copy()  # コピーを作成し、元データを保護
        
        # ステップ1: 欠損値処理
        print("欠損値処理中...")
        original_count = len(df_processed)
        # 重要な列（レビュー本文、評価）に欠損があるレコードを除外
        df_processed = df_processed.dropna(subset=["review_text", "rating"])
        dropped_count = original_count - len(df_processed)
        if dropped_count > 0:
            print(f"   欠損値のある{dropped_count}件を除外しました")
        
        # ステップ2: テキストの前処理
        print("テキスト前処理中...")
        # 文字列の前後の空白除去
        df_processed["review_text_clean"] = df_processed["review_text"].str.strip()
        # レビューの文字数計算
        df_processed["review_length"] = df_processed["review_text_clean"].str.len()
        
        # ステップ3: 評価の正規化
        # 1-5の評価を0-1の範囲に変換
        df_processed["rating_normalized"] = (df_processed["rating"] - 1) / 4
        
        # ステップ4: 感情ラベルの付与
        print("感情ラベル付与中...")
        df_processed["sentiment_label"] = df_processed["rating"].apply(self._rating_to_sentiment)
        
        # ステップ5: 有用性の計算
        # 「役に立った」票の割合を計算
        df_processed["helpfulness_ratio"] = df_processed.apply(
            lambda x: x["helpful_votes"] / max(x["total_votes"], 1), axis=1
        )
        
        # ステップ6: 日付の処理
        if "review_date" in df_processed.columns:
            # 文字列の日付をdatetime型に変換
            df_processed["review_date"] = pd.to_datetime(df_processed["review_date"])
            # 年・月を別列として抽出（時系列分析で使用）
            df_processed["review_year"] = df_processed["review_date"].dt.year
            df_processed["review_month"] = df_processed["review_date"].dt.month
        
        # ステップ7: カテゴリ別統計の追加
        print("カテゴリ別統計計算中...")
        # カテゴリごとの平均評価等を計算し、各レコードに追加
        category_stats = df_processed.groupby("product_category")["rating"].agg([
            "mean", "std", "count"  # 平均、標準偏差、件数
        ]).add_prefix("category_")
        
        # 元データに統計情報を結合
        df_processed = df_processed.merge(
            category_stats, 
            left_on="product_category", 
            right_index=True
        )
        
        # ステップ8: データ品質フラグ
        # 高品質なレビューかどうかを判定
        df_processed["is_high_quality"] = (
            (df_processed["verified_purchase"] == True) &      # 購入確認済み
            (df_processed["review_length"] >= 20) &            # ある程度長い
            (df_processed["total_votes"] >= 1)                 # 投票がある
        )
        
        print(f"前処理完了: {len(df_processed)}件のデータ")
        return df_processed
    
    def _rating_to_sentiment(self, rating: int) -> str:
        """
        評価を感情ラベルに変換
        
        Args:
            rating (int): 評価（1-5）
            
        Returns:
            str: 感情ラベル（positive/neutral/negative）
        """
        if rating >= 4:
            return "positive"    # 4-5点：ポジティブ
        elif rating == 3:
            return "neutral"     # 3点：ニュートラル
        else:
            return "negative"    # 1-2点：ネガティブ
    
    def save_data(self, df: pd.DataFrame, filename: str = "raw_reviews.csv") -> str:
        """
        データをCSVファイルに保存
        
        Args:
            df (pd.DataFrame): 保存するデータ
            filename (str): ファイル名
            
        Returns:
            str: 保存されたファイルパス
        """
        filepath = os.path.join(self.data_dir, filename)
        
        try:
            # UTF-8エンコードでCSV保存
            df.to_csv(filepath, index=False, encoding="utf-8")
            
            # 保存結果の報告
            print(f"データを保存しました: {filepath}")
            print(f"   件数: {len(df):,}件")
            print(f"   ファイルサイズ: {os.path.getsize(filepath) / 1024:.1f} KB")
            return filepath
            
        except Exception as e:
            print(f"データ保存エラー: {e}")
            raise  # エラーを上位に伝播
    
    def load_data(self, filename: str = "raw_reviews.csv") -> pd.DataFrame:
        """
        保存されたデータを読み込み
        
        Args:
            filename (str): ファイル名
            
        Returns:
            pd.DataFrame: 読み込まれたデータ
        """
        filepath = os.path.join(self.data_dir, filename)
        
        # ファイル存在チェック
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"ファイルが見つかりません: {filepath}")
        
        try:
            # UTF-8エンコードでCSV読み込み
            df = pd.read_csv(filepath, encoding="utf-8")
            print(f"データを読み込みました: {filepath}")
            print(f"   件数: {len(df):,}件")
            return df
            
        except Exception as e:
            print(f"データ読み込みエラー: {e}")
            raise
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """
        データの概要統計を取得
        
        Args:
            df (pd.DataFrame): 分析対象データ
            
        Returns:
            Dict: 概要統計情報（辞書形式）
        """
        
        # 基本情報の計算
        basic_info = {
            "total_reviews": len(df),                           # 総レビュー数
            "unique_products": df["product_id"].nunique(),      # ユニーク商品数
            "unique_categories": df["product_category"].nunique(), # カテゴリ数
            "date_range": {
                "start": str(df["review_date"].min()) if "review_date" in df.columns else "N/A",
                "end": str(df["review_date"].max()) if "review_date" in df.columns else "N/A"
            }
        }
        
        # 評価分析
        rating_analysis = {
            "average_rating": round(df["rating"].mean(), 2),           # 平均評価
            "rating_distribution": df["rating"].value_counts().sort_index().to_dict(), # 評価分布
            "rating_std": round(df["rating"].std(), 2)                 # 評価の標準偏差
        }
        
        # 感情分析
        sentiment_analysis = {
            "sentiment_distribution": df["sentiment_label"].value_counts().to_dict() if "sentiment_label" in df.columns else {},
            "negative_ratio": round((df["sentiment_label"] == "negative").mean() * 100, 1) if "sentiment_label" in df.columns else 0
        }
        
        # カテゴリ分析
        category_analysis = {
            "category_distribution": df["product_category"].value_counts().to_dict(),
            "category_ratings": df.groupby("product_category")["rating"].mean().round(2).to_dict()
        }
        
        # テキスト分析
        text_analysis = {
            "average_review_length": round(df["review_length"].mean(), 1) if "review_length" in df.columns else 0,
            "median_review_length": round(df["review_length"].median(), 1) if "review_length" in df.columns else 0,
            "verified_purchase_ratio": round((df["verified_purchase"] == True).mean() * 100, 1) if "verified_purchase" in df.columns else 0
        }
        
        # 品質指標
        quality_metrics = {
            "high_quality_reviews": (df["is_high_quality"] == True).sum() if "is_high_quality" in df.columns else 0,
            "reviews_with_votes": (df["total_votes"] > 0).sum() if "total_votes" in df.columns else 0,
            "average_helpfulness": round(df["helpfulness_ratio"].mean(), 3) if "helpfulness_ratio" in df.columns else 0
        }
        
        # 全統計情報をまとめて返す
        summary = {
            "basic_info": basic_info,
            "rating_analysis": rating_analysis,
            "sentiment_analysis": sentiment_analysis,
            "category_analysis": category_analysis,
            "text_analysis": text_analysis,
            "quality_metrics": quality_metrics
        }
        
        return summary
    
    def print_summary(self, df: pd.DataFrame):
        """
        データ概要を見やすく表示
        
        Args:
            df (pd.DataFrame): 分析対象データ
        """
        summary = self.get_data_summary(df)
        
        print("\n" + "="*60)
        print("AMAZON レビューデータ 概要統計")
        print("="*60)
        
        # 基本情報の表示
        print(f"\n📋 基本情報:")
        basic = summary["basic_info"]
        print(f"   総レビュー数: {basic['total_reviews']:,} 件")
        print(f"   ユニーク商品数: {basic['unique_products']:,} 件")
        print(f"   カテゴリ数: {basic['unique_categories']} 種類")
        print(f"   期間: {basic['date_range']['start']} ～ {basic['date_range']['end']}")
        
        # 評価分析の表示
        print(f"\n評価分析:")
        rating = summary["rating_analysis"]
        print(f"   平均評価: {rating['average_rating']}/5.0")
        print(f"   評価分布: {rating['rating_distribution']}")
        
        # 感情分析の表示
        print(f"\n感情分析:")
        sentiment = summary["sentiment_analysis"]
        print(f"   感情分布: {sentiment['sentiment_distribution']}")
        print(f"   ネガティブ率: {sentiment['negative_ratio']}%")
        
        # カテゴリ分析の表示
        print(f"\nカテゴリ分析:")
        category = summary["category_analysis"]
        print(f"   カテゴリ分布: {category['category_distribution']}")
        print(f"   カテゴリ別評価: {category['category_ratings']}")
        
        # テキスト分析の表示
        print(f"\nテキスト分析:")
        text = summary["text_analysis"]
        print(f"   平均レビュー長: {text['average_review_length']} 文字")
        print(f"   購入確認済み率: {text['verified_purchase_ratio']}%")
        
        # 品質指標の表示
        print(f"\n品質指標:")
        quality = summary["quality_metrics"]
        print(f"   高品質レビュー: {quality['high_quality_reviews']} 件")
        print(f"   投票付きレビュー: {quality['reviews_with_votes']} 件")
        print(f"   平均有用性: {quality['average_helpfulness']}")
        
        print("="*60)


def main():
    """
    メイン実行関数 - データ収集から前処理まで一括実行
    
    1. データ収集器の初期化
    2. サンプルデータ生成
    3. データ前処理・クリーニング
    4. CSV保存
    5. 統計分析・品質チェック
    """
    print("Amazon レビュー分析プロジェクト - データ収集開始")
    print("="*60)
    
    try:
        # ステップ1: データ収集器の初期化
        print("データ収集器を初期化中...")
        collector = AmazonReviewCollector()
        print("   初期化完了（データフォルダ、カテゴリ設定済み）")
        
        # ステップ2: データセットの読み込み（サンプル生成）
        print("\nデータセットの読み込み中...")
        df_raw = collector.load_kaggle_dataset()
        print(f"   生データ取得完了: {len(df_raw)}件")
        
        # ステップ3: データの前処理
        print("\nデータの前処理中...")
        df_processed = collector.preprocess_data(df_raw)
        print(f"   前処理完了: 感情ラベル、統計量等を追加")
        
        # ステップ4: データの保存
        print("\nデータの保存中...")
        collector.save_data(df_raw, "raw_reviews.csv")              # 生データ保存
        collector.save_data(df_processed, "processed_data.csv")     # 前処理済みデータ保存
        print("   両方のデータファイルを保存完了")
        
        # ステップ5: データ概要の表示
        print("\nデータ概要分析:")
        collector.print_summary(df_processed)
        
        # ステップ6: 品質チェック
        print("\nデータ品質チェック:")
        quality_issues = []
        
        # 欠損値チェック
        missing_values = df_processed.isnull().sum()
        if missing_values.any():
            quality_issues.append(f"欠損値: {missing_values[missing_values > 0].to_dict()}")
        
        # レビュー長チェック（短すぎるレビューの検出）
        short_reviews = (df_processed["review_length"] < 10).sum()
        if short_reviews > 0:
            quality_issues.append(f"短すぎるレビュー: {short_reviews}件")
        
        # 感情ラベル分布チェック（極端な偏りの検出）
        sentiment_dist = df_processed["sentiment_label"].value_counts()
        if sentiment_dist.min() < len(df_processed) * 0.05:  # 5%未満のクラス
            quality_issues.append("感情ラベルの偏りを検出")
        
        # 品質チェック結果の表示
        if quality_issues:
            print("   品質課題:")
            for issue in quality_issues:
                print(f"      - {issue}")
        else:
            print("   データ品質良好")
        
        # ステップ7: 次のステップの案内
        print("\nPhase 1 完了! 次のステップ:")
        print("   1. 感情分析モデルの実装（Phase 2）")
        print("   2. キーワード抽出機能の追加")
        print("   3. 改善提案エンジンの開発")
        print("   4. Streamlit Webアプリの作成")
        
        # ステップ8: 成果物の確認
        print(f"\n生成されたファイル:")
        for filename in ["raw_reviews.csv", "processed_data.csv"]:
            filepath = os.path.join(collector.data_dir, filename)
            if os.path.exists(filepath):
                size_kb = os.path.getsize(filepath) / 1024
                print(f"   {filename}: {size_kb:.1f} KB")
            else:
                print(f"   {filename}: 生成失敗")
        
        print("\nデータ収集・前処理が正常に完了しました!")
        print("="*60)
        
        return df_processed
        
    except Exception as e:
        # エラーが発生した場合の詳細情報表示
        print(f"\nエラーが発生しました: {e}")
        print("詳細なエラー情報:")
        import traceback
        traceback.print_exc()
        return None


def demo_analysis(df: pd.DataFrame):
    """
    デモ用の簡単な分析を実行
    
    Args:
        df (pd.DataFrame): 分析対象データ
    """
    print("\nデモ分析を実行中...")
    
    # ネガティブレビューの分析（改善提案の原点）
    negative_reviews = df[df["sentiment_label"] == "negative"]
    print(f"\nネガティブレビュー分析:")
    print(f"   件数: {len(negative_reviews)}件 ({len(negative_reviews)/len(df)*100:.1f}%)")
    
    if len(negative_reviews) > 0:
        # カテゴリ別ネガティブ率の計算
        negative_by_category = df.groupby("product_category").apply(
            lambda x: (x["sentiment_label"] == "negative").sum() / len(x) * 100
        ).round(1)
        
        print("   カテゴリ別ネガティブ率:")
        for category, rate in negative_by_category.items():
            print(f"     {category}: {rate}%")
        
        # 最もネガティブなレビューのサンプル表示
        worst_reviews = negative_reviews.nsmallest(3, "rating")
        print("\n   代表的なネガティブレビュー:")
        for idx, (_, review) in enumerate(worst_reviews.iterrows(), 1):
            print(f"     {idx}. [{review['product_name']}] 評価{review['rating']}")
            print(f"        「{review['review_text'][:50]}...」")
    
    # カテゴリ別詳細統計
    print(f"\nカテゴリ別詳細統計:")
    category_stats = df.groupby("product_category").agg({
        "rating": ["count", "mean", "std"],        # 件数、平均、標準偏差
        "review_length": "mean",                   # 平均レビュー長
        "helpfulness_ratio": "mean"                # 平均有用性
    }).round(2)
    
    print(category_stats)
    
    # 高品質レビューの分析
    if "is_high_quality" in df.columns:
        high_quality = df[df["is_high_quality"] == True]
        print(f"\n高品質レビュー分析:")
        print(f"   件数: {len(high_quality)}件 ({len(high_quality)/len(df)*100:.1f}%)")
        print(f"   平均評価: {high_quality['rating'].mean():.2f}")
        
        # 高品質レビューと通常レビューの比較
        normal_quality = df[df["is_high_quality"] == False]
        print(f"   高品質 vs 通常の平均評価: {high_quality['rating'].mean():.2f} vs {normal_quality['rating'].mean():.2f}")


def test_kaggle_integration():
    """Kaggle API 統合テスト"""
    collector = AmazonReviewCollector()
    
    print("Kaggle API統合テスト")
    
    # 実データ取得テスト
    df_real = collector.load_kaggle_dataset(use_real_data=True)
    print(f"実データ取得結果: {len(df_real)}件")
    
    # サンプルデータとの比較
    df_sample = collector.load_kaggle_dataset(use_real_data=False)
    print(f"サンプルデータ: {len(df_sample)}件")
    
    return df_real, df_sample


def quick_test():
    """
    クイックテスト - 少量データで動作確認
    
    """
    print("クイックテスト実行中...")
    
    # 少量データでテスト
    collector = AmazonReviewCollector()
    
    # 100件のテストデータ生成
    df_test = collector._generate_sample_data(n_samples=100)
    print(f"テストデータ生成: {len(df_test)}件")
    
    # 前処理テスト
    df_processed = collector.preprocess_data(df_test)
    print(f"前処理完了: {len(df_processed)}件")
    
    # 基本統計の計算・表示
    summary = collector.get_data_summary(df_processed)
    print(f"統計計算完了")
    print(f"   平均評価: {summary['rating_analysis']['average_rating']}")
    print(f"   感情分布: {summary['sentiment_analysis']['sentiment_distribution']}")
    
    print("クイックテスト完了!")
    return df_processed


def analyze_sample_reviews(df: pd.DataFrame, n_samples: int = 5):
    """
    生成されたレビューのサンプルを表示・確認
    
    Args:
        df (pd.DataFrame): レビューデータ
        n_samples (int): 表示するサンプル数
    """
    print(f"\nレビューサンプル確認 (ランダム {n_samples}件):")
    print("-" * 80)
    
    # ランダムにサンプルを選択
    sample_reviews = df.sample(n_samples)
    
    for idx, (_, row) in enumerate(sample_reviews.iterrows(), 1):
        print(f"{idx}. 商品: {row['product_name']} | カテゴリ: {row['product_category']}")
        print(f"   評価: {row['rating']}/5 | 感情: {row['sentiment_label']}")
        print(f"   レビュー: {row['review_text']}")
        print(f"   メタ: 購入確認={row['verified_purchase']}, 長さ={row['review_length']}文字")
        print("-" * 80)


if __name__ == "__main__":
    """
    このファイルが直接実行された場合のメイン処理
    
    実行方法：
    1. 通常実行: python src/data_collector.py
    2. クイックテスト: python -c "from src.data_collector import quick_test; quick_test()"
    """
    
    print("Amazon レビュー分析プロジェクト - Phase 1")
    print("目的: レビューデータ収集・前処理システムの構築")
    print("="*60)
    
    # メイン処理実行
    result_df = main()
    
    # 成功した場合はデモ分析も実行
    if result_df is not None:
        print("\n追加のデモ分析を実行中...")
        demo_analysis(result_df)
        
        # サンプルレビューの確認
        analyze_sample_reviews(result_df, n_samples=3)
    
