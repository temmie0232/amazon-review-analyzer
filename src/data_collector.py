import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional
from tqdm import tqdm
import json
from datetime import datetime, timedelta


class AmazonReviewCollector:
    """
    Amazon レビューデータを収集・前処理するクラス
    """
    def __init__(self,data_dir: str = "data"): # 引数がなければ"data"を使う
        """
        クラスの初期化処理
        
        Args:
            data_dir (str): データを保存するディレクトリのパス
        """
        self.data_dir = data_dir

        # データディレクトリの存在確認・作成
        self.ensure_data_dir()

        # 商品カテゴリの定義
        # 各カテゴリに属する商品キーワードを定義
        self.target_categories = {
            "electronics": ["充電器", "イヤホン", "ケーブル", "バッテリー"],
            "books": ["プログラミング本", "ビジネス書", "自己啓発本"],
            "daily_goods": ["キッチン用品", "掃除用具", "収納ボックス"]
        }
        
        # レビューパターンの定義
        # 好評(4-5)
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
        
        # 悪評(1-2)
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
        
        # 普通(3)
        self.neutral_patterns = [
            "普通の商品だと思います。可もなく不可もなく。",
            "値段相応の品質です。特に問題はありません。",
            "期待していた通りの商品でした。",
            "使えますが、特別良いというわけではありません。",
            "標準的な商品です。必要最低限の機能はあります。"
        ]        
        
    def ensure_data_dir(self):
        """データディレクトリの存在確認・作成"""
        # 存在しなかったら
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"データディレクトリを作成しました: {self.data_dir}")
            
    def load_kaggle_dataset(self,dataset_name: str = "amazon_reviews") -> pd.DataFrame:
        """
        Kaggleデータセットからレビューデータを読み込む
        
        Args:
            dataset_name (str): データセット名 (後で実装)
            
        Returns:
            pd.DataFrame: レビューデータ
        """
        try:
            # ここでKaggle APIを使用してデータセットをダウンロード(後で実装)
            # 現在はサンプルデータを生成
            sample_data = self._generate_sample_data()
            return sample_data
        except Exception as e:
            print(f"データセット読み込みエラー:{e}")
            return self._generate_sample_data()
        
    def _generate_sample_data(self, n_samples: int = 3000) -> pd.DataFrame:
        """
        デモ用サンプルデータの生成
        
        Args:
            n_samples (int): 生成するサンプル数 (デフォルト:3000)
        
        Returns:
            pd.DataFrame: 生成されたサンプルレビューデータ

        サンプルデータの構成:
        - 商品情報 (ID, 名前, カテゴリ)
        - レビュー情報 (ID, 評価, テキスト, 登校日)
        - メタ情報 (有用性投票, 購入確認等)
        """
        
        # シード設定
        np.random.seed(42)

        # サンプルの商品リスト
        products = [
            "iPhone充電器", "ワイヤレスイヤホン", "USB-Cケーブル", "モバイルバッテリー",
            "Pythonプログラミング本", "ビジネス書", "自己啓発本",
            "キッチン用品", "掃除用具", "収納ボックス"
        ]
        
        # ポジティブレビューのサンプル
        positive_reviews = [
            "とても良い商品です。品質が高く、長く使えそうです。",
            "期待以上の性能でした。買って良かったです。",
            "デザインが素晴らしく、使いやすいです。",
            "コストパフォーマンスが最高です。おすすめします。",
            "配送も早く、梱包も丁寧でした。また利用したいです。"
        ]
        
        # ネガティブレビューのサンプル
        negative_reviews = [
            "品質が期待より低く、すぐに壊れました。",
            "説明と実際の商品が違いました。残念です。",
            "価格に見合わない品質だと思います。",
            "ケーブルが短すぎて使いにくいです。",
            "充電速度が遅く、期待していたより時間がかかります。",
            "音質が悪く、ノイズが気になります。",
            "本の内容が薄く、値段が高すぎると思います。"
        ]
    
        data = []
        
        # 指定されたサンプル数分のデータを生成
        for i in range(n_samples):
            # ランダムに商品
            product = np.random.choice(products)

            # 実際のAmazonの評価分布に近い確率で星評価を生成
            # 一般的に高評価（4-5星）が多い傾向を反映
            rating = np.random.choice([1,2,3,4,5], p=[0.1,0.1,0.2,0.3,0.3])

            # 評価に基づいて適切なレビューテキストを選択
            if rating >= 4:
                review_text = np.random.choice(positive_reviews)
            else:
                review_text = np.random.choice(negative_reviews)

            data.append({
                "product_id": f"B{i:06d}",           # 商品ID（Amazonの形式に似せる）
                "product_name": product,              # 商品名
                "product_category": self._get_category(product),  # カテゴリ（自動判別）
                "review_id": f"R{i:08d}",            # レビューID
                "rating": rating,                     # 星評価（1-5）
                "review_text": review_text,           # レビューテキスト
                "helpful_votes": np.random.randint(0, 50),      # 「役に立った」投票数
                "total_votes": np.random.randint(0, 100),       # 総投票数
                "verified_purchase": np.random.choice([True, False], p=[0.8, 0.2]),  # 購入確認済み
                "review_date": pd.date_range("2023-01-01", "2024-12-31", periods=n_samples)[i]  # 投稿日
            })

        return pd.DataFrame(data)

    def _get_category(self,product_name: str) -> str:
        """
        商品名からカテゴリを判別
        
        Args:
            product_name (str): 商品名
            
        Returns: 
            str: カテゴリ名 ("electronics", "books", "daily_goods"のどれか)
        """
        product_lower = product_name.lower()
        
        # 電子機器関連のキーワードチェック
        if any(keyword in product_lower for keyword in ["充電", "ケーブル", "イヤホン", "バッテリー"]):
            return "electronics"
        # 書籍関連のキーワードチェック
        elif "本" in product_lower:
            return "books"
        # その他は日用品として分類
        else:
            return "daily_goods"
        
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        データの前処理
        
        Args:
            df (pd.DataFrame): 生データ
            
        Returns:
            pd.DataFrame: 前処理済みデータ
            
        処理内容:
        1. 欠損値の除去
        2. テキストのクリーニング
        3. 特徴量の生成 (長さ, 正規化評価, 感情ラベルなど)
        4. 有用性指標の計算
        """
        
        # 1. 必須フィールドの欠損値を持つ行を除去
        original_count = len(df)
        df = df.dropna(subset=["review_text", "rating"])
        removed_count = original_count - len(df)
        if removed_count > 0:
            print(f"  欠損値を含む行を {removed_count} 件除去しました  ")

        # 2. テキストの前処理
        # 前後の空白を除去
        df["review_text_clean"] = df["review_text"].str.strip()

        # レビューの文字数を計算 (テキスト解析の特徴量として使用する)
        df["review_length"] = df["review_text_clean"].str.len()
        
        # 3.評価の正規化 (1-5 -> 0-1)
        df["rating_normalized"] = (df["rating"] - 1) / 4  # 0-1に正規化
        
        # 4.感情ラベルの追加
        # 星評価を三段階の感情カテゴリに変換
        df["sentiment_label"] = df["rating"].apply(self._rating_to_sentiment)
        
        # 有用性比率の計算
        # 役に立った投票の割合を算出 (品質指標として使用)
        df["helpfulness_ratio"] = df.apply(
            lambda x: x["helpful_votes"] / max(x["total_votes"], 1), axis=1 # ゼロ除算を防ぐ
        )
        
        print(f"  前処理完了: {len(df)} 件のレビューを処理しました")
        return df
    
    def _rating_to_sentiment(self, rating: int) -> str:
        """
        星評価を感情ラベルに変換
        
        Args:
            rating (int): 星評価 (1-5)

        Returns:
            str: 感情ラベル ("positive", "neutral", "negative")
        """
        
        if rating >= 4:
            return "positive"   # 星4-5: ポジティブ
        elif rating == 3:
            return "neutral"    # 星3: ニュートラル
        else:
            return "negative"   # 星1-2: ネガティブ
    
    def save_data(self, df: pd.DataFrame, filename: str = "raw_reviews.csv"):
        """
        データをCSVファイルとして保存
        
        Args:
            df (pd.DataFrame): 保存するデータ
            filename (str): ファイル名

        Returns:
            str: 保存されたファイルのパス
        """
        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath, index=False, encoding="utf-8")
        print(f"データを保存しました: {filepath}")
        return filepath
    
    def load_data(self, filename: str = "raw_reviews.csv") -> pd.DataFrame:
        """
        保存されたデータを読み込み
        
        Args:
            filename (str): 読み込むファイル名
            
        Returns:
            pd.DataFrame: 読み込まれたデータ
            
        Raises:
            FileNotFoundError: ファイルが存在しない場合
        """
        filepath = os.path.join(self.data_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, encoding="utf-8")
            print(f"データを読み込みました: {filepath} ({len(df)} 件)")
            return df
        else:
            raise FileNotFoundError(f"ファイルが見つかりません: {filepath}")
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """
        データの概要統計を取得
        
        Args:
            df (pd.DataFrame): 分析対象のデータ

        Returns:
            Dict: 統計情報を含む辞書
        """
        summary = {
            # 基本統計
            "total_reviews": len(df),
            "unique_products": df["product_id"].nunique(),
            
            # 分布統計
            "rating_distribution": df["rating"].value_counts().to_dict(),
            "category_distribution": df["product_category"].value_counts().to_dict(),
            "sentiment_distribution": df["sentiment_label"].value_counts().to_dict(),
            
            # テキスト統計
            "avg_review_length": df["review_length"].mean(),
            
            # 時系列統計
            "date_range": {
                "start": df["review_date"].min(),
                "end": df["review_date"].max()
            }
        }
        return summary

def main():
    """
    メイン実行関数
    
    Returns:
        pd.DataFrame: 処理済みのレビューデータ
        
    実行手順:
    1. データ収収集器の初期化
    2. サンプルデータの生成
    3. データの前処理
    4. データの保存
    5. 統計情報の表示
    """
    print("=== Amazon レビューデータの収集開始 ===")

    # 1. データ収集器の初期化
    collector = AmazonReviewCollector()

    # 2. データセットの読み込み (現在はサンプルデータの生成)
    print("1. データセットの読み込み中...")
    df = collector.load_kaggle_dataset()
    print(f"  {len(df)} 件のレビューデータを取得しました")
    
    # 3. データ前処理
    print("2. データの前処理中...")
    df_processed = collector.preprocess_data(df)
    
    # 4. データ保存
    print("3. データの保存中...")
    collector.save_data(df_processed, "raw_reviews.csv")    # 生データ
    collector.save_data(df_processed, "processed_data.csv") # 処理済みデータ
    
    # データ概要の表示
    print("4. データ概要:")
    summary = collector.get_data_summary(df_processed)

    # 統計情報を表示
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for sub_key, sub_value in value.items():
                print(f"     {sub_key}: {sub_value}")
        else:
            print(f"   {key}: {value}")

    print("\n=== データ収集完了 ===")
    
    return df_processed

if __name__ == "__main__":
    df = main()