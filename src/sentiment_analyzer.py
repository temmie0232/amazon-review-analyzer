"""
Amazon レビュー分析プロジェクト - 感情分析エンジン (Phase 2)

このファイルの目的：
1. HuggingFace BERTを使った高精度感情分析（英語対応）
2. 568,454件の実データに対応した大規模バッチ処理
3. TF-IDFベースのキーワード抽出
4. 問題点の自動分類
5. 感情分析精度の評価・検証

実データ仕様：
- Amazon Fine Food Reviews (568k件)
- 英語レビューのみ
- 評価: 1-5点
- 期間: 1999-2012年
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.pipelines import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import gc
import os
from typing import Dict, List, Tuple, Optional
import re
from collections import defaultdict, Counter

# 警告を抑制
warnings.filterwarnings('ignore')


class SentimentAnalyzer:
    """
    BERT活用の感情分析エンジン
    
    特徴:
    - 大規模データ対応（568k件）
    - バッチ処理による効率化
    - メモリ最適化
    - キーワード抽出機能
    - 問題分類システム
    """
    
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest", 
                 batch_size: int = 32, device: str = "auto"):
        """
        感情分析器の初期化
        
        Args:
            model_name (str): 使用するBERTモデル名
            batch_size (int): バッチサイズ（メモリに応じて調整）
            device (str): 計算デバイス ("auto", "cpu", "cuda")
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = self._setup_device(device)
        
        # モデル・トークナイザーの初期化（遅延読み込み）
        self.tokenizer = None
        self.model = None
        self.sentiment_pipeline = None
        
        # キーワード抽出用
        self.tfidf_vectorizer = None
        self.problem_keywords = None
        
        # 結果保存用
        self.results = {}
        
        print(f"SentimentAnalyzer初期化完了")
        print(f"   モデル: {self.model_name}")
        print(f"   バッチサイズ: {self.batch_size}")
        print(f"   デバイス: {self.device}")
    
    def _setup_device(self, device: str) -> str:
        """
        計算デバイスの設定
        
        Args:
            device (str): デバイス指定
            
        Returns:
            str: 実際に使用するデバイス
        """
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                print(f"GPU利用可能: {torch.cuda.get_device_name(0)}")
            else:
                device = "cpu"
                print("CPUを使用")
        elif device == "cuda" and not torch.cuda.is_available():
            print("⚠️ CUDA指定されましたが利用できません。CPUに変更します。")
            device = "cpu"
        
        return device
    
    def load_bert_model(self):
        """
        BERTモデルの読み込み（英語感情分析用）
        
        大規模データ処理に最適化されたpipelineを構築
        """
        print(f"BERTモデル読み込み中: {self.model_name}")
        
        try:
            # トークナイザー読み込み
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # モデル読み込み
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            # パイプライン作成（バッチ処理対応）
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                batch_size=self.batch_size,
                truncation=True,
                max_length=512  # BERT最大長
            )
            
            print("✅ BERTモデル読み込み完了")
            
            # メモリ使用量の表示
            if self.device == "cuda":
                print(f"   GPUメモリ使用量: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
            
            return True
            
        except Exception as e:
            print(f"❌ BERTモデル読み込みエラー: {e}")
            return False
    
    def analyze_sentiment_batch(self, df: pd.DataFrame, text_column: str = "review_text", 
                               chunk_size: int = 1000) -> pd.DataFrame:
        """
        大規模データの感情分析（バッチ処理）
        
        Args:
            df (pd.DataFrame): 分析対象データ
            text_column (str): テキスト列名
            chunk_size (int): チャンク処理サイズ
            
        Returns:
            pd.DataFrame: 感情分析結果付きデータ
        """
        if self.sentiment_pipeline is None:
            print("BERTモデルが読み込まれていません。load_bert_model()を先に実行してください。")
            return df
        
        print(f"感情分析開始: {len(df):,}件のレビュー")
        print(f"   チャンクサイズ: {chunk_size}件")
        print(f"   推定処理時間: {len(df) / (chunk_size * 2):.1f}分")
        
        # 結果保存用リスト
        bert_predictions = []
        bert_scores = []
        
        # テキストデータの前処理
        texts = df[text_column].fillna("").astype(str).tolist()
        
        # チャンク処理
        total_chunks = (len(texts) + chunk_size - 1) // chunk_size
        
        with tqdm(total=total_chunks, desc="BERT感情分析") as pbar:
            for i in range(0, len(texts), chunk_size):
                chunk_texts = texts[i:i + chunk_size]
                
                try:
                    # BERT推論実行
                    chunk_results = self.sentiment_pipeline(chunk_texts)
                    
                    # 結果の処理
                    for result in chunk_results: # type: ignore
                        # ラベルの正規化（POSITIVE/NEGATIVE → positive/negative）
                        label = result['label'].lower() # type: ignore
                        if 'pos' in label:
                            label = 'positive'
                        elif 'neg' in label:
                            label = 'negative'
                        else:
                            label = 'neutral'
                        
                        bert_predictions.append(label)
                        bert_scores.append(result['score']) # type: ignore
                    
                    # メモリクリーンアップ
                    if i % (chunk_size * 10) == 0:  # 10チャンクごと
                        gc.collect()
                        if self.device == "cuda":
                            torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"⚠️ チャンク{i//chunk_size + 1}でエラー: {e}")
                    # エラー時はデフォルト値を設定
                    for _ in chunk_texts:
                        bert_predictions.append('neutral')
                        bert_scores.append(0.5)
                
                pbar.update(1)
        
        # 結果をDataFrameに追加
        df_result = df.copy()
        df_result['bert_sentiment'] = bert_predictions
        df_result['bert_confidence'] = bert_scores
        
        # 結果の統計
        bert_dist = pd.Series(bert_predictions).value_counts()
        print(f"\n✅ BERT感情分析完了")
        print(f"   処理件数: {len(bert_predictions):,}件")
        print(f"   感情分布: {bert_dist.to_dict()}")
        print(f"   平均信頼度: {np.mean(bert_scores):.3f}")
        
        return df_result
    
    def extract_keywords(self, df: pd.DataFrame, text_column: str = "review_text", 
                        sentiment_column: str = "bert_sentiment", 
                        max_features: int = 1000) -> Dict[str, List[Tuple[str, float]]]:
        """
        TF-IDF + 感情別キーワード抽出
        
        Args:
            df (pd.DataFrame): 分析対象データ
            text_column (str): テキスト列名
            sentiment_column (str): 感情列名
            max_features (int): 最大特徴量数
            
        Returns:
            Dict[str, List[Tuple[str, float]]]: 感情別キーワードリスト
        """
        print(f"キーワード抽出開始（TF-IDF）")
        
        # テキスト前処理
        def preprocess_text(text):
            if pd.isna(text):
                return ""
            # 英語テキスト用の前処理
            text = re.sub(r'[^a-zA-Z\s]', '', str(text))  # 英数字のみ
            text = text.lower()  # 小文字変換
            return text
        
        df['processed_text'] = df[text_column].apply(preprocess_text)
        
        # 感情別キーワード抽出
        sentiment_keywords = {}
        
        for sentiment in ['positive', 'negative', 'neutral']:
            sentiment_texts = df[df[sentiment_column] == sentiment]['processed_text'].tolist()
            
            if len(sentiment_texts) == 0:
                print(f"⚠️ {sentiment}感情のレビューが見つかりません")
                sentiment_keywords[sentiment] = []
                continue
            
            print(f"   {sentiment}感情: {len(sentiment_texts):,}件のレビュー")
            
            # TF-IDF計算
            vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                ngram_range=(1, 2),  # 1-2グラム
                min_df=5,  # 最低5回出現
                max_df=0.7  # 70%以下のドキュメントに出現
            )
            
            try:
                tfidf_matrix = vectorizer.fit_transform(sentiment_texts)
                feature_names = vectorizer.get_feature_names_out()
                
                # 重要度の計算（平均TF-IDFスコア）
                importance_scores = tfidf_matrix.mean(axis=0).A1 # type: ignore
                
                # キーワードと重要度のペア作成
                keyword_importance = list(zip(feature_names, importance_scores))
                keyword_importance.sort(key=lambda x: x[1], reverse=True)
                
                # 上位キーワードを保存
                sentiment_keywords[sentiment] = keyword_importance[:50]
                
                print(f"     上位5キーワード: {[kw for kw, _ in keyword_importance[:5]]}")
                
            except Exception as e:
                print(f"⚠️ {sentiment}感情のTF-IDF計算エラー: {e}")
                sentiment_keywords[sentiment] = []
        
        # クラス変数に保存
        self.sentiment_keywords = sentiment_keywords
        
        print(f"✅ キーワード抽出完了")
        return sentiment_keywords
    
    def classify_problems(self, df: pd.DataFrame, 
                         negative_keywords: List[Tuple[str, float]]) -> pd.DataFrame:
        """
        ネガティブレビューの問題分類
        
        Args:
            df (pd.DataFrame): 分析対象データ
            negative_keywords (List[Tuple[str, float]]): ネガティブキーワードリスト
            
        Returns:
            pd.DataFrame: 問題分類結果付きデータ
        """
        print("問題分類開始...")
        
        # 問題カテゴリの定義
        problem_categories = {
            'quality': ['bad', 'terrible', 'awful', 'poor', 'worst', 'horrible', 'disgusting'],
            'price': ['expensive', 'overpriced', 'costly', 'price', 'money', 'cheap'],
            'shipping': ['shipping', 'delivery', 'arrived', 'package', 'packaging'],
            'taste': ['taste', 'flavor', 'bland', 'bitter', 'sweet', 'salty'],
            'service': ['service', 'customer', 'support', 'help', 'response']
        }
        
        # 各レビューの問題分類
        def classify_review_problems(text):
            if pd.isna(text):
                return 'other'
            
            text_lower = str(text).lower()
            category_scores = {}
            
            for category, keywords in problem_categories.items():
                score = sum(1 for keyword in keywords if keyword in text_lower)
                category_scores[category] = score
            
            # 最高スコアのカテゴリを返す
            if max(category_scores.values()) > 0:
                return max(category_scores, key=category_scores.get) # type: ignore
            else:
                return 'other'
        
        # ネガティブレビューのみ分類
        negative_reviews = df[df['bert_sentiment'] == 'negative'].copy()
        
        if len(negative_reviews) > 0:
            negative_reviews['problem_category'] = negative_reviews['review_text'].apply(classify_review_problems)
            
            # 問題分布の計算
            problem_dist = negative_reviews['problem_category'].value_counts()
            
            print(f"✅ 問題分類完了")
            print(f"   ネガティブレビュー: {len(negative_reviews):,}件")
            print(f"   問題分布: {problem_dist.to_dict()}")
            
            # 結果を元のDataFrameにマージ
            df_result = df.copy()
            df_result['problem_category'] = 'none'  # デフォルト値
            
            # ネガティブレビューの分類結果をマージ
            for idx in negative_reviews.index:
                df_result.loc[idx, 'problem_category'] = negative_reviews.loc[idx, 'problem_category']
            
            return df_result
        else:
            print("⚠️ ネガティブレビューが見つかりません")
            df['problem_category'] = 'none'
            return df
    
    def validate_predictions(self, df: pd.DataFrame) -> Dict:
        """
        BERT予測精度の評価
        
        Args:
            df (pd.DataFrame): 予測結果付きデータ
            
        Returns:
            Dict: 評価結果
        """
        print("予測精度評価開始...")
        
        # 元の感情ラベル（rating-based）とBERT予測の比較
        if 'sentiment_label' not in df.columns:
            print("⚠️ 元の感情ラベルが見つかりません")
            return {}
        
        # 有効なデータのみ抽出
        valid_data = df[df['bert_sentiment'].notna() & df['sentiment_label'].notna()].copy()
        
        if len(valid_data) == 0:
            print("⚠️ 比較可能なデータがありません")
            return {}
        
        y_true = valid_data['sentiment_label']
        y_pred = valid_data['bert_sentiment']
        
        # 精度指標の計算
        accuracy = accuracy_score(y_true, y_pred)
        
        # 詳細レポート
        class_report = classification_report(y_true, y_pred, output_dict=True)
        
        # 混同行列
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        # 結果の整理
        validation_results = {
            'accuracy': round(accuracy, 4),
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'total_samples': len(valid_data),
            'label_distribution_true': y_true.value_counts().to_dict(),
            'label_distribution_pred': y_pred.value_counts().to_dict()
        }
        
        print(f"✅ 精度評価完了")
        print(f"   全体精度: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"   比較件数: {len(valid_data):,}件")
        
        # クラス別精度の表示
        for label, metrics in class_report.items(): # type: ignore
            if isinstance(metrics, dict) and label != 'accuracy':
                precision = metrics.get('precision', 0)
                recall = metrics.get('recall', 0)
                f1 = metrics.get('f1-score', 0)
                print(f"   {label}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
        
        self.validation_results = validation_results
        return validation_results
    
    def visualize_results(self, df: pd.DataFrame, save_plots: bool = True):
        """
        分析結果の可視化
        
        Args:
            df (pd.DataFrame): 分析結果データ
            save_plots (bool): グラフ保存フラグ
        """
        print("結果可視化中...")
        
        # 図のスタイル設定
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('BERT感情分析結果', fontsize=16, fontweight='bold')
        
        # 1. 感情分布比較（BERT vs Rule-based）
        if 'sentiment_label' in df.columns:
            ax1 = axes[0, 0]
            
            sentiment_comparison = pd.DataFrame({
                'Rule-based': df['sentiment_label'].value_counts(),
                'BERT': df['bert_sentiment'].value_counts()
            }).fillna(0)
            
            sentiment_comparison.plot(kind='bar', ax=ax1)
            ax1.set_title('感情分布比較: Rule-based vs BERT')
            ax1.set_xlabel('感情ラベル')
            ax1.set_ylabel('件数')
            ax1.legend()
            ax1.tick_params(axis='x', rotation=45)
        
        # 2. 信頼度分布
        ax2 = axes[0, 1]
        df['bert_confidence'].hist(bins=50, ax=ax2, alpha=0.7, edgecolor='black')
        ax2.set_title('BERT予測信頼度分布')
        ax2.set_xlabel('信頼度')
        ax2.set_ylabel('件数')
        ax2.axvline(df['bert_confidence'].mean(), color='red', linestyle='--', 
                   label=f'平均: {df["bert_confidence"].mean():.3f}')
        ax2.legend()
        
        # 3. 問題カテゴリ分布
        if 'problem_category' in df.columns:
            ax3 = axes[1, 0]
            problem_dist = df[df['problem_category'] != 'none']['problem_category'].value_counts()
            
            if len(problem_dist) > 0:
                problem_dist.plot(kind='pie', ax=ax3, autopct='%1.1f%%')
                ax3.set_title('ネガティブレビュー問題分布')
            else:
                ax3.text(0.5, 0.5, 'ネガティブレビューなし', ha='center', va='center')
                ax3.set_title('ネガティブレビュー問題分布')
        
        # 4. 評価別感情分布
        ax4 = axes[1, 1]
        if 'rating' in df.columns:
            rating_sentiment = pd.crosstab(df['rating'], df['bert_sentiment'], normalize='index') * 100
            rating_sentiment.plot(kind='bar', stacked=True, ax=ax4)
            ax4.set_title('評価別BERT感情分布 (%)')
            ax4.set_xlabel('評価 (1-5)')
            ax4.set_ylabel('割合 (%)')
            ax4.legend(title='感情')
            ax4.tick_params(axis='x', rotation=0)
        
        plt.tight_layout()
        
        if save_plots:
            os.makedirs('results', exist_ok=True)
            plt.savefig('results/sentiment_analysis_results.png', dpi=300, bbox_inches='tight')
            print("📊 グラフを保存: results/sentiment_analysis_results.png")
        
        plt.show()
    
    def save_results(self, df: pd.DataFrame, filename: str = "sentiment_analysis_results.csv"):
        """
        分析結果の保存
        
        Args:
            df (pd.DataFrame): 結果データ
            filename (str): 保存ファイル名
        """
        os.makedirs('results', exist_ok=True)
        filepath = os.path.join('results', filename)
        
        # 重要な列のみ選択
        columns_to_save = [
            'product_id', 'rating', 'review_text',
            'sentiment_label', 'bert_sentiment', 'bert_confidence',
            'problem_category', 'review_length'
        ]
        
        # 存在する列のみ保存
        available_columns = [col for col in columns_to_save if col in df.columns]
        df_to_save = df[available_columns].copy()
        
        df_to_save.to_csv(filepath, index=False, encoding='utf-8')
        
        print(f"💾 結果を保存: {filepath}")
        print(f"   保存件数: {len(df_to_save):,}件")
        print(f"   保存列: {available_columns}")
    
    def generate_summary_report(self, df: pd.DataFrame) -> str:
        """
        分析結果サマリーレポート生成
        
        Args:
            df (pd.DataFrame): 分析結果データ
            
        Returns:
            str: サマリーレポート
        """
        report = []
        report.append("="*80)
        report.append("AMAZON レビュー BERT感情分析 - サマリーレポート")
        report.append("="*80)
        
        # 基本統計
        report.append(f"\n📊 基本統計:")
        report.append(f"   総レビュー数: {len(df):,}件")
        report.append(f"   分析完了率: {df['bert_sentiment'].notna().sum() / len(df) * 100:.1f}%")
        
        # BERT感情分析結果
        if 'bert_sentiment' in df.columns:
            bert_dist = df['bert_sentiment'].value_counts()
            report.append(f"\n🤖 BERT感情分析結果:")
            for sentiment, count in bert_dist.items():
                percentage = count / len(df) * 100
                report.append(f"   {sentiment}: {count:,}件 ({percentage:.1f}%)")
            
            avg_confidence = df['bert_confidence'].mean()
            report.append(f"   平均信頼度: {avg_confidence:.3f}")
        
        # 精度評価
        if hasattr(self, 'validation_results') and self.validation_results:
            accuracy = self.validation_results['accuracy']
            report.append(f"\n🎯 精度評価:")
            report.append(f"   全体精度: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        # 問題分析
        if 'problem_category' in df.columns:
            negative_count = (df['bert_sentiment'] == 'negative').sum()
            problem_dist = df[df['problem_category'] != 'none']['problem_category'].value_counts()
            
            report.append(f"\n❌ ネガティブレビュー分析:")
            report.append(f"   ネガティブレビュー: {negative_count:,}件 ({negative_count/len(df)*100:.1f}%)")
            
            if len(problem_dist) > 0:
                report.append(f"   主要問題:")
                for problem, count in problem_dist.head(5).items():
                    percentage = count / negative_count * 100
                    report.append(f"     - {problem}: {count}件 ({percentage:.1f}%)")
        
        # キーワード分析
        if hasattr(self, 'sentiment_keywords') and self.sentiment_keywords:
            report.append(f"\n🔑 キーワード分析:")
            for sentiment in ['negative', 'positive']:
                if sentiment in self.sentiment_keywords:
                    keywords = self.sentiment_keywords[sentiment][:5]
                    keyword_list = [kw for kw, _ in keywords]
                    report.append(f"   {sentiment}の主要キーワード: {', '.join(keyword_list)}")
        
        # 次のステップ
        report.append(f"\n🚀 次のステップ (Phase 3):")
        report.append(f"   1. 改善提案エンジンの実装")
        report.append(f"   2. ROI算出システムの構築")
        report.append(f"   3. 競合比較機能の追加")
        report.append(f"   4. Webアプリケーション開発")
        
        report.append("="*80)
        
        return "\n".join(report)


def main():
    """
    Phase 2メイン実行関数
    
    1. データ読み込み
    2. BERT感情分析実行
    3. キーワード抽出
    4. 問題分類
    5. 精度評価
    6. 結果可視化・保存
    """
    print("Phase 2: BERT感情分析エンジン - 実行開始")
    print("="*60)
    
    try:
        # Step 1: データ読み込み
        print("Step 1: データ読み込み...")
        data_path = "data/processed_data.csv"
        
        if not os.path.exists(data_path):
            print(f"❌ データファイルが見つかりません: {data_path}")
            print("Phase 1のdata_collector.pyを先に実行してください")
            return None
        
        df = pd.read_csv(data_path)
        print(f"✅ データ読み込み完了: {len(df):,}件")
        
        # Step 2: 感情分析器の初期化
        print("\nStep 2: BERT感情分析器初期化...")
        analyzer = SentimentAnalyzer(batch_size=16)  # メモリに応じて調整
        
        # BERTモデル読み込み
        if not analyzer.load_bert_model():
            print("❌ BERTモデルの読み込みに失敗しました")
            return None
        
        # Step 3: 感情分析実行（サンプルテストから開始）
        print("\nStep 3: BERT感情分析実行...")
        
        # 最初は少量データでテスト
        test_sample_size = 1000
        print(f"⚠️ まずは{test_sample_size:,}件でテスト実行...")
        
        df_sample = df.head(test_sample_size)
        df_analyzed = analyzer.analyze_sentiment_batch(df_sample, chunk_size=100)
        
        print(f"✅ サンプル分析完了。全データ分析を実行しますか？")
        user_input = input("全データ分析を続行しますか？ (y/n): ")
        
        if user_input.lower() == 'y':
            print("全データ分析を実行中...")
            df_analyzed = analyzer.analyze_sentiment_batch(df, chunk_size=500)
        else:
            print("サンプルデータでの分析を続行...")
        
        # Step 4: キーワード抽出
        print("\nStep 4: キーワード抽出...")
        keywords = analyzer.extract_keywords(df_analyzed)
        
        # Step 5: 問題分類
        print("\nStep 5: 問題分類...")
        negative_keywords = keywords.get('negative', [])
        df_final = analyzer.classify_problems(df_analyzed, negative_keywords)
        
        # Step 6: 精度評価
        print("\nStep 6: 精度評価...")
        validation_results = analyzer.validate_predictions(df_final)
        
        # Step 7: 結果可視化
        print("\nStep 7: 結果可視化...")
        analyzer.visualize_results(df_final)
        
        # Step 8: 結果保存
        print("\nStep 8: 結果保存...")
        analyzer.save_results(df_final)
        
        # Step 9: サマリーレポート生成
        print("\nStep 9: サマリーレポート生成...")
        summary_report = analyzer.generate_summary_report(df_final)
        print(summary_report)
        
        # レポートファイル保存
        os.makedirs('results', exist_ok=True)
        with open('results/phase2_summary_report.txt', 'w', encoding='utf-8') as f:
            f.write(summary_report)
        
        print("\n🎉 Phase 2 完了!")
        print("   📁 保存ファイル:")
        print("      - results/sentiment_analysis_results.csv")
        print("      - results/sentiment_analysis_results.png")
        print("      - results/phase2_summary_report.txt")
        
        return df_final
        
    except Exception as e:
        print(f"\n❌ Phase 2実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return None


def quick_test():
    """
    クイックテスト用関数
    少量データでの動作確認
    """
    print("Phase 2 クイックテスト実行...")
    
    try:
        # テストデータ作成
        test_data = pd.DataFrame({
            'review_text': [
                "This product is amazing! I love it so much.",
                "Terrible quality. Very disappointed.",
                "It's okay, nothing special.",
                "Worst purchase ever. Complete waste of money.",
                "Great value for money. Highly recommended!"
            ],
            'rating': [5, 1, 3, 1, 5],
            'sentiment_label': ['positive', 'negative', 'neutral', 'negative', 'positive']
        })
        
        print(f"テストデータ: {len(test_data)}件")
        
        # 分析器初期化
        analyzer = SentimentAnalyzer(batch_size=2)
        
        if analyzer.load_bert_model():
            # 感情分析実行
            result = analyzer.analyze_sentiment_batch(test_data)
            
            # 結果表示
            print("\nテスト結果:")
            for i, row in result.iterrows():
                print(f"  {i+1}. [{row['rating']}★] {row['sentiment_label']} → {row['bert_sentiment']} ({row['bert_confidence']:.3f})") # type: ignore
                print(f"     「{row['review_text'][:50]}...」")
            
            print("\n✅ クイックテスト完了!")
            return result
        else:
            print("❌ クイックテスト失敗")
            return None
            
    except Exception as e:
        print(f"❌ クイックテストエラー: {e}")
        return None


if __name__ == "__main__":
    """
    Phase 2メイン実行
    
    実行方法:
    1. 通常実行: python src/sentiment_analyzer.py
    2. クイックテスト: python -c "from src.sentiment_analyzer import quick_test; quick_test()"
    """
    
    print("Amazon レビュー分析プロジェクト - Phase 2")
    print("目的: BERT感情分析エンジンの構築")
    print("="*60)
    
    # クイックテストから開始
    print("まずクイックテストを実行しますか？ (推奨)")
    test_choice = input("クイックテスト実行? (y/n): ")
    
    if test_choice.lower() == 'y':
        test_result = quick_test()
        if test_result is not None:
            print("\nメイン処理を実行しますか？")
            main_choice = input("メイン処理実行? (y/n): ")
            if main_choice.lower() == 'y':
                main()
    else:
        main()