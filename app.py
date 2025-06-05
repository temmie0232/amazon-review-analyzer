"""
Amazon レビュー分析ダッシュボード - Phase 4 (デプロイ対応版)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from io import StringIO

# ページ設定
st.set_page_config(
    page_title="Amazon レビュー分析ダッシュボード",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """メイン関数"""
    
    # ヘッダー
    st.markdown('<h1 class="main-header">📊 Amazon レビュー分析ダッシュボード</h1>', 
                unsafe_allow_html=True)
    
    # サイドバーでページ選択
    st.sidebar.title("🧭 ナビゲーション")
    page = st.sidebar.selectbox(
        "ページを選択",
        [
            "🏠 ホーム",
            "📊 デモ・サンプル分析",
            "📄 分析結果確認",
            "💡 改善提案",
            "📈 ROI分析",
            "📄 レポート・技術情報"
        ]
    )
    
    # デプロイ版の注意書き
    st.sidebar.markdown("---")
    st.sidebar.info("""
    💡 **デプロイ版について**
    
    このデプロイ版では軽量化のため、事前に計算済みの分析結果を表示しています。
    
    実際の開発では568,454件のKaggleデータを使用。
    """)
    
    # 各ページの表示
    if page == "🏠 ホーム":
        show_home_page()
    elif page == "📊 デモ・サンプル分析":
        show_demo_page()
    elif page == "📄 分析結果確認":
        show_results_page()
    elif page == "💡 改善提案":
        show_improvement_page()
    elif page == "📈 ROI分析":
        show_roi_page()
    elif page == "📄 レポート・技術情報":
        show_tech_page()


def show_home_page():
    """ホームページ"""
    st.header("🏠 プロジェクト概要")
    
    # プロジェクト成果の表示
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📈 実データ処理",
            value="568,454件",
            delta="Kaggle API取得"
        )
    
    with col2:
        st.metric(
            label="🎯 BERT精度",
            value="83.7%",
            delta="商用レベル"
        )
    
    with col3:
        st.metric(
            label="💰 最高ROI",
            value="37.2%",
            delta="味・風味改善"
        )
    
    with col4:
        st.metric(
            label="⭐ 改善項目",
            value="5カテゴリ",
            delta="優先度付き"
        )
    
    # プロジェクト説明
    st.markdown("---")
    st.subheader("📋 プロジェクトについて")
    
    st.markdown("""
    ### 🎯 目的
    **Amazon商品レビューから具体的な改善提案を自動生成するAIシステム**
    
    ### 🌟 特徴
    - **実データ活用**: Kaggleから568,454件の実際のAmazonレビューを取得
    - **高精度AI分析**: HuggingFace BERTで83.7%の感情分析精度を達成
    - **ビジネス価値**: 37.2%のROIで投資価値のある改善提案を実現
    - **エンドツーエンド**: データ収集→分析→改善提案→レポート出力の全工程
    
    ### 🛠️ 技術スタック
    - **機械学習**: HuggingFace BERT, scikit-learn
    - **データ処理**: pandas, numpy, Kaggle API
    - **可視化**: plotly, matplotlib, seaborn
    - **Webアプリ**: Streamlit
    
    ### 📊 開発期間・プロセス
    - **総期間**: 7日間
    - **Phase 1-2**: データ収集・BERT感情分析（83.7%精度達成）
    - **Phase 3**: 改善提案エンジン・ROI算出システム
    - **Phase 4**: Streamlit Webアプリ開発・デプロイ
    """)
    
    # 使い方ガイド
    st.markdown("---")
    st.subheader("🚀 デモの使い方")
    
    with st.expander("📖 ステップバイステップガイド"):
        st.markdown("""
        1. **📊 デモ・サンプル分析**: サンプルデータでシステムを体験
        2. **📄 分析結果確認**: BERT感情分析の結果を確認
        3. **💡 改善提案**: 優先度付きの改善案を確認
        4. **📈 ROI分析**: 投資対効果を確認
        5. **📄 レポート・技術情報**: 技術詳細・開発プロセスを確認
        """)


def show_demo_page():
    """デモ・サンプル分析ページ"""
    st.header("📊 デモ・サンプル分析")
    
    st.info("""
    💡 **このデモについて**
    
    実際の開発では568,454件のKaggle Amazonレビューデータを使用しましたが、
    デプロイ版では軽量化のため、事前に分析済みの結果データを表示します。
    """)
    
    # サンプルデータ生成
    sample_data = create_sample_data()
    
    st.subheader("📋 サンプルデータ（1,000件相当）")
    st.dataframe(sample_data.head(10))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("総レビュー数", "1,000件")
    with col2:
        st.metric("平均評価", "4.18/5.0")
    with col3:
        st.metric("ネガティブ率", "17.8%")
    
    # 基本分析結果
    st.subheader("😊 感情分析結果")
    
    sentiment_data = {
        'Sentiment': ['Positive', 'Negative', 'Neutral'],
        'Count': [742, 178, 80],
        'Percentage': [74.2, 17.8, 8.0]
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_pie = px.pie(
            values=sentiment_data['Count'],
            names=sentiment_data['Sentiment'],
            title="BERT感情分析結果分布"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        fig_bar = px.bar(
            x=sentiment_data['Sentiment'],
            y=sentiment_data['Count'],
            title="感情別レビュー件数",
            color=sentiment_data['Sentiment'],
            color_discrete_map={
                'Positive': '#2E8B57',
                'Negative': '#DC143C', 
                'Neutral': '#4682B4'
            }
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # 問題分析
    st.subheader("❌ 問題分析結果")
    
    problem_data = {
        'Category': ['味・風味', '品質', '価格', '配送', 'サービス'],
        'Count': [73, 30, 16, 11, 3],
        'Percentage': [41.0, 16.9, 9.0, 6.2, 1.7]
    }
    
    fig_problems = px.bar(
        x=problem_data['Category'],
        y=problem_data['Count'],
        title="ネガティブレビュー問題分類",
        color=problem_data['Count'],
        color_continuous_scale='Reds'
    )
    st.plotly_chart(fig_problems, use_container_width=True)


def show_results_page():
    """分析結果確認ページ"""
    st.header("📄 分析結果詳細")
    
    # BERT分析結果
    st.subheader("🤖 BERT感情分析詳細")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("全体精度", "83.7%", delta="商用レベル")
    with col2:
        st.metric("平均信頼度", "0.844", delta="高信頼")
    with col3:
        st.metric("処理速度", "1000件/分", delta="高速処理")
    
    # 精度詳細
    st.subheader("🎯 クラス別精度")
    
    accuracy_data = {
        'Class': ['Positive', 'Negative', 'Neutral'],
        'Precision': [0.947, 0.663, 0.200],
        'Recall': [0.901, 0.814, 0.213],
        'F1-Score': [0.924, 0.731, 0.206]
    }
    
    df_accuracy = pd.DataFrame(accuracy_data)
    st.dataframe(df_accuracy)
    
    # サンプルレビュー表示
    st.subheader("📝 分析済みレビューサンプル")
    
    sample_reviews = create_sample_analyzed_data()
    st.dataframe(sample_reviews)


def show_improvement_page():
    """改善提案ページ"""
    st.header("💡 改善提案")
    
    st.subheader("🎯 改善優先度ランキング")
    
    # 改善提案データ
    improvement_data = create_improvement_data()
    
    for idx, row in improvement_data.iterrows():
        with st.expander(f"#{row['rank']} {row['category']} (優先度: {row['priority']:.3f})"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**問題件数**: {row['affected_reviews']}件")
                st.write(f"**投資額**: ${row['investment']:,.0f}")
            
            with col2:
                st.write(f"**期待リターン**: ${row['expected_return']:,.0f}")
                st.write(f"**ROI**: {row['roi']:.1f}%")
            
            st.write(f"**改善案**: {row['suggestion']}")
            
            col3, col4 = st.columns(2)
            with col3:
                st.write(f"**実装期間**: {row['timeline']}")
                st.write(f"**リスクレベル**: {row['risk']}")
            with col4:
                st.write(f"**成功確率**: {row['success']:.1f}%")


def show_roi_page():
    """ROI分析ページ"""
    st.header("📈 ROI分析")
    
    # ROI データ
    improvement_data = create_improvement_data()
    
    # ROI可視化
    fig_roi = px.bar(
        improvement_data,
        x='category',
        y='roi',
        title="カテゴリ別ROI分析",
        color='roi',
        color_continuous_scale='RdYlGn',
        labels={'roi': 'ROI (%)'}
    )
    st.plotly_chart(fig_roi, use_container_width=True)
    
    # 投資対効果スキャッター
    fig_scatter = px.scatter(
        improvement_data,
        x='investment',
        y='expected_return',
        size='priority',
        color='category',
        title="投資額 vs 期待リターン",
        hover_data=['roi'],
        labels={
            'investment': '投資額 ($)',
            'expected_return': '期待リターン ($)',
            'priority': '優先度スコア'
        }
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # ROI効率ランキング
    st.subheader("🏆 ROI効率ランキング")
    
    for idx, row in improvement_data.iterrows():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("カテゴリ", row['category'])
        with col2:
            st.metric("ROI", f"{row['roi']:.1f}%")
        with col3:
            st.metric("投資額", f"${row['investment']:,.0f}")
        with col4:
            st.metric("期待リターン", f"${row['expected_return']:,.0f}")
    
    # 投資サマリー
    st.subheader("💰 投資サマリー")
    total_investment = improvement_data['investment'].sum()
    total_return = improvement_data['expected_return'].sum()
    total_roi = ((total_return - total_investment) / total_investment) * 100
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("総投資額", f"${total_investment:,.0f}")
    with col2:
        st.metric("総期待リターン", f"${total_return:,.0f}")
    with col3:
        st.metric("全体ROI", f"{total_roi:.1f}%")


def show_tech_page():
    """技術情報・レポートページ"""
    st.header("📄 技術情報・開発レポート")
    
    # 技術仕様
    st.subheader("🛠️ 技術仕様")
    
    tech_info = """
    ### 📊 データ処理
    - **データソース**: Kaggle Amazon Fine Food Reviews (568,454件)
    - **期間**: 1999-2012年の実際のAmazonレビュー
    - **前処理**: 欠損値処理、感情ラベル付与、品質フラグ
    
    ### 🤖 機械学習
    - **感情分析モデル**: cardiffnlp/twitter-roberta-base-sentiment-latest
    - **精度**: 83.7% (商用レベル)
    - **キーワード抽出**: TF-IDF + 感情別重み付け
    
    ### 💡 改善提案システム
    - **優先度算出**: 頻度40% + 感情30% + 影響20% + 容易10%
    - **ROI計算**: 3年継続効果・85%改善率・$100/件売上影響
    - **問題分類**: taste, quality, price, shipping, service
    
    ### 🏗️ システム構成
    - **フロントエンド**: Streamlit
    - **バックエンド**: Python (pandas, numpy, scikit-learn)
    - **可視化**: plotly, matplotlib
    - **デプロイ**: Streamlit Cloud
    """
    
    st.markdown(tech_info)
    
    # 開発プロセス
    st.subheader("📅 7日間開発プロセス")
    
    phases = {
        'Phase': ['Phase 1', 'Phase 2', 'Phase 3', 'Phase 4'],
        '内容': [
            'データ収集・前処理システム',
            'BERT感情分析エンジン', 
            '改善提案・ROI算出システム',
            'Streamlit Webアプリ'
        ],
        '期間': ['1-2日目', '3-4日目', '5-6日目', '7日目'],
        '主要成果': [
            '568k件実データ取得',
            '83.7%精度達成',
            '37.2% ROI提案',
            'Webアプリ公開'
        ]
    }
    
    df_phases = pd.DataFrame(phases)
    st.dataframe(df_phases)
    
    # ダウンロード用レポート
    st.subheader("📄 レポートダウンロード")
    
    # サマリーレポート生成
    summary_report = generate_summary_report()
    
    st.download_button(
        label="📄 プロジェクトサマリーレポート",
        data=summary_report,
        file_name="amazon_review_analysis_summary.md",
        mime="text/markdown"
    )
    
    # GitHub リンク
    st.subheader("🔗 関連リンク")
    st.markdown("""
    - **GitHub Repository**: プロジェクトの完全なソースコード
    - **Kaggle Dataset**: [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
    - **HuggingFace Model**: [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
    """)


# データ生成関数群（依存関係なし）
def create_sample_data():
    """サンプルデータ生成（numpy依存）"""
    np.random.seed(42)
    
    # より現実的なレビューデータ
    sample_reviews = [
        "This product is amazing! Great taste and quality.",
        "Not satisfied with the flavor. Too bland and disappointing.",
        "Decent product, but overpriced for what it offers.",
        "Excellent customer service and fast shipping.",
        "Quality issues - product arrived damaged and unusable.",
        "Love the taste! Will definitely buy again.",
        "Poor packaging, contents were spilled everywhere.",
        "Great value for money. Highly recommend to others.",
        "Product doesn't match the description at all.",
        "Outstanding quality and delicious flavor profile."
    ]
    
    data = {
        'review_id': range(1, 11),
        'rating': np.random.choice([1,2,3,4,5], 10, p=[0.1, 0.05, 0.15, 0.35, 0.35]),
        'review_text': sample_reviews,
        'bert_sentiment': ['positive', 'negative', 'neutral', 'positive', 'negative', 
                          'positive', 'negative', 'positive', 'negative', 'positive'],
        'bert_confidence': [0.95, 0.89, 0.72, 0.91, 0.84, 0.93, 0.87, 0.88, 0.82, 0.96]
    }
    return pd.DataFrame(data)


def create_sample_analyzed_data():
    """分析済みサンプルデータ"""
    data = {
        'Review Text': [
            "Amazing taste and quality!",
            "Terrible flavor, very disappointed",
            "Okay product, nothing special",
            "Poor packaging and shipping",
            "Great customer service"
        ],
        'Rating': [5, 1, 3, 2, 5],
        'BERT Sentiment': ['positive', 'negative', 'neutral', 'negative', 'positive'],
        'Confidence': [0.95, 0.89, 0.72, 0.84, 0.91],
        'Problem Category': ['none', 'taste', 'none', 'shipping', 'none']
    }
    return pd.DataFrame(data)


def create_improvement_data():
    """改善提案データ生成"""
    data = {
        'rank': [1, 2, 3, 4, 5],
        'category': ['味・風味改善', '品質改善', '価格戦略', '配送・梱包改善', 'カスタマーサービス改善'],
        'priority': [0.526, 0.453, 0.379, 0.314, 0.272],
        'affected_reviews': [73, 30, 16, 11, 3],
        'investment': [50000, 75000, 25000, 30000, 40000],
        'expected_return': [68600, 82650, 29075, 32400, 42400],
        'roi': [37.2, 10.2, 16.3, 8.0, 6.0],
        'timeline': ['3-6ヶ月', '2-4ヶ月', '1-2ヶ月', '1-3ヶ月', '2-3ヶ月'],
        'risk': ['中', '高', '低', '低', '中'],
        'success': [80.6, 79.1, 77.7, 76.3, 75.4],
        'suggestion': [
            '味の調整・レシピ見直し',
            'QA・検査工程の強化', 
            '価格帯の見直し・最適化',
            '梱包材料・方法の改良',
            'カスタマーサポート体制の強化'
        ]
    }
    return pd.DataFrame(data)


def generate_summary_report():
    """サマリーレポート生成"""
    report = """# Amazon レビュー分析プロジェクト - サマリーレポート

## 📊 プロジェクト概要
- **開発期間**: 7日間
- **技術スタック**: Python, HuggingFace BERT, Streamlit, Kaggle API
- **データ規模**: 568,454件の実Amazon レビューデータ

## 🎯 主要成果
### Phase 1-2: データ収集・AI分析
- Kaggle APIで568,454件の実データ取得
- BERT感情分析で83.7%の高精度達成
- 178件のネガティブレビューを詳細分析

### Phase 3: 改善提案システム
- 5カテゴリの問題分類（味41%、品質17%等）
- ROI算出システム（最高37.2%の効率的提案）
- 優先度スコアリング（4要素重み付け）

### Phase 4: Webアプリ化
- Streamlit による直感的UI
- リアルタイム分析・可視化
- Streamlit Cloud での無料公開

## 💰 ビジネス価値
- **総投資額**: $220,000
- **総期待リターン**: $255,125
- **全体ROI**: 15.9%
- **最高効率**: 味・風味改善 37.2% ROI

## 🛠️ 技術的特徴
- **大規模データ処理**: 568k件の効率的処理
- **高精度AI**: 商用レベル83.7%精度
- **実用的システム**: エンドツーエンドML開発
- **ビジネス直結**: 具体的改善提案の自動生成

## 📈 改善提案詳細
1. **味・風味改善** (優先度1位)
   - 投資額: $50,000
   - 期待リターン: $68,600
   - ROI: 37.2%

2. **品質改善** (優先度2位)
   - 投資額: $75,000
   - 期待リターン: $82,650
   - ROI: 10.2%

3. **価格戦略** (優先度3位)
   - 投資額: $25,000
   - 期待リターン: $29,075
   - ROI: 16.3%

---
生成日時: 2025年6月5日
プロジェクト: Amazon商品レビュー分析による改善提案システム
"""
    return report


if __name__ == "__main__":
    main()