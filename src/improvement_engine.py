"""
Amazon レビュー分析プロジェクト - 改善提案エンジン (Phase 3)

このファイルの目的：
1. Phase2の感情分析結果を基に具体的な改善提案を生成
2. 改善優先度をスコア化（ROI・影響度・実装コスト考慮）
3. 競合比較分析による差別化戦略の提示
4. ビジネス直結の実用的な改善レポート生成

Phase2からの引き継ぎ：
- BERT感情分析結果（83.7%精度）
- 問題分類結果（taste: 41%, quality: 17%等）
- キーワード抽出結果
- 568,454件の実データ分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class ImprovementEngine:
    """
    改善提案エンジン
    
    機能:
    - ネガティブレビュー分析からの改善提案生成
    - ROI・影響度を考慮した優先度算出
    - 競合比較分析
    - ビジネスレポート自動生成
    """
    
    def __init__(self, data_path: str = "results/sentiment_analysis_results.csv"):
        """
        改善提案エンジンの初期化
        
        Args:
            data_path (str): Phase2の分析結果ファイルパス
        """
        self.data_path = data_path
        self.df = None
        self.negative_reviews = None
        self.improvement_suggestions = []
        self.priority_scores = {}
        self.competitor_analysis = {}
        
        # 改善提案のテンプレート定義
        self.improvement_templates = {
            'taste': {
                'category': '味・風味改善',
                'suggestions': [
                    '味の調整・レシピ見直し',
                    '品質管理プロセスの強化',
                    '原材料の見直し・グレードアップ',
                    '製造工程の最適化',
                    '味のバリエーション追加'
                ],
                'base_cost': 50000,  # 基本実装コスト（USD）
                'impact_multiplier': 1.2  # ビジネス影響度倍数
            },
            'quality': {
                'category': '品質改善',
                'suggestions': [
                    'QA・検査工程の強化',
                    '製造設備のアップグレード',
                    'パッケージング改良',
                    '保存方法・賞味期限の見直し',
                    'サプライヤー品質基準の向上'
                ],
                'base_cost': 75000,
                'impact_multiplier': 1.5
            },
            'price': {
                'category': '価格戦略',
                'suggestions': [
                    '価格帯の見直し・最適化',
                    'コストダウン施策の実施',
                    'バリューパック・お得サイズの追加',
                    '競合価格分析による戦略調整',
                    'プロモーション・キャンペーンの実施'
                ],
                'base_cost': 25000,
                'impact_multiplier': 0.8
            },
            'shipping': {
                'category': '配送・梱包改善',
                'suggestions': [
                    '梱包材料・方法の改良',
                    '配送スピードの向上',
                    '破損防止対策の強化',
                    '配送業者の見直し',
                    '追跡システムの改善'
                ],
                'base_cost': 30000,
                'impact_multiplier': 0.6
            },
            'service': {
                'category': 'カスタマーサービス改善',
                'suggestions': [
                    'カスタマーサポート体制の強化',
                    '返品・交換プロセスの改善',
                    'FAQ・ヘルプドキュメントの充実',
                    '問い合わせ対応時間の短縮',
                    'ユーザーフィードバック収集の改善'
                ],
                'base_cost': 40000,
                'impact_multiplier': 0.7
            }
        }
        
        print(f"ImprovementEngine初期化完了")
        print(f"   データパス: {self.data_path}")
        print(f"   改善テンプレート: {len(self.improvement_templates)}カテゴリ")
    
    def load_analysis_results(self) -> bool:
        """
        Phase2の分析結果を読み込み
        
        Returns:
            bool: 読み込み成功フラグ
        """
        print("Phase2分析結果読み込み中...")
        
        if not os.path.exists(self.data_path):
            print(f"❌ 分析結果ファイルが見つかりません: {self.data_path}")
            print("Phase2のsentiment_analyzer.pyを先に実行してください")
            return False
        
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✅ 分析結果読み込み完了: {len(self.df):,}件")
            
            # ネガティブレビューの抽出
            self.negative_reviews = self.df[self.df['bert_sentiment'] == 'negative'].copy()
            print(f"   ネガティブレビュー: {len(self.negative_reviews):,}件")
            
            # データ品質チェック
            required_columns = ['bert_sentiment', 'bert_confidence', 'problem_category', 'rating']
            missing_columns = [col for col in required_columns if col not in self.df.columns]
            
            if missing_columns:
                print(f"⚠️ 不足している列: {missing_columns}")
                return False
            
            # 基本統計の表示
            self._print_data_summary()
            return True
            
        except Exception as e:
            print(f"❌ データ読み込みエラー: {e}")
            return False
    
    def _print_data_summary(self):
        """データの基本統計を表示"""
        print(f"\n📊 データ概要:")
        
        # 感情分布
        sentiment_dist = self.df['bert_sentiment'].value_counts() # type: ignore
        print(f"   感情分布: {sentiment_dist.to_dict()}")
        
        # 問題分布（ネガティブレビューのみ）
        if len(self.negative_reviews) > 0: # type: ignore
            problem_dist = self.negative_reviews['problem_category'].value_counts() # type: ignore
            print(f"   問題分布: {problem_dist.to_dict()}")
        
        # 評価分布
        rating_dist = self.df['rating'].value_counts().sort_index() # type: ignore
        print(f"   評価分布: {rating_dist.to_dict()}")
    
    def analyze_negative_patterns(self) -> Dict:
        """
        ネガティブレビューのパターン分析
        
        Returns:
            Dict: パターン分析結果
        """
        print("ネガティブパターン分析開始...")
        
        if self.negative_reviews is None or len(self.negative_reviews) == 0:
            print("⚠️ ネガティブレビューがありません")
            return {}
        
        patterns = {}
        
        # 1. 問題カテゴリ別分析
        problem_analysis = {}
        for category in self.negative_reviews['problem_category'].unique():
            if category == 'none':
                continue
                
            category_reviews = self.negative_reviews[
                self.negative_reviews['problem_category'] == category
            ]
            
            problem_analysis[category] = {
                'count': len(category_reviews),
                'percentage': len(category_reviews) / len(self.negative_reviews) * 100,
                'avg_rating': category_reviews['rating'].mean(),
                'avg_confidence': category_reviews['bert_confidence'].mean()
            }
        
        patterns['problem_analysis'] = problem_analysis
        
        # 2. 評価別ネガティブ分析
        rating_analysis = {}
        for rating in sorted(self.negative_reviews['rating'].unique()):
            rating_reviews = self.negative_reviews[self.negative_reviews['rating'] == rating]
            rating_analysis[rating] = {
                'count': len(rating_reviews),
                'problem_distribution': rating_reviews['problem_category'].value_counts().to_dict()
            }
        
        patterns['rating_analysis'] = rating_analysis
        
        # 3. 信頼度分析
        confidence_analysis = {
            'avg_confidence': self.negative_reviews['bert_confidence'].mean(),
            'high_confidence_count': (self.negative_reviews['bert_confidence'] > 0.8).sum(),
            'low_confidence_count': (self.negative_reviews['bert_confidence'] < 0.6).sum()
        }
        
        patterns['confidence_analysis'] = confidence_analysis
        
        print(f"✅ ネガティブパターン分析完了")
        print(f"   主要問題: {max(problem_analysis.keys(), key=lambda x: problem_analysis[x]['count']) if problem_analysis else 'なし'}")
        
        self.negative_patterns = patterns
        return patterns
    
    def calculate_priority_scores(self) -> Dict:
        """
        改善優先度スコアの算出
        
        Returns:
            Dict: 問題カテゴリ別優先度スコア
        """
        print("改善優先度スコア算出中...")
        
        if not hasattr(self, 'negative_patterns'):
            self.analyze_negative_patterns()
        
        priority_scores = {}
        total_negative = len(self.negative_reviews) # type: ignore
        
        for category, template in self.improvement_templates.items():
            if category not in self.negative_patterns['problem_analysis']:
                continue
                
            problem_data = self.negative_patterns['problem_analysis'][category]
            
            # 各要素の計算
            problem_frequency = problem_data['percentage'] / 100  # 0-1に正規化
            sentiment_intensity = 1.0 - problem_data['avg_confidence']  # 信頼度が低い=強い感情
            business_impact = template['impact_multiplier']
            implementation_ease = 1.0 / (template['base_cost'] / 25000)  # コストに反比例
            
            # 優先度スコアの計算
            priority_score = (
                problem_frequency * 0.4 +
                sentiment_intensity * 0.3 +
                business_impact * 0.2 +
                implementation_ease * 0.1
            )
            
            # ROI推定（簡単な計算）
            potential_impact = problem_data['count'] * 10  # 1件あたり$10の売上影響と仮定
            roi_estimate = potential_impact / template['base_cost']
            
            priority_scores[category] = {
                'priority_score': round(priority_score, 3),
                'problem_frequency': round(problem_frequency, 3),
                'sentiment_intensity': round(sentiment_intensity, 3),
                'business_impact': business_impact,
                'implementation_ease': round(implementation_ease, 3),
                'estimated_cost': template['base_cost'],
                'roi_estimate': round(roi_estimate, 2),
                'affected_reviews': problem_data['count']
            }
        
        # 優先度順にソート
        sorted_priorities = dict(sorted(
            priority_scores.items(), 
            key=lambda x: x[1]['priority_score'], 
            reverse=True
        ))
        
        print(f"✅ 優先度スコア算出完了")
        print(f"   最高優先度: {list(sorted_priorities.keys())[0] if sorted_priorities else 'なし'}")
        
        self.priority_scores = sorted_priorities
        return sorted_priorities
    
    def generate_improvement_suggestions(self) -> List[Dict]:
        """
        具体的な改善提案を生成
        
        Returns:
            List[Dict]: 改善提案リスト
        """
        print("改善提案生成中...")
        
        if not self.priority_scores:
            self.calculate_priority_scores()
        
        suggestions = []
        
        for category, scores in self.priority_scores.items():
            if category not in self.improvement_templates:
                continue
                
            template = self.improvement_templates[category]
            
            suggestion = {
                'category': template['category'],
                'problem_type': category,
                'priority_score': scores['priority_score'],
                'priority_rank': len(suggestions) + 1,
                'affected_reviews': scores['affected_reviews'],
                'estimated_cost': scores['estimated_cost'],
                'roi_estimate': scores['roi_estimate'],
                'suggestions': template['suggestions'],
                'implementation_timeline': self._estimate_timeline(category),
                'risk_level': self._assess_risk(category, scores),
                'success_probability': self._estimate_success_probability(scores)
            }
            
            suggestions.append(suggestion)
        
        print(f"✅ 改善提案生成完了: {len(suggestions)}件")
        
        self.improvement_suggestions = suggestions
        return suggestions
    
    def _estimate_timeline(self, category: str) -> str:
        """実装期間の推定"""
        timeline_map = {
            'taste': '3-6ヶ月',
            'quality': '2-4ヶ月', 
            'price': '1-2ヶ月',
            'shipping': '1-3ヶ月',
            'service': '2-3ヶ月'
        }
        return timeline_map.get(category, '2-4ヶ月')
    
    def _assess_risk(self, category: str, scores: Dict) -> str:
        """リスクレベルの評価"""
        cost = scores['estimated_cost']
        if cost > 60000:
            return '高'
        elif cost > 35000:
            return '中'
        else:
            return '低'
    
    def _estimate_success_probability(self, scores: Dict) -> float:
        """成功確率の推定"""
        base_probability = 0.7
        
        # 優先度が高いほど成功確率UP
        priority_bonus = scores['priority_score'] * 0.2
        
        # ROIが高いほど成功確率UP
        roi_bonus = min(scores['roi_estimate'] * 0.1, 0.2)
        
        success_prob = base_probability + priority_bonus + roi_bonus
        return min(success_prob, 0.95)  # 最大95%
    
    def estimate_roi_impact(self) -> Dict:
        """
        ROI・ビジネス影響度の詳細推定
        
        Returns:
            Dict: ROI・影響度分析結果
        """
        print("ROI・ビジネス影響度推定中...")
        
        if not self.improvement_suggestions:
            self.generate_improvement_suggestions()
        
        roi_analysis = {}
        total_investment = 0
        total_potential_return = 0
        
        for suggestion in self.improvement_suggestions:
            category = suggestion['problem_type']
            cost = suggestion['estimated_cost']
            affected_reviews = suggestion['affected_reviews']
            
            # 売上影響の推定（修正版）
            # 仮定: ネガティブレビュー1件 = $100の年間売上機会損失
            # （口コミ効果、リピート購入への影響、ブランド価値への影響を含む）
            revenue_loss_per_review = 100
            potential_revenue_recovery = affected_reviews * revenue_loss_per_review
            
            # 改善による評価向上効果（修正版）
            # 仮定: 改善により対象問題の85%が解決し、年間継続効果がある
            improvement_rate = 0.85
            annual_multiplier = 3  # 3年間の継続効果を考慮
            expected_revenue_recovery = potential_revenue_recovery * improvement_rate * annual_multiplier
            
            roi_analysis[category] = {
                'investment': cost,
                'potential_revenue_recovery': potential_revenue_recovery,
                'expected_revenue_recovery': expected_revenue_recovery,
                'roi_percentage': (expected_revenue_recovery / cost) * 100,
                'payback_period_months': cost / (expected_revenue_recovery / 12),
                'risk_adjusted_roi': expected_revenue_recovery * suggestion['success_probability'] / cost * 100
            }
            
            total_investment += cost
            total_potential_return += expected_revenue_recovery
        
        # 全体ROI
        overall_roi = {
            'total_investment': total_investment,
            'total_expected_return': total_potential_return,
            'overall_roi_percentage': (total_potential_return / total_investment) * 100 if total_investment > 0 else 0,
            'category_analysis': roi_analysis
        }
        
        print(f"✅ ROI分析完了")
        print(f"   総投資額: ${total_investment:,}")
        print(f"   期待リターン: ${total_potential_return:,}")
        print(f"   全体ROI: {overall_roi['overall_roi_percentage']:.1f}%")
        
        self.roi_analysis = overall_roi
        return overall_roi
    
    def compare_with_competitors(self) -> Dict:
        """
        競合比較分析（実データベース）
        
        Returns:
            Dict: 競合比較結果
        """
        print("競合比較分析中...")
        
        # 実データでの競合比較（同カテゴリ内での相対評価）
        competitor_analysis = {}
        
        # 全体評価分布
        overall_rating_dist = self.df['rating'].value_counts(normalize=True).sort_index() # type: ignore
        avg_rating = self.df['rating'].mean() # type: ignore
        negative_rate = (self.df['bert_sentiment'] == 'negative').mean() # type: ignore
        
        # ベンチマーク指標
        benchmark_metrics = {
            'average_rating': avg_rating,
            'negative_sentiment_rate': negative_rate,
            'rating_distribution': overall_rating_dist.to_dict(),
            'top_quartile_rating': self.df['rating'].quantile(0.75), # type: ignore
            'bottom_quartile_rating': self.df['rating'].quantile(0.25) # type: ignore
        }
        
        # 改善目標の設定
        improvement_targets = {
            'target_avg_rating': min(avg_rating + 0.5, 5.0),
            'target_negative_rate': max(negative_rate - 0.05, 0.05),
            'target_5star_rate': min(overall_rating_dist.get(5, 0) + 0.1, 0.8)
        }
        
        # 競合優位性分析
        competitive_position = {
            'current_percentile': self._calculate_percentile_position(avg_rating),
            'strengths': self._identify_strengths(),
            'weaknesses': self._identify_weaknesses(),
            'opportunities': self._identify_opportunities()
        }
        
        competitor_analysis = {
            'benchmark_metrics': benchmark_metrics,
            'improvement_targets': improvement_targets,
            'competitive_position': competitive_position
        }
        
        print(f"✅ 競合比較分析完了")
        print(f"   現在の平均評価: {avg_rating:.2f}")
        print(f"   改善目標評価: {improvement_targets['target_avg_rating']:.2f}")
        
        self.competitor_analysis = competitor_analysis
        return competitor_analysis
    
    def _calculate_percentile_position(self, rating: float) -> int:
        """評価の百分位を計算"""
        if rating >= 4.5:
            return 90
        elif rating >= 4.0:
            return 75
        elif rating >= 3.5:
            return 50
        elif rating >= 3.0:
            return 25
        else:
            return 10
    
    def _identify_strengths(self) -> List[str]:
        """強みの特定"""
        positive_reviews = self.df[self.df['bert_sentiment'] == 'positive'] # type: ignore
        positive_rate = len(positive_reviews) / len(self.df) # type: ignore
        
        strengths = []
        if positive_rate > 0.7:
            strengths.append("高い顧客満足度")
        if self.df['rating'].mean() > 4.0: # type: ignore
            strengths.append("良好な全体評価")
        if (self.df['rating'] == 5).mean() > 0.4: # type: ignore
            strengths.append("高評価レビューの多さ")
            
        return strengths if strengths else ["分析中"]
    
    def _identify_weaknesses(self) -> List[str]:
        """弱みの特定"""
        weaknesses = []
        
        if hasattr(self, 'negative_patterns'):
            problem_analysis = self.negative_patterns['problem_analysis']
            for category, data in problem_analysis.items():
                if data['percentage'] > 15:  # 15%以上の問題
                    category_name = self.improvement_templates.get(category, {}).get('category', category)
                    weaknesses.append(f"{category_name}の課題")
        
        return weaknesses if weaknesses else ["特定の課題なし"]
    
    def _identify_opportunities(self) -> List[str]:
        """機会の特定"""
        opportunities = []
        
        if hasattr(self, 'priority_scores'):
            top_priority = list(self.priority_scores.keys())[0] if self.priority_scores else None
            if top_priority:
                category_name = self.improvement_templates.get(top_priority, {}).get('category', top_priority)
                opportunities.append(f"{category_name}の改善による大幅な向上")
        
        negative_rate = (self.df['bert_sentiment'] == 'negative').mean() # type: ignore
        if negative_rate > 0.15:
            opportunities.append("ネガティブレビュー削減による評価向上")
        
        return opportunities if opportunities else ["継続的な品質向上"]
    
    def create_business_report(self, save_path: str = "results/business_improvement_report.txt") -> str:
        """
        ビジネス改善レポートの生成
        
        Args:
            save_path (str): レポート保存パス
            
        Returns:
            str: 生成されたレポート内容
        """
        print("ビジネス改善レポート生成中...")
        
        # 必要な分析がない場合は実行
        if not hasattr(self, 'improvement_suggestions'):
            self.generate_improvement_suggestions()
        if not hasattr(self, 'roi_analysis'):
            self.estimate_roi_impact()
        if not hasattr(self, 'competitor_analysis'):
            self.compare_with_competitors()
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("AMAZON レビュー分析 - ビジネス改善提案レポート")
        report_lines.append("="*80)
        report_lines.append(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # エグゼクティブサマリー
        report_lines.append("\n" + "="*60)
        report_lines.append("📋 エグゼクティブサマリー")
        report_lines.append("="*60)
        
        total_reviews = len(self.df) # type: ignore
        negative_count = len(self.negative_reviews) # type: ignore
        negative_rate = negative_count / total_reviews * 100
        
        report_lines.append(f"• 総レビュー数: {total_reviews:,}件")
        report_lines.append(f"• ネガティブレビュー: {negative_count:,}件 ({negative_rate:.1f}%)")
        report_lines.append(f"• 平均評価: {self.df['rating'].mean():.2f}/5.0") # type: ignore
        
        if self.priority_scores:
            top_priority = list(self.priority_scores.keys())[0]
            top_category = self.improvement_templates[top_priority]['category']
            report_lines.append(f"• 最優先改善項目: {top_category}")
        
        if hasattr(self, 'roi_analysis'):
            total_investment = self.roi_analysis['total_investment']
            total_return = self.roi_analysis['total_expected_return']
            overall_roi = self.roi_analysis['overall_roi_percentage']
            report_lines.append(f"• 推定投資額: ${total_investment:,}")
            report_lines.append(f"• 期待リターン: ${total_return:,}")
            report_lines.append(f"• 全体ROI: {overall_roi:.1f}%")
        
        # 改善提案詳細
        report_lines.append("\n" + "="*60)
        report_lines.append("🎯 改善提案詳細（優先度順）")
        report_lines.append("="*60)
        
        for i, suggestion in enumerate(self.improvement_suggestions, 1):
            report_lines.append(f"\n{i}. {suggestion['category']}")
            report_lines.append(f"   優先度スコア: {suggestion['priority_score']}")
            report_lines.append(f"   影響レビュー数: {suggestion['affected_reviews']}件")
            report_lines.append(f"   推定コスト: ${suggestion['estimated_cost']:,}")
            report_lines.append(f"   ROI: {suggestion['roi_estimate']:.1f}%")
            report_lines.append(f"   実装期間: {suggestion['implementation_timeline']}")
            report_lines.append(f"   リスクレベル: {suggestion['risk_level']}")
            report_lines.append(f"   成功確率: {suggestion['success_probability']*100:.0f}%")
            report_lines.append("   具体的施策:")
            for j, action in enumerate(suggestion['suggestions'][:3], 1):
                report_lines.append(f"     {j}) {action}")
        
        # ROI分析
        if hasattr(self, 'roi_analysis'):
            report_lines.append("\n" + "="*60)
            report_lines.append("💰 ROI・投資対効果分析")
            report_lines.append("="*60)
            
            for category, roi_data in self.roi_analysis['category_analysis'].items():
                category_name = self.improvement_templates[category]['category']
                report_lines.append(f"\n{category_name}:")
                report_lines.append(f"   投資額: ${roi_data['investment']:,}")
                report_lines.append(f"   期待収益: ${roi_data['expected_revenue_recovery']:,}")
                report_lines.append(f"   ROI: {roi_data['roi_percentage']:.1f}%")
                report_lines.append(f"   回収期間: {roi_data['payback_period_months']:.1f}ヶ月")
        
        # 競合比較
        if hasattr(self, 'competitor_analysis'):
            report_lines.append("\n" + "="*60)
            report_lines.append("🏆 競合比較・市場ポジション")
            report_lines.append("="*60)
            
            benchmark = self.competitor_analysis['benchmark_metrics']
            position = self.competitor_analysis['competitive_position']
            
            report_lines.append(f"現在のポジション: {position['current_percentile']}%ile")
            report_lines.append(f"業界平均評価: {benchmark['average_rating']:.2f}")
            report_lines.append(f"ネガティブ率: {benchmark['negative_sentiment_rate']*100:.1f}%")
            
            report_lines.append("\n強み:")
            for strength in position['strengths']:
                report_lines.append(f"   • {strength}")
            
            report_lines.append("\n改善機会:")
            for opportunity in position['opportunities']:
                report_lines.append(f"   • {opportunity}")
        
        # 実装ロードマップ
        report_lines.append("\n" + "="*60)
        report_lines.append("🗓️ 実装ロードマップ")
        report_lines.append("="*60)
        
        report_lines.append("\nフェーズ1 (最初の3ヶ月):")
        if self.improvement_suggestions:
            top_suggestion = self.improvement_suggestions[0]
            report_lines.append(f"   • {top_suggestion['category']}の改善開始")
            report_lines.append(f"   • 予算: ${top_suggestion['estimated_cost']:,}")
        
        if len(self.improvement_suggestions) > 1:
            report_lines.append("\nフェーズ2 (4-6ヶ月):")
            second_suggestion = self.improvement_suggestions[1]
            report_lines.append(f"   • {second_suggestion['category']}の改善実施")
            report_lines.append(f"   • 予算: ${second_suggestion['estimated_cost']:,}")
        
        # 成功指標
        report_lines.append("\n" + "="*60)
        report_lines.append("📊 成功指標・KPI")
        report_lines.append("="*60)
        
        current_avg = self.df['rating'].mean() # type: ignore
        target_avg = min(current_avg + 0.5, 5.0)
        current_negative_rate = negative_rate
        target_negative_rate = max(current_negative_rate - 5, 5)
        
        report_lines.append(f"• 平均評価: {current_avg:.2f} → {target_avg:.2f}")
        report_lines.append(f"• ネガティブ率: {current_negative_rate:.1f}% → {target_negative_rate:.1f}%")
        report_lines.append(f"• 5星評価率: +10%向上")
        report_lines.append(f"• 顧客満足度: +15%向上")
        
        report_lines.append("\n" + "="*80)
        
        # レポート保存
        report_content = "\n".join(report_lines)
        
        os.makedirs("results", exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"✅ ビジネス改善レポート生成完了")
        print(f"   保存先: {save_path}")
        
        return report_content
    
    def visualize_results(self, save_plots: bool = True):
        """
        改善提案結果の可視化
        
        Args:
            save_plots (bool): グラフ保存フラグ
        """
        print("改善提案結果可視化中...")
        
        # 必要な分析がない場合は実行
        if not hasattr(self, 'improvement_suggestions'):
            self.generate_improvement_suggestions()
        if not hasattr(self, 'roi_analysis'):
            self.estimate_roi_impact()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('改善提案分析結果', fontsize=16, fontweight='bold')
        
        # 1. 優先度スコア
        ax1 = axes[0, 0]
        categories = []
        scores = []
        for suggestion in self.improvement_suggestions:
            categories.append(suggestion['category'])
            scores.append(suggestion['priority_score'])
        
        bars = ax1.bar(categories, scores, color='skyblue', edgecolor='navy')
        ax1.set_title('改善優先度スコア')
        ax1.set_ylabel('優先度スコア')
        ax1.tick_params(axis='x', rotation=45)
        
        # 値をバーの上に表示
        for bar, score in zip(bars, scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.2f}', ha='center', va='bottom')
        
        # 2. ROI分析
        ax2 = axes[0, 1]
        roi_categories = []
        roi_values = []
        
        if hasattr(self, 'roi_analysis'):
            for category, roi_data in self.roi_analysis['category_analysis'].items():
                category_name = self.improvement_templates[category]['category']
                roi_categories.append(category_name)
                roi_values.append(roi_data['roi_percentage'])
        
        if roi_values:
            bars2 = ax2.bar(roi_categories, roi_values, color='lightgreen', edgecolor='darkgreen')
            ax2.set_title('ROI分析 (%)')
            ax2.set_ylabel('ROI (%)')
            ax2.tick_params(axis='x', rotation=45)
            
            for bar, roi in zip(bars2, roi_values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{roi:.0f}%', ha='center', va='bottom')
        
        # 3. 投資額 vs 期待リターン
        ax3 = axes[1, 0]
        investments = []
        returns = []
        suggestion_labels = []
        
        for suggestion in self.improvement_suggestions:
            investments.append(suggestion['estimated_cost'])
            if hasattr(self, 'roi_analysis'):
                category = suggestion['problem_type']
                if category in self.roi_analysis['category_analysis']:
                    returns.append(self.roi_analysis['category_analysis'][category]['expected_revenue_recovery'])
                else:
                    returns.append(0)
            else:
                returns.append(0)
            suggestion_labels.append(suggestion['category'][:10])  # 短縮
        
        if investments and returns:
            scatter = ax3.scatter(investments, returns, s=100, alpha=0.7, c=scores, cmap='viridis')
            ax3.set_xlabel('投資額 ($)')
            ax3.set_ylabel('期待リターン ($)')
            ax3.set_title('投資額 vs 期待リターン')
            
            # ラベル追加
            for i, label in enumerate(suggestion_labels):
                ax3.annotate(label, (investments[i], returns[i]), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8)
            
            plt.colorbar(scatter, ax=ax3, label='優先度スコア')
        
        # 4. 問題分布
        ax4 = axes[1, 1]
        if hasattr(self, 'negative_patterns'):
            problem_data = self.negative_patterns['problem_analysis']
            problem_names = []
            problem_counts = []
            
            for category, data in problem_data.items():
                category_name = self.improvement_templates.get(category, {}).get('category', category)
                problem_names.append(category_name)
                problem_counts.append(data['count'])
            
            if problem_counts:
                ax4.pie(problem_counts, labels=problem_names, autopct='%1.1f%%', startangle=90)
                ax4.set_title('ネガティブレビュー問題分布')
        
        plt.tight_layout()
        
        if save_plots:
            os.makedirs('results', exist_ok=True)
            plt.savefig('results/improvement_analysis_results.png', dpi=300, bbox_inches='tight')
            print("📊 グラフを保存: results/improvement_analysis_results.png")
        
        plt.show()
    
    def save_results(self, filename: str = "improvement_suggestions.csv"):
        """
        改善提案結果をCSVで保存
        
        Args:
            filename (str): 保存ファイル名
        """
        if not self.improvement_suggestions:
            print("⚠️ 保存する改善提案がありません")
            return
        
        # データフレーム作成
        results_data = []
        for suggestion in self.improvement_suggestions:
            results_data.append({
                'priority_rank': suggestion['priority_rank'],
                'category': suggestion['category'],
                'problem_type': suggestion['problem_type'],
                'priority_score': suggestion['priority_score'],
                'affected_reviews': suggestion['affected_reviews'],
                'estimated_cost': suggestion['estimated_cost'],
                'roi_estimate': suggestion['roi_estimate'],
                'implementation_timeline': suggestion['implementation_timeline'],
                'risk_level': suggestion['risk_level'],
                'success_probability': suggestion['success_probability'],
                'top_suggestion': suggestion['suggestions'][0] if suggestion['suggestions'] else ''
            })
        
        df_results = pd.DataFrame(results_data)
        
        os.makedirs('results', exist_ok=True)
        filepath = os.path.join('results', filename)
        df_results.to_csv(filepath, index=False, encoding='utf-8')
        
        print(f"💾 改善提案結果を保存: {filepath}")
        print(f"   保存件数: {len(results_data)}件")


def main():
    """
    Phase 3メイン実行関数
    """
    print("Phase 3: 改善提案エンジン - 実行開始")
    print("="*60)
    
    try:
        # Step 1: 改善提案エンジン初期化
        print("Step 1: 改善提案エンジン初期化...")
        engine = ImprovementEngine()
        
        # Step 2: Phase2結果読み込み
        print("\nStep 2: Phase2分析結果読み込み...")
        if not engine.load_analysis_results():
            print("❌ Phase2結果の読み込みに失敗しました")
            return None
        
        # Step 3: ネガティブパターン分析
        print("\nStep 3: ネガティブパターン分析...")
        patterns = engine.analyze_negative_patterns()
        
        # Step 4: 優先度スコア算出
        print("\nStep 4: 改善優先度スコア算出...")
        priority_scores = engine.calculate_priority_scores()
        
        # Step 5: 改善提案生成
        print("\nStep 5: 改善提案生成...")
        suggestions = engine.generate_improvement_suggestions()
        
        # Step 6: ROI・影響度分析
        print("\nStep 6: ROI・ビジネス影響度分析...")
        roi_analysis = engine.estimate_roi_impact()
        
        # Step 7: 競合比較分析
        print("\nStep 7: 競合比較分析...")
        competitor_analysis = engine.compare_with_competitors()
        
        # Step 8: 結果可視化
        print("\nStep 8: 結果可視化...")
        engine.visualize_results()
        
        # Step 9: 結果保存
        print("\nStep 9: 結果保存...")
        engine.save_results()
        
        # Step 10: ビジネスレポート生成
        print("\nStep 10: ビジネスレポート生成...")
        report = engine.create_business_report()
        
        print("\n🎉 Phase 3 完了!")
        print("   📁 生成ファイル:")
        print("      - results/improvement_suggestions.csv")
        print("      - results/improvement_analysis_results.png")
        print("      - results/business_improvement_report.txt")
        
        # 簡易結果表示
        print(f"\n📊 改善提案サマリー:")
        for i, suggestion in enumerate(suggestions[:3], 1):
            # ROI分析から実際のROI値を取得
            category = suggestion['problem_type']
            actual_roi = roi_analysis['category_analysis'].get(category, {}).get('roi_percentage', 0)
            print(f"   {i}. {suggestion['category']} (優先度: {suggestion['priority_score']:.2f})")
            print(f"      コスト: ${suggestion['estimated_cost']:,}, ROI: {actual_roi:.1f}%")
        
        return engine
        
    except Exception as e:
        print(f"\n❌ Phase 3実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    """
    Phase 3メイン実行
    """
    print("Amazon レビュー分析プロジェクト - Phase 3")
    print("目的: 改善提案エンジンの構築")
    print("="*60)
    
    main()