# -*- coding: utf-8 -*-
import unittest
from types import SimpleNamespace

from src.market_analyzer import MarketAnalyzer, MarketIndex, MarketOverview


class MarketAnalyzerAIStockSelectionTestCase(unittest.TestCase):
    def _overview(self) -> MarketOverview:
        return MarketOverview(
            date="2026-07-01",
            indices=[
                MarketIndex(
                    code="SPX",
                    name="S&P 500",
                    current=6200.0,
                    change_pct=0.6,
                )
            ],
            top_sectors=[{"name": "半导体", "change_pct": 2.5}],
            top_concepts=[{"name": "AI算力", "change_pct": 3.1}],
        )

    def test_us_prompt_has_separate_ai_stock_selection_radar(self) -> None:
        analyzer = MarketAnalyzer(region="us", config=SimpleNamespace(report_language="zh"))

        prompt = analyzer._build_review_prompt(self._overview(), [])

        self.assertIn("AI选股雷达", prompt)
        self.assertIn("每日热门股", prompt)
        self.assertIn("有潜力的股票", prompt)
        self.assertIn("可抄底观察信号", prompt)
        self.assertIn("只分析美股", prompt)
        self.assertIn("Nvidia(NVDA)", prompt)
        self.assertNotIn("只分析 A 股", prompt)
        self.assertNotIn("中际旭创", prompt)

    def test_cn_prompt_has_separate_ai_stock_selection_radar(self) -> None:
        analyzer = MarketAnalyzer(region="cn", config=SimpleNamespace(report_language="zh"))

        prompt = analyzer._build_review_prompt(self._overview(), [])

        self.assertIn("AI选股雷达", prompt)
        self.assertIn("每日热门股", prompt)
        self.assertIn("有潜力的股票", prompt)
        self.assertIn("可抄底观察信号", prompt)
        self.assertIn("只分析 A 股", prompt)
        self.assertIn("长鑫科技", prompt)
        self.assertIn("中际旭创", prompt)
        self.assertNotIn("比亚迪", prompt)
        self.assertNotIn("只分析美股", prompt)
        self.assertNotIn("Nvidia(NVDA)", prompt)

    def test_template_fallback_contains_ai_stock_selection_radar(self) -> None:
        analyzer = MarketAnalyzer(region="cn", config=SimpleNamespace(report_language="zh"))

        report = analyzer._generate_template_review(self._overview(), [])

        self.assertIn("### AI选股雷达", report)
        self.assertIn("#### 每日热门股", report)
        self.assertIn("#### 有潜力的股票", report)
        self.assertIn("#### 可抄底观察信号", report)


if __name__ == "__main__":
    unittest.main()
