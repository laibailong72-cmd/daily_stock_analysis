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
            limit_up_pool=[
                {
                    "code": "603986",
                    "name": "兆易创新",
                    "industry": "半导体",
                    "limit_stat": "1/1",
                    "consecutive_boards": 1,
                    "first_limit_time": "093501",
                    "last_limit_time": "145600",
                    "break_count": 1,
                    "turnover_rate": 8.6,
                    "seal_amount": 260000000.0,
                    "amount": 1800000000.0,
                }
            ],
        )

    def test_us_prompt_has_separate_ai_stock_selection_radar(self) -> None:
        analyzer = MarketAnalyzer(region="us", config=SimpleNamespace(report_language="zh"))

        prompt = analyzer._build_review_prompt(self._overview(), [])

        self.assertIn("AI选股雷达", prompt)
        self.assertIn("美股早报", prompt)
        self.assertIn("上一交易日", prompt)
        self.assertIn("30秒行动卡", prompt)
        self.assertIn("第一屏", prompt)
        self.assertIn("明日执行清单", prompt)
        self.assertIn("市场状态 -> 明日动作 -> 候选池 -> 风险否决", prompt)
        self.assertIn("为什么现在关注", prompt)
        self.assertIn("次日验证条件", prompt)
        self.assertIn("无有效抄底信号", prompt)
        self.assertIn("禁止编造具体股价", prompt)
        self.assertIn("每日热门股", prompt)
        self.assertIn("有潜力的股票", prompt)
        self.assertIn("可抄底观察信号", prompt)
        self.assertIn("只分析美股", prompt)
        self.assertIn("Nvidia(NVDA)", prompt)
        self.assertIn("多篮子", prompt)
        self.assertIn("服务器/电力/网络", prompt)
        self.assertIn("软件/AI应用", prompt)
        self.assertIn("消费/金融/防守", prompt)
        self.assertIn("至少覆盖 3 个不同篮子", prompt)
        self.assertIn("Dell(DELL)", prompt)
        self.assertIn("Vertiv(VRT)", prompt)
        self.assertIn("ServiceNow(NOW)", prompt)
        self.assertIn("Eli Lilly(LLY)", prompt)
        self.assertNotIn("只分析 A 股", prompt)
        self.assertNotIn("中际旭创", prompt)

    def test_cn_prompt_has_separate_ai_stock_selection_radar(self) -> None:
        analyzer = MarketAnalyzer(region="cn", config=SimpleNamespace(report_language="zh"))

        prompt = analyzer._build_review_prompt(self._overview(), [])

        self.assertIn("涨停板全量观察池", prompt)
        self.assertIn("涨停板漏斗筛选", prompt)
        self.assertIn("30秒行动卡", prompt)
        self.assertIn("明日执行清单", prompt)
        self.assertIn("筛选结论表", prompt)
        self.assertIn("留下/剔除/待复核", prompt)
        self.assertIn("证据 -> 判断 -> 动作", prompt)
        self.assertIn("明日无强行买入目标", prompt)
        self.assertIn("无有效抄底信号", prompt)
        self.assertIn("当日临时自选池", prompt)
        self.assertIn("次日重点候选", prompt)
        self.assertIn("兆易创新", prompt)
        self.assertIn("半导体", prompt)
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

        self.assertIn("### 涨停板漏斗筛选", report)
        self.assertIn("次日重点候选", report)
        self.assertIn("### AI选股雷达", report)
        self.assertIn("#### 每日热门股", report)
        self.assertIn("#### 有潜力的股票", report)
        self.assertIn("#### 可抄底观察信号", report)

    def test_payload_includes_limit_up_pool(self) -> None:
        analyzer = MarketAnalyzer(region="cn", config=SimpleNamespace(report_language="zh"))

        payload = analyzer.build_market_review_payload(self._overview(), [], "A股复盘报告")

        self.assertEqual(payload["limit_up_pool"][0]["name"], "兆易创新")


if __name__ == "__main__":
    unittest.main()
