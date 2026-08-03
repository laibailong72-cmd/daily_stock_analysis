# -*- coding: utf-8 -*-
"""
===================================
大盘复盘分析模块
===================================

职责：
1. 获取大盘指数数据（上证、深证、创业板）
2. 搜索市场新闻形成复盘情报
3. 使用大模型生成每日大盘复盘报告
"""

import logging
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from inspect import getattr_static
from typing import Optional, Dict, Any, List

import pandas as pd

from src.config import get_config
from src.report_language import normalize_report_language
from src.search_service import SearchService
from src.core.market_profile import get_profile, MarketProfile
from src.core.market_strategy import get_market_strategy_blueprint
from src.llm.backend_registry import (
    resolve_generation_backend_id,
    resolve_generation_fallback_backend_id,
)
from src.llm.generation_backend import GenerationError
from src.schemas.market_light import MARKET_LIGHT_REGIONS, MarketLightSnapshot
from src.services.run_diagnostics import record_llm_run, record_llm_run_started
from src.services.intelligence_service import IntelligenceService
from data_provider.base import DataFetcherManager

logger = logging.getLogger(__name__)


_ENGLISH_SECTION_PATTERNS = {
    "market_summary": r"###\s*(?:1\.\s*)?Market Summary",
    "index_commentary": r"###\s*(?:2\.\s*)?(?:Index Commentary|Major Indices)",
    "sector_highlights": r"###\s*(?:4\.\s*)?(?:Sector Highlights|Sector/Theme Highlights)",
}

_CHINESE_SECTION_PATTERNS = {
    "market_summary": r"###\s*一、(?:盘面总览|市场总结)",
    "index_commentary": r"###\s*二、(?:指数结构|指数点评|主要指数)",
    "sector_highlights": r"###\s*三、(?:板块主线|热点解读|板块表现)",
    "funds_sentiment": r"###\s*四、(?:资金与情绪|资金动向)",
    "news_catalysts": r"###\s*五、(?:消息催化|后市展望)",
}


_US_AI_STOCK_POOL = (
    "核心权重: Apple(AAPL)、Microsoft(MSFT)、Alphabet(GOOGL/GOOG)、Amazon(AMZN)、"
    "Meta(META)、Tesla(TSLA); "
    "AI算力/芯片: Nvidia(NVDA)、AMD(AMD)、Broadcom(AVGO)、TSMC(TSM)、"
    "Marvell(MRVL)、Micron(MU)、Arm(ARM)、ASML(ASML)、Applied Materials(AMAT)、"
    "Lam Research(LRCX)、KLA(KLAC)、Qualcomm(QCOM); "
    "服务器/电力/网络: Super Micro Computer(SMCI)、Dell(DELL)、Arista(ANET)、"
    "Vertiv(VRT)、Eaton(ETN)、GE Vernova(GEV); "
    "软件/AI应用: Palantir(PLTR)、ServiceNow(NOW)、Oracle(ORCL)、"
    "Salesforce(CRM)、Adobe(ADBE)、Snowflake(SNOW)、CrowdStrike(CRWD)、"
    "Datadog(DDOG); "
    "消费/金融/防守: Netflix(NFLX)、Costco(COST)、Walmart(WMT)、"
    "JPMorgan(JPM)、Visa(V)、Eli Lilly(LLY)、UnitedHealth(UNH)"
)

_CN_AI_STOCK_POOL = (
    "长鑫科技(CXMT/未上市DRAM主线)、中芯国际、兆易创新、北方华创、中微公司、寒武纪、"
    "海光信息、中际旭创、工业富联、新易盛、宁德时代、紫金矿业、中国平安、贵州茅台、科大讯飞"
)


@dataclass
class MarketIndex:
    """大盘指数数据"""
    code: str                    # 指数代码
    name: str                    # 指数名称
    current: float = 0.0         # 当前点位
    change: float = 0.0          # 涨跌点数
    change_pct: float = 0.0      # 涨跌幅(%)
    open: float = 0.0            # 开盘点位
    high: float = 0.0            # 最高点位
    low: float = 0.0             # 最低点位
    prev_close: float = 0.0      # 昨收点位
    volume: float = 0.0          # 成交量（手）
    amount: float = 0.0          # 成交额（元）
    amplitude: float = 0.0       # 振幅(%)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'code': self.code,
            'name': self.name,
            'current': self.current,
            'change': self.change,
            'change_pct': self.change_pct,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'volume': self.volume,
            'amount': self.amount,
            'amplitude': self.amplitude,
        }


@dataclass
class MarketOverview:
    """市场概览数据"""
    date: str                           # 日期
    indices: List[MarketIndex] = field(default_factory=list)  # 主要指数
    up_count: int = 0                   # 上涨家数
    down_count: int = 0                 # 下跌家数
    flat_count: int = 0                 # 平盘家数
    limit_up_count: int = 0             # 涨停家数
    limit_down_count: int = 0           # 跌停家数
    total_amount: float = 0.0           # 两市成交额（亿元）
    # north_flow: float = 0.0           # 北向资金净流入（亿元）- 已废弃，接口不可用
    
    # 板块涨幅榜
    top_sectors: List[Dict] = field(default_factory=list)     # 涨幅前5板块
    bottom_sectors: List[Dict] = field(default_factory=list)  # 跌幅前5板块
    top_concepts: List[Dict] = field(default_factory=list)    # 涨幅前5概念
    bottom_concepts: List[Dict] = field(default_factory=list) # 跌幅前5概念
    limit_up_pool: List[Dict] = field(default_factory=list)   # 当日涨停板全量观察池


@dataclass
class MarketLightReviewResult:
    """Internal market-review parts built from one overview fetch."""

    overview: MarketOverview
    report: str
    market_light_snapshot: Optional[Dict[str, Any]]
    structured_payload: Dict[str, Any] = field(default_factory=dict)


class MarketAnalyzer:
    """
    大盘复盘分析器
    
    功能：
    1. 获取大盘指数实时行情
    2. 获取市场涨跌统计
    3. 获取板块涨跌榜
    4. 搜索市场新闻
    5. 生成大盘复盘报告
    """
    
    def __init__(
        self,
        search_service: Optional[SearchService] = None,
        analyzer=None,
        region: str = "cn",
        config: Optional[Any] = None,
    ):
        """
        初始化大盘分析器

        Args:
            search_service: 搜索服务实例
            analyzer: AI分析器实例（用于调用LLM）
            region: 市场区域 cn=A股 hk=港股 us=美股 jp=日本 kr=韩国
            config: 本次复盘使用的配置；未传时读取全局配置
        """
        self.config = config or get_config()
        self.search_service = search_service
        self.analyzer = analyzer
        self.data_manager = DataFetcherManager()
        self.region = region if region in ("cn", "us", "hk", "jp", "kr") else "cn"
        self.profile: MarketProfile = get_profile(self.region)
        self.strategy = get_market_strategy_blueprint(self.region)

    def _log_context(self) -> str:
        return f"component=market_review region={self.region}"

    def _get_review_language(self) -> str:
        return normalize_report_language(
            getattr(getattr(self, "config", None), "report_language", "zh")
        )

    def _get_template_review_language(self) -> str:
        return normalize_report_language(
            getattr(getattr(self, "config", None), "report_language", "zh")
        )

    def _get_market_scope_name(self, review_language: str | None = None) -> str:
        review_language = review_language or self._get_review_language()
        if self.region == "us":
            return "US market" if review_language == "en" else "美股市场"
        if self.region == "hk":
            return "Hong Kong market" if review_language == "en" else "港股市场"
        if self.region == "jp":
            return "Japan market" if review_language == "en" else "日本市场"
        if self.region == "kr":
            return "Korea market" if review_language == "en" else "韩国市场"
        if review_language == "en":
            return "A-share market"
        return "A股市场"

    def _get_turnover_unit_label(self) -> str:
        """Return the turnover unit label for the current market/language."""
        if self.region == "us":
            return "USD bn" if self._get_review_language() == "en" else "十亿美元"
        if self.region == "hk":
            return "HKD bn" if self._get_review_language() == "en" else "十亿港元"
        if self.region == "jp":
            return "JPY bn" if self._get_review_language() == "en" else "十亿日元"
        if self.region == "kr":
            return "KRW bn" if self._get_review_language() == "en" else "十亿韩元"
        return "CNY 100m" if self._get_review_language() == "en" else "亿"

    def _format_turnover_value(self, amount_raw: float) -> str:
        """Format raw turnover according to market-specific units."""
        if amount_raw == 0.0:
            return "N/A"
        if self.region in ("us", "hk", "jp", "kr"):
            return f"{amount_raw / 1e9:.2f}"
        if amount_raw > 1e6:
            return f"{amount_raw / 1e8:.0f}"
        return f"{amount_raw:.0f}"

    def _get_index_change_arrow(self, change_pct: float) -> str:
        if change_pct == 0:
            return "⚪"
        color_scheme = getattr(getattr(self, "config", None), "market_review_color_scheme", "green_up")
        if color_scheme == "red_up":
            return "🔴" if change_pct > 0 else "🟢"
        return "🟢" if change_pct > 0 else "🔴"

    def _get_review_title(self, date: str) -> str:
        if self._get_review_language() == "en":
            market_names = {
                "us": "US Market Recap",
                "hk": "HK Market Recap",
                "jp": "Japan Market Recap",
                "kr": "Korea Market Recap",
            }
            market_name = market_names.get(self.region, "A-share Market Recap")
            return f"## {date} {market_name}"
        return f"## {date} 大盘复盘"

    def _get_index_hint(self) -> str:
        if self._get_review_language() == "en":
            if self.region == "us":
                return "Analyze the key moves in the S&P 500, Nasdaq, Dow, and other major indices."
            if self.region == "hk":
                return "Analyze the key moves in the HSI, Hang Seng Tech, HSCEI, and other major indices."
            if self.region == "jp":
                return "Analyze the key moves in the Nikkei 225, TOPIX, and other major Japanese indices."
            if self.region == "kr":
                return "Analyze the key moves in the KOSPI, KOSDAQ, and other major Korean indices."
            return "Analyze the price action in the SSE, SZSE, ChiNext, and other major indices."
        return self.profile.prompt_index_hint

    def _get_strategy_prompt_block(self) -> str:
        if self.region == "hk" and self._get_review_language() == "en":
            return """## Strategy Blueprint: Hong Kong Market Regime Strategy
Focus on HSI trend, southbound flow dynamics, and sector rotation to define next-session risk posture.

### Strategy Principles
- Read market regime from HSI, HSTECH, and HSCEI alignment first.
- Track southbound capital flow as a key sentiment driver.
- Translate recap into actionable risk-on/risk-off stance with clear invalidation points.

### Analysis Dimensions
- Trend Regime: Classify the market as momentum, range, or risk-off.
  - Are HSI/HSTECH/HSCEI directionally aligned
  - Did volume confirm the move
  - Are key index levels reclaimed or lost
- Capital Flows: Map southbound flow and macro narrative into equity risk appetite.
  - Southbound net flow direction and magnitude
  - USD/HKD and China policy implications
  - Breadth and leadership concentration
- Sector Themes: Identify persistent leaders and vulnerable laggards.
  - Tech/internet platform trend persistence
  - Financials/property sensitivity to policy shifts
  - Defensive vs growth factor rotation

### Action Framework
- Risk-on: broad index breakout with expanding southbound participation.
- Neutral: mixed index signals; focus on selective relative strength.
- Risk-off: failed breakouts and rising volatility; prioritize capital preservation."""
        if self.region == "jp" and self._get_review_language() == "en":
            return """## Strategy Blueprint: Japan Market Regime Strategy
Focus on Nikkei 225, TOPIX, currency dynamics, and global risk appetite to define the next-session trading plan.

### Strategy Principles
- Read Nikkei 225 and TOPIX alignment first, then assess yen moves, semiconductor/export chains, and financials.
- Translate index conclusions into position sizing, trading pace, and risk-control actions.
- Base judgments only on available index data, news, and price action without inventing breadth or sector statistics.

### Analysis Dimensions
- Trend Regime: Classify Japan equities as advancing, range-bound, or defensive.
  - Are Nikkei 225 and TOPIX directionally aligned
  - Have key index ranges been reclaimed or lost
  - Are large-cap weights and growth chains moving together
- Macro & FX: Map yen, rates, and global risk appetite into equity impact.
  - Yen direction and implications for exporters
  - Bank of Japan and US Treasury yield narratives
  - Overseas technology and semiconductor read-through
- Theme Signals: Identify durable leadership and crowded areas to avoid.
  - Semiconductor, automation, and auto-chain persistence
  - Rotation between financials and domestic-demand stocks
  - Whether news catalysts confirm price action

### Action Framework
- Risk-on: major indices rise together with improving external risk appetite and stronger leadership.
- Neutral: index divergence or FX disruption; avoid chasing and wait for confirmation.
- Risk-off: major indices weaken or external risk rises; prioritize position control."""
        if self.region == "kr" and self._get_review_language() == "en":
            return """## Strategy Blueprint: Korea Market Regime Strategy
Focus on KOSPI, KOSDAQ, semiconductor heavyweights, and global technology risk appetite to define the next-session trading plan.

### Strategy Principles
- Read KOSPI and KOSDAQ alignment first, then assess heavyweight signals from Samsung Electronics, SK Hynix, and related technology leaders.
- Separate broad index beta, semiconductor cycle exposure, and growth-stock risk appetite.
- Base judgments only on available index data, news, and price action without inventing breadth or sector statistics.

### Analysis Dimensions
- Trend Regime: Classify Korea equities as advancing, range-bound, or defensive.
  - Are KOSPI and KOSDAQ directionally aligned
  - Are heavyweight technology names supporting the indices
  - Have key support or resistance levels been reclaimed or lost
- Technology Cycle: Map semiconductor, AI hardware, and global technology moves into Korea equity risk.
  - Memory and semiconductor-chain catalysts
  - US technology-market read-through
  - Foreign investor risk appetite signals
- Theme Signals: Identify durable leadership and crowded areas to avoid.
  - Rotation across batteries, autos, and internet platforms
  - KOSDAQ growth-stock risk appetite
  - Whether news catalysts confirm price action

### Action Framework
- Risk-on: KOSPI and KOSDAQ rise together with confirmed technology leadership and improving external risk appetite.
- Neutral: index or heavyweight divergence; keep sizing controlled and wait for confirmation.
- Risk-off: technology heavyweights weaken or external risk rises; prioritize drawdown control."""
        if self.region == "us" and self._get_review_language() == "zh":
            return """## 美股市场三段式复盘策略
聚焦指数趋势、宏观叙事与板块轮动，给出次日风控与仓位框架。

### 策略原则
- 先看标普500、纳斯达克、道琼斯是否同向，确认主线是否一致。
- 结合宏观与流动性指标，识别风险偏好是修复还是转弱。
- 将复盘输出映射为“进攻/均衡/防守”动作建议，并给出明确触发失效条件。

### 分析维度
- 趋势结构：明确市场处于上冲、震荡还是防守转向，判断是否存在关键支撑位背离。
- 资金与情绪：区分宏观政策、货币面与波动率对权益风险的影响。
- 主题线索：识别持续性最强的主题与板块轮动是否形成可交易主线。

### 行动框架
- 进攻：主板块联动上行且量能/风险位同步改善。
- 均衡：指数分化或量能未明显放大，仓位保守执行。
- 防守：突破失守且波动率抬升时，优先减码并保留反弹可交易性。"""
        if not (self.region == "cn" and self._get_review_language() == "en"):
            return self.strategy.to_prompt_block()
        return """## Strategy Blueprint: A-share Three-Phase Recap Strategy
Focus on index trend, liquidity, and sector rotation to shape the next-session trading plan.

### Strategy Principles
- Read index direction first, then confirm liquidity structure, and finally test sector persistence.
- Every conclusion must map to position sizing, trading pace, and risk-control actions.
- Base judgments on today's data and the latest 3-day news flow without inventing unverified information.

### Analysis Dimensions
- Trend Structure: Determine whether the market is in an uptrend, range, or defensive phase.
  - Are the SSE, SZSE, and ChiNext moving in the same direction
  - Is the market advancing on expanding volume or slipping on contracting volume
  - Have key support or resistance levels been reclaimed or broken
- Liquidity & Sentiment: Identify near-term risk appetite and market temperature.
  - Advance/decline breadth and limit-up/limit-down structure
  - Whether turnover is expanding or fading
  - Whether high-beta leaders are showing divergence
- Leading Themes: Distill tradable leadership and areas to avoid.
  - Whether leading sectors have clear event catalysts
  - Whether sector leaders are pulling the group higher
  - Whether weakness is broadening across lagging sectors

### Action Framework
- Offensive: indices rise in sync, turnover expands, and core themes strengthen.
- Balanced: index divergence or low-volume consolidation; keep sizing controlled and wait for confirmation.
- Defensive: indices weaken and laggards broaden; prioritize risk control and de-risking."""

    def _get_strategy_markdown_block(self, review_language: str | None = None) -> str:
        review_language = review_language or self._get_review_language()
        if self.region == "hk" and review_language == "en":
            return """### 6. Strategy Framework
- **Trend Regime**: Classify the market as momentum, range, or risk-off based on HSI/HSTECH/HSCEI alignment.
- **Capital Flows**: Track southbound flow direction and macro narrative for risk appetite signals.
- **Sector Themes**: Focus on tech/internet platform persistence and financials/property policy sensitivity.
"""
        if self.region == "jp" and review_language == "en":
            return """### 6. Strategy Framework
- **Trend Regime**: Classify Japan equities as advancing, range-bound, or defensive based on Nikkei 225/TOPIX alignment.
- **Macro & FX**: Track yen, rates, and global risk appetite for exporter and financial-sector implications.
- **Theme Signals**: Focus on semiconductor, automation, auto-chain, financial, and domestic-demand rotation.
"""
        if self.region == "kr" and review_language == "en":
            return """### 6. Strategy Framework
- **Trend Regime**: Classify Korea equities as advancing, range-bound, or defensive based on KOSPI/KOSDAQ alignment.
- **Technology Cycle**: Track semiconductor, AI hardware, and global technology read-through for market risk appetite.
- **Theme Signals**: Focus on battery, auto, internet-platform, and KOSDAQ growth-stock rotation.
"""
        if self.region == "us" and review_language == "zh":
            return """### 六、策略框架
- **趋势结构**：判断市场在进攻、震荡与防守中的状态是否一致。
- **资金与情绪**：结合波动率、宽度和主题轮动评估风险偏好。
- **主题主线**：识别可延续和可放大的行业主线与防守线索。
"""
        if not (self.region == "cn" and review_language == "en"):
            return self.strategy.to_markdown_block()
        return """### 6. Strategy Framework
- **Trend Structure**: Determine whether the market is in an uptrend, range, or defensive phase.
- **Liquidity & Sentiment**: Track breadth, turnover expansion, and whether leaders are diverging.
- **Leading Themes**: Focus on sectors with catalysts and sustained leadership while avoiding broadening weakness.
"""

    def _get_market_mood_text(self, mood_key: str, review_language: str | None = None) -> str:
        review_language = review_language or self._get_review_language()
        if review_language == "en":
            mapping = {
                "strong_up": "strong gains",
                "mild_up": "moderate gains",
                "mild_down": "mild losses",
                "strong_down": "clear weakness",
                "range": "range-bound trading",
            }
        else:
            mapping = {
                "strong_up": "强势上涨",
                "mild_up": "小幅上涨",
                "mild_down": "小幅下跌",
                "strong_down": "明显下跌",
                "range": "震荡整理",
            }
        return mapping[mood_key]

    def get_market_overview(self) -> MarketOverview:
        """
        获取市场概览数据
        
        Returns:
            MarketOverview: 市场概览数据对象
        """
        today = datetime.now().strftime('%Y-%m-%d')
        overview = MarketOverview(date=today)
        
        # 1. 获取主要指数行情（按 region 切换 A 股/美股）
        overview.indices = self._get_main_indices()

        # 2. 获取涨跌统计（A 股有，美股无等效数据）
        if self.profile.has_market_stats:
            self._get_market_statistics(overview)

        # 3. 获取板块涨跌榜（A 股有，美股暂无）
        if self.profile.has_sector_rankings:
            self._get_sector_rankings(overview)
            self._get_concept_rankings(overview)

        # 4. A 股涨停板全量观察池：仅作为当日报告筛选输入，不写回固定 STOCK_LIST。
        if self.region == "cn":
            self._get_limit_up_pool(overview)
        
        # 5. 获取北向资金（可选）
        # self._get_north_flow(overview)
        
        return overview

    
    def _get_main_indices(self) -> List[MarketIndex]:
        """获取主要指数实时行情"""
        indices = []

        try:
            logger.info("[大盘] %s action=get_main_indices status=start", self._log_context())

            # 使用 DataFetcherManager 获取指数行情（按 region 切换）
            data_list = self.data_manager.get_main_indices(region=self.region)

            if data_list:
                for item in data_list:
                    index = MarketIndex(
                        code=item['code'],
                        name=item['name'],
                        current=item['current'],
                        change=item['change'],
                        change_pct=item['change_pct'],
                        open=item['open'],
                        high=item['high'],
                        low=item['low'],
                        prev_close=item['prev_close'],
                        volume=item['volume'],
                        amount=item['amount'],
                        amplitude=item['amplitude']
                    )
                    indices.append(index)

            if not indices:
                logger.warning("[大盘] %s action=get_main_indices status=empty", self._log_context())
            else:
                logger.info(
                    "[大盘] %s action=get_main_indices status=success count=%d",
                    self._log_context(),
                    len(indices),
                )

        except Exception as e:
            logger.error("[大盘] %s action=get_main_indices status=failed error=%s", self._log_context(), e)

        return indices

    def _get_market_statistics(self, overview: MarketOverview):
        """获取市场涨跌统计"""
        try:
            logger.info("[大盘] %s action=get_market_stats status=start", self._log_context())

            stats = self.data_manager.get_market_stats(purpose=f"market_review:{self.region}")

            if stats:
                overview.up_count = stats.get('up_count', 0)
                overview.down_count = stats.get('down_count', 0)
                overview.flat_count = stats.get('flat_count', 0)
                overview.limit_up_count = stats.get('limit_up_count', 0)
                overview.limit_down_count = stats.get('limit_down_count', 0)
                overview.total_amount = stats.get('total_amount', 0.0)

                logger.info(
                    "[大盘] %s action=get_market_stats status=success up=%s down=%s flat=%s "
                    "limit_up=%s limit_down=%s amount=%.0f亿",
                    self._log_context(),
                    overview.up_count,
                    overview.down_count,
                    overview.flat_count,
                    overview.limit_up_count,
                    overview.limit_down_count,
                    overview.total_amount,
                )
            else:
                logger.warning("[大盘] %s action=get_market_stats status=empty", self._log_context())

        except Exception as e:
            logger.error("[大盘] %s action=get_market_stats status=failed error=%s", self._log_context(), e)

    def _get_sector_rankings(self, overview: MarketOverview):
        """获取板块涨跌榜"""
        try:
            logger.info("[大盘] %s action=get_sector_rankings status=start", self._log_context())

            top_sectors, bottom_sectors = self.data_manager.get_sector_rankings(5)

            if top_sectors or bottom_sectors:
                overview.top_sectors = top_sectors
                overview.bottom_sectors = bottom_sectors

                logger.info(
                    "[大盘] %s action=get_sector_rankings status=success top=%s bottom=%s",
                    self._log_context(),
                    [s['name'] for s in overview.top_sectors],
                    [s['name'] for s in overview.bottom_sectors],
                )
            else:
                logger.warning("[大盘] %s action=get_sector_rankings status=empty", self._log_context())

        except Exception as e:
            logger.error("[大盘] %s action=get_sector_rankings status=failed error=%s", self._log_context(), e)

    def _get_concept_rankings(self, overview: MarketOverview):
        """获取概念/题材涨跌榜（fail-open）。"""
        try:
            logger.info("[大盘] %s action=get_concept_rankings status=start", self._log_context())

            top_concepts, bottom_concepts = self.data_manager.get_concept_rankings(5)

            if top_concepts or bottom_concepts:
                overview.top_concepts = top_concepts
                overview.bottom_concepts = bottom_concepts

                logger.info(
                    "[大盘] %s action=get_concept_rankings status=success top=%s bottom=%s",
                    self._log_context(),
                    [s.get('name') for s in overview.top_concepts],
                    [s.get('name') for s in overview.bottom_concepts],
                )
            else:
                logger.warning("[大盘] %s action=get_concept_rankings status=empty", self._log_context())

        except Exception as e:
            logger.warning("[大盘] %s action=get_concept_rankings status=failed error=%s", self._log_context(), e)

    def _get_limit_up_pool(self, overview: MarketOverview) -> None:
        """获取当日 A 股涨停池，作为盘后漏斗筛选输入。"""
        try:
            logger.info("[大盘] %s action=get_limit_up_pool status=start", self._log_context())
            pool = self.data_manager.get_limit_up_pool(n=120)
            overview.limit_up_pool = list(pool or [])
            logger.info(
                "[大盘] %s action=get_limit_up_pool status=success count=%d",
                self._log_context(),
                len(overview.limit_up_pool),
            )
        except Exception as e:
            overview.limit_up_pool = []
            logger.warning("[大盘] %s action=get_limit_up_pool status=failed error=%s", self._log_context(), e)
    
    # def _get_north_flow(self, overview: MarketOverview):
    #     """获取北向资金流入"""
    #     try:
    #         logger.info("[大盘] 获取北向资金...")
    #         
    #         # 获取北向资金数据
    #         df = ak.stock_hsgt_north_net_flow_in_em(symbol="北上")
    #         
    #         if df is not None and not df.empty:
    #             # 取最新一条数据
    #             latest = df.iloc[-1]
    #             if '当日净流入' in df.columns:
    #                 overview.north_flow = float(latest['当日净流入']) / 1e8  # 转为亿元
    #             elif '净流入' in df.columns:
    #                 overview.north_flow = float(latest['净流入']) / 1e8
    #                 
    #             logger.info(f"[大盘] 北向资金净流入: {overview.north_flow:.2f}亿")
    #             
    #     except Exception as e:
    #         logger.warning(f"[大盘] 获取北向资金失败: {e}")
    
    def search_market_news(self) -> List[Dict]:
        """
        搜索市场新闻
        
        Returns:
            新闻列表
        """
        if not self.search_service:
            logger.warning(
                "[大盘] %s action=search_market_news status=skipped reason=no_search_service",
                self._log_context(),
            )
            return []
        
        all_news = []

        # 按 region 使用不同的新闻搜索词
        search_queries = self.profile.news_queries
        review_language = self._get_review_language()
        market_names = {
            "cn": "大盘" if review_language == "zh" else "A-share market",
            "us": "美股市场" if review_language == "zh" else "US market",
            "hk": "港股市场" if review_language == "zh" else "HK market",
            "jp": "日本股市" if review_language == "zh" else "Japan stock market",
            "kr": "韩国股市" if review_language == "zh" else "Korea stock market",
        }
        
        try:
            logger.info("[大盘] %s action=search_market_news status=start", self._log_context())
            
            # 根据 region 设置搜索上下文名称，避免美股搜索被解读为 A 股语境
            market_name = market_names.get(self.region, "大盘")
            for query in search_queries:
                response = self.search_service.search_stock_news(
                    stock_code="market",
                    stock_name=market_name,
                    max_results=3,
                    focus_keywords=query.split()
                )
                if response and response.results:
                    all_news.extend(response.results)
                    logger.info(
                        "[大盘] %s action=search_market_news status=query_success count=%d",
                        self._log_context(),
                        len(response.results),
                    )
            
            logger.info(
                "[大盘] %s action=search_market_news status=success count=%d",
                self._log_context(),
                len(all_news),
            )
            
        except Exception as e:
            logger.error("[大盘] %s action=search_market_news status=failed error=%s", self._log_context(), e)
        
        return all_news
    
    def generate_market_review(self, overview: MarketOverview, news: List) -> str:
        """
        使用大模型生成大盘复盘报告
        
        Args:
            overview: 市场概览数据
            news: 市场新闻列表 (SearchResult 对象列表)
            
        Returns:
            大盘复盘报告文本
        """
        backend_error = self._get_analyzer_generation_backend_config_error()
        if backend_error is not None:
            logger.error(
                "[大盘] %s action=generate_review status=failed error_type=%s error=%s",
                self._log_context(),
                type(backend_error).__name__,
                backend_error,
            )
            record_llm_run(
                success=False,
                provider="litellm",
                model=getattr(self.config, "litellm_model", None),
                call_type="market_review",
                error_type=type(backend_error).__name__,
                error_message=backend_error,
            )
            raise backend_error

        if not self.analyzer or not self.analyzer.is_available():
            logger.warning(
                "[大盘] %s action=generate_review status=fallback_template reason=no_analyzer",
                self._log_context(),
            )
            return self._generate_template_review(overview, news)

        # 构建 Prompt
        prompt = self._build_review_prompt(overview, news)

        logger.info("[大盘] %s action=generate_review status=start", self._log_context())
        # Use the public generate_text() entry point - never access private analyzer attributes.
        llm_started_at = time.perf_counter()
        try:
            record_llm_run_started(
                provider="litellm",
                model=getattr(self.config, "litellm_model", None),
                call_type="market_review",
            )
            review = self.analyzer.generate_text(prompt, max_tokens=8192, temperature=0.7)
        except Exception as exc:
            record_llm_run(
                success=False,
                provider="litellm",
                model=getattr(self.config, "litellm_model", None),
                call_type="market_review",
                duration_ms=int((time.perf_counter() - llm_started_at) * 1000),
                error_type=type(exc).__name__,
                error_message=exc,
            )
            raise

        record_llm_run(
            success=bool(review),
            provider="litellm",
            model=getattr(self.config, "litellm_model", None),
            call_type="market_review",
            duration_ms=int((time.perf_counter() - llm_started_at) * 1000),
            error_type=None if review else "EmptyResponse",
            error_message=None if review else "empty market review response",
        )

        if review:
            logger.info(
                "[大盘] %s action=generate_review status=success length=%d",
                self._log_context(),
                len(review),
            )
            # Inject structured data tables into LLM prose sections
            return self._inject_data_into_review(review, overview, news)

        logger.warning(
            "[大盘] %s action=generate_review status=fallback_template reason=empty_llm_response",
            self._log_context(),
        )
        return self._generate_template_review(overview, news)

    def _get_analyzer_generation_backend_config_error(self) -> Optional[GenerationError]:
        """Return analyzer backend config errors without relying on dynamic mock attributes."""
        if self.analyzer is None:
            try:
                resolve_generation_backend_id(self.config)
                resolve_generation_fallback_backend_id(self.config)
            except GenerationError as exc:
                return exc
            return None
        missing = object()
        if getattr_static(self.analyzer, "get_generation_backend_config_error", missing) is missing:
            return None
        method = getattr(self.analyzer, "get_generation_backend_config_error", None)
        if not callable(method):
            return None
        error = method()
        return error if isinstance(error, GenerationError) else None

    def build_market_review_payload(
        self,
        overview: MarketOverview,
        news: List,
        report: str,
        market_light_snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build the structured market-review contract consumed by API, Web, and notifications."""
        language = self._get_review_language()
        sections = self._split_report_sections(report)
        title = self._extract_report_title(report) or self._get_review_title(overview.date).lstrip("# ").strip()
        light = (
            market_light_snapshot or self.build_market_light_snapshot(overview)
            if self._supports_market_light()
            else None
        )
        breadth_dimensions = None
        if isinstance(light, dict):
            dimensions = light.get("dimensions")
            if isinstance(dimensions, dict):
                breadth_dimensions = dimensions.get("breadth")

        breadth_supported = bool(self.profile.has_market_stats)
        if breadth_supported and isinstance(breadth_dimensions, dict) and "available" in breadth_dimensions:
            breadth_supported = bool(breadth_dimensions.get("available"))

        has_breadth_data = False
        if breadth_supported:
            if isinstance(breadth_dimensions, dict) and "available" in breadth_dimensions:
                has_breadth_data = bool(breadth_dimensions.get("available"))
            else:
                breadth_available = overview.up_count + overview.down_count + overview.flat_count > 0
                limit_available = overview.limit_up_count + overview.limit_down_count > 0
                has_breadth_data = bool(breadth_available or limit_available)

        payload = {
            "version": 1,
            "kind": "market_review",
            "region": self.region,
            "language": language,
            "title": title,
            "generated_at": datetime.now().isoformat(),
            "date": overview.date,
            "market_scope": self._get_market_scope_name(language),
            "indices": [idx.to_dict() for idx in overview.indices],
            "sectors": {
                "top": list(overview.top_sectors or []),
                "bottom": list(overview.bottom_sectors or []),
            },
            "concepts": {
                "top": list(overview.top_concepts or []),
                "bottom": list(overview.bottom_concepts or []),
            },
            "limit_up_pool": list(overview.limit_up_pool or []),
            "news": [self._normalize_news_item(item) for item in (news or [])[:8]],
            "sections": sections,
            "markdown_report": report,
        }

        if light is not None:
            payload["market_light"] = light

        if has_breadth_data:
            payload["breadth"] = {
                "up_count": overview.up_count,
                "down_count": overview.down_count,
                "flat_count": overview.flat_count,
                "limit_up_count": overview.limit_up_count,
                "limit_down_count": overview.limit_down_count,
                "total_amount": overview.total_amount,
                "turnover_unit": self._get_turnover_unit_label(),
            }

        return payload

    def _supports_market_light(self) -> bool:
        return self.region in MARKET_LIGHT_REGIONS

    @staticmethod
    def _extract_report_title(report: str) -> str:
        for line in (report or "").splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip()
        return ""

    @classmethod
    def _split_report_sections(cls, report: str) -> List[Dict[str, str]]:
        text = (report or "").strip()
        if not text:
            return []
        matches = list(re.finditer(r"^(#{2,3})\s+(.+?)\s*$", text, flags=re.MULTILINE))
        if not matches:
            return [{"key": "full_review", "title": "Review", "markdown": text}]

        sections: List[Dict[str, str]] = []
        first_match = matches[0]
        starts_with_report_title = first_match.start() == 0 and first_match.group(1) == "##"
        content_start_index = 1 if starts_with_report_title else 0
        intro_start = first_match.end() if starts_with_report_title else 0
        intro_end = (
            matches[1].start()
            if starts_with_report_title and len(matches) > 1
            else (len(text) if starts_with_report_title else matches[0].start())
        )
        intro = text[intro_start:intro_end].strip()
        if intro:
            sections.append({"key": "overview", "title": "Overview", "markdown": intro})

        for index, match in enumerate(matches[content_start_index:], start=content_start_index):
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            title = match.group(2).strip()
            markdown = text[start:end].strip()
            if not markdown:
                continue
            key = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "_", title).strip("_").lower()
            sections.append({
                "key": key or f"section_{index + 1}",
                "title": title,
                "markdown": markdown,
            })
        return sections

    @classmethod
    def _normalize_news_item(cls, item: Any) -> Dict[str, str]:
        return {
            "title": cls._compact_news_text(cls._get_news_field(item, "title"), limit=120),
            "snippet": cls._compact_news_text(cls._get_news_field(item, "snippet"), limit=260),
            "source": cls._compact_news_text(cls._get_news_field(item, "source"), limit=80),
            "published_date": cls._compact_news_text(cls._get_news_field(item, "published_date"), limit=40),
            "url": cls._compact_news_text(cls._get_news_field(item, "url"), limit=240),
        }
    
    def _inject_data_into_review(
        self,
        review: str,
        overview: MarketOverview,
        news: Optional[List] = None,
    ) -> str:
        """Inject structured data tables into the corresponding LLM prose sections."""
        # Build data blocks
        stats_block = self._build_stats_block(overview)
        indices_block = self._build_indices_block(overview)
        sector_block = self._build_sector_block(overview)
        patterns = (
            _ENGLISH_SECTION_PATTERNS
            if self._get_review_language() == "en"
            else _CHINESE_SECTION_PATTERNS
        )

        if stats_block:
            review = self._insert_after_section(
                review,
                patterns["market_summary"],
                stats_block,
            )

        if indices_block:
            review = self._insert_after_section(
                review,
                patterns["index_commentary"],
                indices_block,
            )

        if sector_block:
            review = self._insert_after_section(
                review,
                patterns["sector_highlights"],
                sector_block,
            )

        return review

    @staticmethod
    def _insert_after_section(text: str, heading_pattern: str, block: str) -> str:
        """Insert a data block at the end of a markdown section (before the next ### heading)."""
        import re
        # Find the heading
        match = re.search(heading_pattern, text)
        if not match:
            return text
        start = match.end()
        # Find the next ### heading after this one
        next_heading = re.search(r'\n###\s', text[start:])
        if next_heading:
            insert_pos = start + next_heading.start()
        else:
            # No next heading — append at end
            insert_pos = len(text)
        # Insert the block before the next heading, with spacing
        return text[:insert_pos].rstrip() + '\n\n' + block + '\n\n' + text[insert_pos:].lstrip('\n')

    def _build_stats_block(self, overview: MarketOverview) -> str:
        """Build market statistics block."""
        has_stats = overview.up_count or overview.down_count or overview.total_amount
        if not has_stats:
            return ""
        if self._get_review_language() == "en":
            light = self.build_market_light_snapshot(overview)
            return "\n".join(
                [
                    f"- **Market Signal**: {light['score']}/100 "
                    f"({light['temperature_label']}, {light['label']})",
                    f"- **Drivers**: {'; '.join(light['reasons'])}",
                    f"- **Guidance**: {light['guidance']}",
                    "",
                    f"- **Breadth**: Advancers {overview.up_count} / Decliners {overview.down_count} / "
                    f"Flat {overview.flat_count}; "
                    f"Limit-up {overview.limit_up_count} / Limit-down {overview.limit_down_count}; "
                    f"Turnover {overview.total_amount:.0f} ({self._get_turnover_unit_label()})",
                ]
            )
        light = self.build_market_light_snapshot(overview)
        score, label = light["score"], light["temperature_label"]
        participation = overview.up_count + overview.down_count
        up_ratio = overview.up_count / participation if participation else 0.0
        limit_spread = overview.limit_up_count - overview.limit_down_count
        lines = [
            f"- **盘面信号**：{score}/100（{label}，{light['label']}）",
            f"- **信号依据**：{'；'.join(light['reasons'])}",
            f"- **操作建议**：{light['guidance']}",
            "",
            "| 指标 | 数值 | 观察 |",
            "|------|------|------|",
            f"| 上涨/下跌/平盘 | {overview.up_count} / {overview.down_count} / {overview.flat_count} | 上涨占比(不含平盘) {up_ratio:.1%} |",
            f"| 涨停/跌停 | {overview.limit_up_count} / {overview.limit_down_count} | 涨跌停差 {limit_spread:+d} |",
            f"| 两市成交额 | {overview.total_amount:.0f} 亿 | {self._describe_turnover(overview.total_amount)} |",
        ]
        return "\n".join(lines)

    def build_market_light_snapshot(self, overview: MarketOverview) -> Dict[str, Any]:
        """Build a deterministic market-light snapshot from structured breadth data."""
        scores = self._build_market_light_scores(overview)
        score = int(scores["score"])
        temperature_label = str(scores["temperature_label"])
        if score >= 60:
            status = "green"
        elif score >= 40:
            status = "yellow"
        else:
            status = "red"

        if self._get_review_language() == "en":
            label_map = {
                "green": "risk-on",
                "yellow": "balanced",
                "red": "risk-off",
            }
            guidance_map = {
                "green": "Risk appetite is acceptable; focus on leading themes and position discipline.",
                "yellow": "Signals are mixed; keep position sizing moderate and wait for confirmation.",
                "red": "Risk is elevated; prioritize drawdown control and avoid chasing weak rebounds.",
            }
            reasons = self._build_market_light_reasons_en(overview, score)
        else:
            label_map = {
                "green": "可进攻",
                "yellow": "需观察",
                "red": "偏防守",
            }
            guidance_map = {
                "green": "风险偏好尚可，关注主线延续与仓位纪律。",
                "yellow": "信号分化，控制仓位并等待量价确认。",
                "red": "风险偏高，优先控制回撤，避免追高弱反弹。",
            }
            reasons = self._build_market_light_reasons_zh(overview, score)

        snapshot = MarketLightSnapshot(
            region=self.region,
            trade_date=overview.date,
            status=status,
            label=label_map[status],
            score=score,
            temperature_label=temperature_label,
            reasons=reasons,
            guidance=guidance_map[status],
            dimensions=scores["dimensions"],
            data_quality=str(scores["data_quality"]),
        )
        return snapshot.model_dump()

    def _build_market_light_reasons_zh(self, overview: MarketOverview, score: int) -> List[str]:
        participation = overview.up_count + overview.down_count
        up_ratio = overview.up_count / participation if participation else None
        reasons: List[str] = []
        if up_ratio is not None:
            if up_ratio >= 0.6:
                reasons.append(f"上涨家数占比 {up_ratio:.0%}，赚钱效应扩散")
            elif up_ratio <= 0.4:
                reasons.append(f"上涨家数占比 {up_ratio:.0%}，亏钱效应较强")
            else:
                reasons.append(f"上涨家数占比 {up_ratio:.0%}，市场分化")
        index_changes = [idx.change_pct for idx in overview.indices if idx.change_pct is not None]
        if index_changes:
            avg_change = sum(index_changes) / len(index_changes)
            reasons.append(f"主要指数平均涨跌幅 {avg_change:+.2f}%")
        if overview.limit_up_count or overview.limit_down_count:
            reasons.append(f"涨跌停差 {overview.limit_up_count - overview.limit_down_count:+d}")
        if not reasons and overview.total_amount:
            reasons.append(f"成交额 {overview.total_amount:.0f} 亿，{self._describe_turnover(overview.total_amount)}")
        if not reasons:
            reasons.append("结构化涨跌数据有限，按可用行情综合判断")
        return reasons[:4]

    def _build_market_light_reasons_en(self, overview: MarketOverview, score: int) -> List[str]:
        participation = overview.up_count + overview.down_count
        up_ratio = overview.up_count / participation if participation else None
        reasons: List[str] = []
        if up_ratio is not None:
            if up_ratio >= 0.6:
                reasons.append(f"advancers ratio {up_ratio:.0%}, breadth is expanding")
            elif up_ratio <= 0.4:
                reasons.append(f"advancers ratio {up_ratio:.0%}, downside pressure dominates")
            else:
                reasons.append(f"advancers ratio {up_ratio:.0%}, breadth is mixed")
        index_changes = [idx.change_pct for idx in overview.indices if idx.change_pct is not None]
        if index_changes:
            avg_change = sum(index_changes) / len(index_changes)
            reasons.append(f"average major-index change {avg_change:+.2f}%")
        if overview.limit_up_count or overview.limit_down_count:
            reasons.append(f"limit-up/down spread {overview.limit_up_count - overview.limit_down_count:+d}")
        if not reasons and overview.total_amount:
            reasons.append(f"turnover {overview.total_amount:.0f} ({self._get_turnover_unit_label()})")
        if not reasons:
            reasons.append("limited structured breadth data; using available market inputs")
        return reasons[:4]

    def _build_indices_block(self, overview: MarketOverview) -> str:
        """构建指数行情表格"""
        if not overview.indices:
            return ""
        if self._get_review_language() == "en":
            lines = [
                f"| Index | Last | Change % | Open | High | Low | Amplitude | Turnover ({self._get_turnover_unit_label()}) |",
                "|-------|------|----------|------|------|-----|-----------|-----------------|",
            ]
        else:
            lines = [
                "| 指数 | 最新 | 涨跌幅 | 开盘 | 最高 | 最低 | 振幅 | 成交额(亿) |",
                "|------|------|--------|------|------|------|------|-----------|",
            ]
        for idx in overview.indices:
            arrow = self._get_index_change_arrow(idx.change_pct)
            amount_raw = idx.amount or 0.0
            amount_str = self._format_turnover_value(amount_raw)
            lines.append(
                f"| {idx.name} | {idx.current:.2f} | {arrow} {idx.change_pct:+.2f}% | "
                f"{self._format_optional_number(idx.open)} | {self._format_optional_number(idx.high)} | "
                f"{self._format_optional_number(idx.low)} | {self._format_optional_pct(idx.amplitude)} | {amount_str} |"
            )
        return "\n".join(lines)

    def _build_sector_block(self, overview: MarketOverview) -> str:
        """Build industry and concept ranking blocks."""
        if (
            not overview.top_sectors
            and not overview.bottom_sectors
            and not overview.top_concepts
            and not overview.bottom_concepts
        ):
            return ""
        lines = []
        language = self._get_review_language()

        def append_ranking(title: str, name_label: str, rows: List[Dict]) -> None:
            if not rows:
                return
            if lines:
                lines.append("")
            lines.extend([
                title,
                f"| {'Rank' if language == 'en' else '排名'} | {name_label} | {'Change' if language == 'en' else '涨跌幅'} |",
                "|------|------|--------|",
            ])
            for rank, item in enumerate(rows[:5], 1):
                lines.append(
                    f"| {rank} | {item.get('name', '-')} | {self._format_signed_pct(item.get('change_pct'))} |"
                )

        if language == "en":
            append_ranking("#### Leading Industry Sectors", "Sector", overview.top_sectors)
            append_ranking("#### Lagging Industry Sectors", "Sector", overview.bottom_sectors)
            append_ranking("#### Leading Concept Themes", "Concept", overview.top_concepts)
            append_ranking("#### Lagging Concept Themes", "Concept", overview.bottom_concepts)
        else:
            append_ranking("#### 行业板块领涨 Top 5", "行业板块", overview.top_sectors)
            append_ranking("#### 行业板块领跌 Top 5", "行业板块", overview.bottom_sectors)
            append_ranking("#### 概念板块领涨 Top 5", "概念板块", overview.top_concepts)
            append_ranking("#### 概念板块领跌 Top 5", "概念板块", overview.bottom_concepts)
        return "\n".join(lines)

    def _format_limit_up_pool_summary(self, pool: List[Dict], *, limit: int = 80) -> str:
        """Format A-share limit-up pool as compact prompt input."""
        rows = [item for item in (pool or []) if isinstance(item, dict)]
        if not rows:
            return "暂无涨停池数据。"

        industry_counts = Counter(
            str(item.get("industry") or "未分类").strip() or "未分类"
            for item in rows
        )
        top_industries = "、".join(
            f"{name}({count})"
            for name, count in industry_counts.most_common(8)
        )
        lines = [
            f"- 今日涨停池数量：{len(rows)} 只",
            f"- 行业分布 Top：{top_industries or '暂无'}",
            "- 说明：这是当日临时观察池，不会写回固定 STOCK_LIST。",
            "",
            "| 序号 | 代码 | 名称 | 行业 | 连板/统计 | 首封 | 末封 | 炸板 | 换手率 | 封板资金 | 成交额 |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for index, item in enumerate(rows[:limit], 1):
            code = str(item.get("code") or "-").strip() or "-"
            name = str(item.get("name") or "-").strip() or "-"
            industry = str(item.get("industry") or "-").strip() or "-"
            stat = str(item.get("limit_stat") or "").strip()
            boards = item.get("consecutive_boards")
            board_text = stat or (f"{boards}板" if boards not in (None, "", 0) else "-")
            lines.append(
                "| {index} | {code} | {name} | {industry} | {board_text} | {first} | {last} | {breaks} | {turnover} | {seal} | {amount} |".format(
                    index=index,
                    code=code,
                    name=name,
                    industry=industry,
                    board_text=board_text,
                    first=str(item.get("first_limit_time") or "-"),
                    last=str(item.get("last_limit_time") or "-"),
                    breaks=item.get("break_count") if item.get("break_count") is not None else "-",
                    turnover=self._format_optional_pct(item.get("turnover_rate")),
                    seal=self._format_money_value(item.get("seal_amount")),
                    amount=self._format_money_value(item.get("amount")),
                )
            )
        if len(rows) > limit:
            lines.append(f"- 其余 {len(rows) - limit} 只已省略，优先分析上表中连板更高、封板更早的个股。")
        return "\n".join(lines)

    @staticmethod
    def _format_money_value(value: Any) -> str:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return "N/A"
        if numeric_value == 0:
            return "N/A"
        if abs(numeric_value) >= 1e8:
            return f"{numeric_value / 1e8:.2f}亿"
        if abs(numeric_value) >= 1e4:
            return f"{numeric_value / 1e4:.0f}万"
        return f"{numeric_value:.0f}"

    def _build_news_block(self, news: List) -> str:
        """Build a compact source-aware news catalyst list for the rendered report."""
        if not news:
            return ""
        language = self._get_review_language()
        if language == "en":
            lines = [
                "#### News Catalysts",
            ]
        else:
            lines = [
                "#### 近三日市场线索",
            ]

        for idx, item in enumerate(news[:5], 1):
            lines.append(self._format_news_catalyst_line(idx, item, language=language))
        return "\n".join(lines)

    @staticmethod
    def _get_news_field(item: Any, field: str) -> str:
        if hasattr(item, field):
            value = getattr(item, field, "") or ""
        elif isinstance(item, dict):
            value = item.get(field, "") or ""
        else:
            value = ""
        return str(value).strip()

    @classmethod
    def _format_news_catalyst_line(cls, idx: int, item: Any, *, language: str = "zh") -> str:
        fallback_title = "Untitled catalyst" if language == "en" else "未命名线索"
        title = cls._compact_news_text(cls._get_news_field(item, "title"), limit=90) or fallback_title
        source = cls._compact_news_text(cls._get_news_field(item, "source"), limit=40)
        date_text = cls._compact_news_text(cls._get_news_field(item, "published_date"), limit=24)
        url = cls._compact_news_text(cls._get_news_field(item, "url"), limit=0)
        title_text = cls._escape_markdown_link_label(title)
        if url:
            title_text = f"[{title_text}]({url})"
        meta_parts = [part for part in (source, date_text) if part]
        if language == "en":
            meta = f" ({' / '.join(meta_parts)})" if meta_parts else ""
        else:
            meta = f"（{' / '.join(meta_parts)}）" if meta_parts else ""
        return f"- {idx}. {title_text}{meta}"

    @staticmethod
    def _compact_news_text(value: str, *, limit: int) -> str:
        text = " ".join(str(value or "").split())
        if limit <= 0 or len(text) <= limit:
            return text
        return text[: max(0, limit - 3)].rstrip() + "..."

    @staticmethod
    def _format_optional_number(value: float) -> str:
        return "N/A" if value in (None, 0, 0.0) else f"{value:.2f}"

    @staticmethod
    def _format_optional_pct(value: float) -> str:
        return "N/A" if value in (None, 0, 0.0) else f"{value:.2f}%"

    @staticmethod
    def _format_signed_pct(value: Any) -> str:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return "N/A"
        return f"{numeric_value:+.2f}%"

    @classmethod
    def _format_ranking_summary(cls, rows: List[Dict], limit: int = 3) -> str:
        parts = []
        for item in (rows or [])[:limit]:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            parts.append(f"{name}({cls._format_signed_pct(item.get('change_pct'))})")
        return ", ".join(parts)

    @staticmethod
    def _escape_markdown_link_label(value: str) -> str:
        return value.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")

    @staticmethod
    def _describe_turnover(total_amount: float) -> str:
        if total_amount >= 15000:
            return "高活跃度"
        if total_amount >= 9000:
            return "中等活跃"
        if total_amount > 0:
            return "缩量观望"
        return "暂无数据"

    def _build_market_light_scores(self, overview: MarketOverview) -> Dict[str, Any]:
        """Build the canonical Market Light scores used by reports and alerts."""

        participants = overview.up_count + overview.down_count
        breadth_available = bool(self.profile.has_market_stats and participants > 0)
        breadth_score = 50
        if breadth_available:
            breadth_score = int(overview.up_count / participants * 100)

        index_changes = [idx.change_pct for idx in overview.indices if idx.change_pct is not None]
        index_available = bool(overview.indices and index_changes)
        index_score = 50
        if index_available:
            avg_change = sum(index_changes) / len(index_changes)
            index_score = int(max(0, min(100, 50 + avg_change * 12)))

        limit_total = overview.limit_up_count + overview.limit_down_count
        limit_available = bool(self.profile.has_market_stats and limit_total > 0)
        limit_score = 50
        if limit_available:
            limit_score = int(overview.limit_up_count / limit_total * 100)

        dimensions = {
            "breadth": {"score": breadth_score, "available": breadth_available},
            "index": {"score": index_score, "available": index_available},
            "limit": {"score": limit_score, "available": limit_available},
        }

        if not index_available:
            data_quality = "unavailable"
        elif all(dimension["available"] for dimension in dimensions.values()):
            data_quality = "ok"
        else:
            data_quality = "partial"

        score = int(round(breadth_score * 0.45 + index_score * 0.35 + limit_score * 0.20))
        if self._get_review_language() == "en":
            if score >= 70:
                label = "risk-on"
            elif score >= 55:
                label = "constructive"
            elif score >= 40:
                label = "mixed"
            else:
                label = "defensive"
        else:
            if score >= 70:
                label = "强势"
            elif score >= 55:
                label = "偏暖"
            elif score >= 40:
                label = "震荡"
            else:
                label = "偏弱"
        return {
            "score": score,
            "temperature_label": label,
            "dimensions": dimensions,
            "data_quality": data_quality,
        }

    def _build_market_temperature(self, overview: MarketOverview) -> tuple[int, str]:
        scores = self._build_market_light_scores(overview)
        score = int(scores["score"])
        label = str(scores["temperature_label"])
        return score, label

    def _build_output_template_sections(self, review_language: str) -> str:
        """Build LLM output sections according to market data capabilities."""
        if review_language == "en":
            if self.profile.has_market_stats and self.profile.has_sector_rankings:
                return """### 3. Main Themes
(Rank the top 3 themes by strength, sustainability, and risk. For each theme, explain what confirms continuation and what invalidates it.)

### 4. Fund Flows
(Interpret what turnover, participation, and flow signals imply.)

### 5. Sector Highlights
(Distinguish industry-sector moves from concept/theme moves, then analyze drivers and persistence.)

### 6. AI Stock Selection Radar
(For the current market only, provide three separate lists: daily hot stocks, high-potential stocks, and dip-buy/reversal watch signals. Use at most 3 candidates per list. Include theme, observation reason, trigger condition, invalidation condition, and risk note. Do not invent exact stock prices or moving-average values when individual quote data is not supplied.)

### 7. Tomorrow's Watch Plan
(Provide the near-term outlook based on price action and news.)

### 8. Risk Alerts
(List the main risks to monitor.)

### 9. Strategy Plan
(Provide an offensive/balanced/defensive stance, a position-sizing guideline, one invalidation trigger, and end with "For reference only, not investment advice.")"""

            section_number = 3
            sections: List[str] = []
            if self.profile.has_market_stats:
                sections.append(f"""### {section_number}. Fund Flows
(Interpret only the provided turnover, participation, breadth, and flow signals.)""")
                section_number += 1
            if self.profile.has_sector_rankings:
                sections.append(f"""### {section_number}. Sector Highlights
(Analyze only the provided industry-sector and concept/theme rankings.)""")
                section_number += 1
            sections.extend([
                f"""### {section_number}. AI Stock Selection Radar
(For the current market only, provide three separate lists: daily hot stocks, high-potential stocks, and dip-buy/reversal watch signals. Use at most 3 candidates per list. Include theme, observation reason, trigger condition, invalidation condition, and risk note. Do not invent exact stock prices or moving-average values when individual quote data is not supplied.)""",
                f"""### {section_number + 1}. News Catalysts
(Connect recent news to index price action and macro/external-market clues. Do not infer unsupported breadth, fund-flow, or sector-ranking data.)""",
                f"""### {section_number + 2}. Tomorrow's Watch Plan
(Provide the near-term outlook based on index price action and the available news.)""",
                f"""### {section_number + 3}. Risk Alerts
(List the main risks to monitor.)""",
                f"""### {section_number + 4}. Strategy Plan
(Provide an offensive/balanced/defensive stance, a position-sizing guideline, one invalidation trigger, and end with "For reference only, not investment advice.")""",
            ])
            return "\n\n".join(sections)

        if self.region == "us":
            return """### 三、核心趋势与热门权重
（用“上一交易日/昨夜美股”的口径说明核心趋势：大型科技、AI算力、半导体设备、服务器/电力、企业软件、消费/金融/医药防守、利率/VIX/美元等；必须区分“指数权重驱动”“AI硬件驱动”“非科技扩散/防守轮动”；明确哪些只是指数层面的推断，哪些有新闻或数据支持；控制在 6 条以内）

### 四、AI选股雷达
（固定输出一张表，三类：每日热门股、有潜力的股票、可抄底观察信号；每类最多3个；列为“类别/股票/主线/篮子/证据/触发/失效/风险”；候选必须从不同篮子中选，至少覆盖 3 个篮子：核心权重、AI算力/芯片、服务器/电力/网络、软件/AI应用、消费/金融/防守；不得连续只给 Apple/Nvidia/Microsoft/AMD/Broadcom/Meta/Tesla 这类少数热门股；没有个股行情数据时不得写具体价格、跌幅百分比、均线数值，只能写相对条件；若抄底条件不成立，必须写“无有效抄底信号”）

### 五、明日观察计划
（给出进攻/均衡/防守结论；只列 3 条：优先观察、暂时回避、开盘后确认/失效；目标是帮助快速观察市场，不直接给买入指令）

### 六、风险提示
（最多 4 条，列出科技权重、AI高位拥挤、宏观利率、财报/监管等风险；最后补充“建议仅供参考，不构成投资建议”。）"""

        if self.profile.has_market_stats and self.profile.has_sector_rankings:
            return """### 三、板块主线
（区分行业板块与概念题材，分析领涨/领跌背后的逻辑、持续性和是否形成主线；只保留最强 3 条主线和最弱 2 条风险线）

### 四、涨停板漏斗筛选
（仅 A 股使用：不要重复完整涨停池；先给行业集中度和核心题材，再按“涨停逻辑 -> 题材热点 -> 多头趋势/突破平台/资金封板 -> 利空与分时换手风险”逐层剔除；必须先给“筛选结论表”，列出“留下/剔除/待复核”及原因；最后只给 0-2 只“次日重点候选”，候选必须来自“留下”池；必须写触发条件、失效条件、仓位纪律和风险提示；如果没有合格标的，必须写“明日无强行买入目标”）

### 五、AI选股雷达
（固定输出一张表，三类：每日热门股、有潜力的股票、可抄底观察信号；每类最多2个；列为“类别/股票/主线/证据/触发/失效/风险”；优先覆盖能反映国内整体局势的热门方向，例如半导体、AI算力、机器人、新能源、消费电子、高端制造、金融/消费、资源；可抄底必须同时满足“缩量止跌、板块修复、重新站回短期强弱线”的相对条件，否则写“无有效抄底信号”；不得输出无条件买入，不得编造具体股价/均线数值）

### 六、资金与情绪
（最多 4 条，解读成交额、涨跌停结构、市场宽度和风险偏好）

### 七、消息催化
（最多 4 条，结合近三日新闻，提炼真正影响明日交易的催化或扰动；新闻为空时写“暂无已验证消息催化”）

### 八、明日观察计划
（给出进攻/均衡/防守结论；只列 4 条：优先方向、回避方向、触发条件、失效条件；目标是帮助快速理解市场，不直接给买入指令）

### 九、风险提示
（最多 4 条；最后补充“建议仅供参考，不构成投资建议”。）"""

        numerals = ["一", "二", "三", "四", "五", "六", "七", "八"]
        section_number = 3
        sections: List[str] = []

        def add_section(title: str, hint: str) -> None:
            nonlocal section_number
            sections.append(f"### {numerals[section_number - 1]}、{title}\n{hint}")
            section_number += 1

        if self.profile.has_sector_rankings:
            add_section("板块主线", "（仅分析已提供的行业板块与概念题材榜单，不扩展未提供的数据）")
        if self.region == "cn":
            add_section(
                "涨停板漏斗筛选",
                "（把今日涨停板作为临时观察池，按涨停逻辑、题材热点、多头趋势/突破平台、资金封板、利空/分时/换手风险逐层筛选；必须用表格列出“留下/剔除/待复核”及原因；最终只给 1-2 只“次日重点候选”，必须包含触发条件和失效条件，不得写成无条件买入）",
            )
        add_section(
            "AI选股雷达",
            "（当前市场单独输出三类：每日热门股、有潜力的股票、可抄底观察信号；每类最多3个；每个候选必须按“证据/判断/动作”包含所属主线、观察理由、触发条件、失效条件和风险提示；没有个股行情数据时必须标注为“观察池，需量价确认”，不得编造具体股价/均线数值）",
        )
        if self.profile.has_market_stats:
            add_section("资金与情绪", "（仅解读已提供的成交额、涨跌停结构、市场宽度和风险偏好数据）")
        add_section(
            "消息催化",
            "（结合近三日新闻和指数表现，提炼真正影响明日交易的催化或扰动；不要推断未提供的资金流、市场宽度或板块榜）",
        )
        add_section("明日观察计划", "（给出进攻/均衡/防守结论、观察方向、回避方向、触发条件和失效条件；目标是帮助快速理解市场，不直接给买入指令）")
        add_section("风险提示", "（列出需要关注的风险点；最后补充“建议仅供参考，不构成投资建议”。）")
        return "\n\n".join(sections)

    def _get_hot_watchlist_prompt_block(self) -> str:
        """Return market-specific watchlist guidance for market-review prompts."""
        if self._get_review_language() == "en":
            if self.region == "us":
                return f"""## AI Stock Selection Radar Guidance
- Current market scope: US only. Do not mix in A-share candidates.
- Focus themes: mega-cap leadership, AI compute, semiconductors, semiconductor equipment, servers/networking/power, enterprise AI software, consumer/financial/defensive rotation.
- Reference pool for screening, not automatic recommendations: {_US_AI_STOCK_POOL}.
- Use multiple baskets instead of repeating the same mega-cap technology names every day.
- In the "AI Stock Selection Radar" section, output exactly three sublists: "Daily Hot Stocks", "High-Potential Stocks", and "Dip-Buy/Reversal Watch".
- Each candidate must include theme, basket, reason, trigger condition, invalidation condition, and a risk note. Cover at least three different baskets unless the input clearly supports only one theme. If individual-stock quote data was not supplied, mark it as a watchlist idea requiring price/volume confirmation."""
            if self.region == "cn":
                return f"""## AI Stock Selection Radar Guidance
- Current market scope: A-share only. Do not mix in US candidates.
- Focus themes: semiconductors, AI computing, robotics, new energy, consumer electronics, advanced manufacturing, and policy-driven themes.
- Reference pool for screening, not automatic recommendations: {_CN_AI_STOCK_POOL}.
- In the "AI Stock Selection Radar" section, output exactly three sublists: "Daily Hot Stocks", "High-Potential Stocks", and "Dip-Buy/Reversal Watch".
- Each candidate must include theme, reason, trigger condition, invalidation condition, and a risk note. If individual-stock quote data was not supplied, mark it as a watchlist idea requiring price/volume confirmation."""
            return """## Hot Theme Watchlist Guidance
- Name the strongest themes and representative stocks only when supported by the available data or news.
- Mark unsupported individual-stock ideas as a watchlist that requires price confirmation."""

        if self.region == "us":
            return f"""## AI选股雷达指引
- 当前市场范围：只分析美股，不混入 A 股候选。
- 这是北京时间早间美股早报，分析对象是“上一交易日/昨夜美股”，不要写成 A 股盘后复盘。
- 候选参考池仅用于筛选，不代表自动推荐：{_US_AI_STOCK_POOL}。
- 美股 AI 选股不能每天只围绕同一批大科技；必须从多篮子中轮动筛选。
- 篮子定义：
  1. 核心权重：AAPL/MSFT/GOOGL/AMZN/META/TSLA。
  2. AI算力/芯片：NVDA/AMD/AVGO/TSM/MRVL/MU/ARM/ASML/AMAT/LRCX/KLAC/QCOM。
  3. 服务器/电力/网络：SMCI/DELL/ANET/VRT/ETN/GEV。
  4. 软件/AI应用：PLTR/NOW/ORCL/CRM/ADBE/SNOW/CRWD/DDOG。
  5. 消费/金融/防守：NFLX/COST/WMT/JPM/V/LLY/UNH。
- 在“AI选股雷达”章节必须固定分成三类：“每日热门股”“有潜力的股票”“可抄底观察信号”，并用同一张表输出。
- 每类最多 3 只；整张表至少覆盖 3 个不同篮子，除非输入数据明确显示只有单一主线有效。
- 每个候选必须写清：所属主线、所属篮子、为什么现在关注、次日验证条件、失效条件、风险提示。
- 每栏候选按“股票 - 主线 - 篮子 - 证据 - 触发 - 失效 - 风险”压缩输出，优先给能代表指数和热门方向的票，不要堆长篇解释。
- 每次最多只能有 1 只来自“核心权重”篮子进入“有潜力的股票”，避免把潜力股写成大型科技复读。
- “可抄底观察信号”不是每天都必须有；只有出现止跌、主线修复、风险释放三个相对条件时才列出，否则写“无有效抄底信号”。
- 没有个股行情数据时只能作为“观察池，需量价确认”；禁止编造具体股价、跌幅百分比、均线数值或“连续几日”条件。"""
        if self.region == "cn":
            return f"""## AI选股雷达指引
- 当前市场范围：只分析 A 股，不混入美股候选。
- 必须覆盖 A 股半导体、AI/算力、机器人、新能源、消费电子、高端制造、政策催化等主线。
- 候选参考池仅用于筛选，不代表自动推荐：{_CN_AI_STOCK_POOL}。
- 结合已提供的行业/概念榜、指数结构和新闻线索，在“AI选股雷达”章节固定分成三类：“每日热门股”“有潜力的股票”“可抄底观察信号”，并用同一张表输出。
- 每类最多 2 只；每个候选必须写清：所属主线、入选理由、触发条件、失效条件、风险提示；没有个股行情数据时只能作为“观察池，需量价确认”。
- “可抄底观察信号”不是每天都必须有；只有出现缩量止跌、板块修复、重新站回短期强弱线三个相对条件时才列出，否则写“无有效抄底信号”。"""
        return """## 热门股票与主题观察指引
- 结合已提供的指数和新闻线索，列出当前最强主题和代表股票观察池。
- 没有个股行情数据时必须标注为“观察池，需量价确认”，不得输出无条件买入。"""

    def _build_ai_stock_selection_template_block(self, language: str) -> str:
        """Build a conservative fallback AI-stock-selection section for template reviews."""
        if language == "en":
            if self.region == "us":
                return f"""### AI Stock Selection Radar
> Watchlist only. Confirm with price/volume triggers and invalidation discipline; not investment advice.

#### Daily Hot Stocks
- Multi-basket focus: mega-cap leadership, AI compute/chips, semiconductor equipment, servers/networking/power, enterprise AI software, and defensive rotation. Reference pool: {_US_AI_STOCK_POOL}.

#### High-Potential Stocks
- Screen beyond mega-cap technology: include at least one non-mega-cap basket when supported by index/news clues. Prefer relative-strength leaders in AI infrastructure, software adoption, power/networking, or defensive rotation.

#### Dip-Buy/Reversal Watch
- Only watch for names that pull back toward support, stop falling with improving volume, and reclaim intraday resistance. If no setup qualifies, state "No valid dip-buy signal". Invalidation: support breaks with expanding downside volume.
"""
            if self.region == "cn":
                return f"""### AI Stock Selection Radar
> Watchlist only. Confirm with price/volume triggers and invalidation discipline; not investment advice.

#### Daily Hot Stocks
- Semiconductor / AI computing / robotics / new-energy focus: {_CN_AI_STOCK_POOL}. Prioritize names tied to today's strongest sector and concept clues.

#### High-Potential Stocks
- Screen for sector leaders with improving breadth, stronger relative strength than major A-share indices, and policy or earnings catalyst confirmation.

#### Dip-Buy/Reversal Watch
- Only watch for oversold leaders that stabilize near support, show shrinking downside volume, and recover short-term moving averages. Invalidation: support breaks or sector leadership fades.
"""
            return """### AI Stock Selection Radar
> Watchlist only. Confirm with price/volume triggers and invalidation discipline; not investment advice.

#### Daily Hot Stocks
- Track representative stocks in the strongest themes confirmed by today's index and news clues.

#### High-Potential Stocks
- Screen for relative-strength leaders with clear catalysts.

#### Dip-Buy/Reversal Watch
- Only watch pullbacks that stabilize at support and reclaim short-term resistance.
"""

        if self.region == "us":
            return f"""### AI选股雷达
> 仅作观察池：必须等待量价触发和失效条件确认，不构成投资建议。

#### 每日热门股
- 多篮子观察：核心权重、AI算力/芯片、服务器/电力/网络、软件/AI应用、消费/金融/防守。参考池：{_US_AI_STOCK_POOL}。

#### 有潜力的股票
- 不只看大型科技。优先观察相对纳指/标普更强、主线催化延续、且来自不同篮子的候选，例如服务器电力、企业软件、半导体设备或防守轮动。

#### 可抄底观察信号
- 只观察回踩支撑后止跌、缩量调整后重新放量、并收复日内压力位的候选。若无合格形态，写“无有效抄底信号”。失效条件：放量跌破支撑或主线退潮。
"""
        if self.region == "cn":
            return f"""### AI选股雷达
> 仅作观察池：必须等待量价触发和失效条件确认，不构成投资建议。

#### 每日热门股
- A股半导体/AI算力/机器人/新能源方向：{_CN_AI_STOCK_POOL}。优先筛选与当日行业、概念榜和新闻线索共振的个股。

#### 有潜力的股票
- 重点观察板块内强于指数、资金承接改善、政策或业绩催化明确的核心标的。

#### 可抄底观察信号
- 只观察超跌后靠近支撑企稳、缩量回踩后重新放量、并收复短期均线的候选。失效条件：跌破支撑、板块主线退潮或放量长阴。
"""
        return """### AI选股雷达
> 仅作观察池：必须等待量价触发和失效条件确认，不构成投资建议。

#### 每日热门股
- 跟踪当日最强主题中的代表股票。

#### 有潜力的股票
- 筛选具备相对强度和明确催化的龙头。

#### 可抄底观察信号
- 只观察回踩支撑企稳并重新放量的候选。
"""

    def _build_limit_up_funnel_template_block(self, overview: MarketOverview) -> str:
        """Build fallback limit-up funnel section for A-share template reviews."""
        if self.region != "cn":
            return ""
        rows = [item for item in (overview.limit_up_pool or []) if isinstance(item, dict)]
        if not rows:
            return """### 涨停板漏斗筛选
- 今日暂无涨停池数据，无法执行“涨停逻辑 -> 题材热点 -> 趋势资金 -> 风险排除”的漏斗筛选。
- 次日重点候选：暂无；等待涨停池和量价数据恢复后再筛选。
"""
        top_rows = rows[:5]
        industry_counts = Counter(
            str(item.get("industry") or "未分类").strip() or "未分类"
            for item in rows
        )
        pool_text = "、".join(
            f"{item.get('name') or '-'}({item.get('code') or '-'})"
            for item in top_rows
        )
        industry_text = "、".join(
            f"{name}({count})"
            for name, count in industry_counts.most_common(5)
        )
        return f"""### 涨停板漏斗筛选
> 当日涨停板为临时观察池，不写回固定自选股。以下为模板筛选结果，需结合实时分时、盘口和消息复核。

- 涨停池概况：共 {len(rows)} 只；行业集中在 {industry_text or "暂无分类"}。
- 初筛观察：{pool_text}。
- 继续筛选标准：优先保留符合当下题材热点、多头趋势、突破平台、封板资金强、近一年涨幅不过大的个股。
- 风险排除：剔除明显利空、炸板频繁、高换手分歧过大、封板资金弱或高位过热的个股。
- 次日重点候选：暂无自动定论；等待 AI 报告或人工复核后，只从通过漏斗的个股中选择 1-2 只，并设置触发条件和失效条件。
"""

    def _build_review_prompt(self, overview: MarketOverview, news: List) -> str:
        """构建复盘报告 Prompt"""
        review_language = self._get_review_language()

        # 指数行情信息（简洁格式，不用emoji）
        indices_text = ""
        for idx in overview.indices:
            direction = "↑" if idx.change_pct > 0 else "↓" if idx.change_pct < 0 else "-"
            indices_text += f"- {idx.name}: {idx.current:.2f} ({direction}{abs(idx.change_pct):.2f}%)\n"
        
        # 板块信息
        top_sectors_text = self._format_ranking_summary(overview.top_sectors)
        bottom_sectors_text = self._format_ranking_summary(overview.bottom_sectors)
        top_concepts_text = self._format_ranking_summary(overview.top_concepts)
        bottom_concepts_text = self._format_ranking_summary(overview.bottom_concepts)
        limit_up_pool_text = self._format_limit_up_pool_summary(overview.limit_up_pool)
        
        # 新闻信息 - 支持 SearchResult 对象或字典
        news_text = ""
        for i, n in enumerate(news[:6], 1):
            # 兼容 SearchResult 对象和字典
            title = self._compact_news_text(self._get_news_field(n, "title"), limit=90)
            snippet = self._compact_news_text(self._get_news_field(n, "snippet"), limit=220)
            source = self._compact_news_text(self._get_news_field(n, "source"), limit=60)
            published_date = self._compact_news_text(self._get_news_field(n, "published_date"), limit=30)
            url = self._compact_news_text(self._get_news_field(n, "url"), limit=180)
            meta_parts = [part for part in (source, published_date) if part]
            meta = f" ({' / '.join(meta_parts)})" if meta_parts else ""
            url_line = f"\n   URL: {url}" if url else ""
            news_text += f"{i}. {title}{meta}\n   {snippet or '-'}{url_line}\n"
        
        # 按 region 组装市场概况与板块区块（美股/港股/日韩无涨跌家数、板块数据）
        stats_block = ""
        sector_block = ""
        data_limits_block = ""
        if review_language == "en":
            if self.profile.has_market_stats:
                stats_block = f"""## Market Breadth
- Advancers: {overview.up_count} | Decliners: {overview.down_count} | Flat: {overview.flat_count}
- Limit-up: {overview.limit_up_count} | Limit-down: {overview.limit_down_count}
- Turnover: {overview.total_amount:.0f} ({self._get_turnover_unit_label()})"""

            if self.profile.has_sector_rankings:
                sector_block = f"""## Sector / Theme Performance
Industry leading: {top_sectors_text if top_sectors_text else "N/A"}
Industry lagging: {bottom_sectors_text if bottom_sectors_text else "N/A"}
Concept leading: {top_concepts_text if top_concepts_text else "N/A"}
Concept lagging: {bottom_concepts_text if bottom_concepts_text else "N/A"}"""

            data_limit_lines = []
            if not self.profile.has_market_stats:
                data_limit_lines.append(
                    "- Market breadth, aggregate turnover, participation, and fund-flow signals are not available for this market."
                )
            if not self.profile.has_sector_rankings:
                data_limit_lines.append("- Sector/theme ranking data is not available for this market.")
            if data_limit_lines:
                data_limits_block = "## Data Limits\n" + "\n".join(data_limit_lines)
        else:
            if self.profile.has_market_stats:
                stats_block = f"""## 市场概况
- 上涨: {overview.up_count} 家 | 下跌: {overview.down_count} 家 | 平盘: {overview.flat_count} 家
- 涨停: {overview.limit_up_count} 家 | 跌停: {overview.limit_down_count} 家
- 两市成交额: {overview.total_amount:.0f} 亿元"""

            if self.profile.has_sector_rankings:
                sector_block = f"""## 板块表现
行业领涨: {top_sectors_text if top_sectors_text else "暂无数据"}
行业领跌: {bottom_sectors_text if bottom_sectors_text else "暂无数据"}
概念领涨: {top_concepts_text if top_concepts_text else "暂无数据"}
概念领跌: {bottom_concepts_text if bottom_concepts_text else "暂无数据"}"""

            limit_up_block = ""
            if self.region == "cn":
                limit_up_block = f"""## 涨停板全量观察池
{limit_up_pool_text}

## 涨停板漏斗筛选规则
- 第0层：把今日全部涨停板视为“当日临时自选池”，用于盘后筛选；不要写成永久自选股配置变更。
- 第1层：逐只解释涨停逻辑，优先用行业/概念/政策/业绩/订单/资金/情绪线索；证据不足必须写“逻辑待验证”。
- 第2层：只保留符合当下题材热点、行业榜或概念榜主线的个股，其余剔除。
- 第3层：继续筛选多头趋势、突破平台、资金或封板强、近一年涨幅不过大的个股；缺少年内涨幅或趋势数据时必须标注“需日线复核”，不得编造数值。
- 第4层：排除存在明显利空、炸板频繁、高换手分歧过大、封板资金弱或高位过热的个股；如未提供完整涨停分时图，只能根据首封/末封/炸板/封板资金判断，并标注“需分时复核”。
- 最终只输出 1-2 只“次日重点候选”，不是无条件买入；必须给出开盘/盘中触发条件、失效条件、仓位纪律和风险提示。"""

            data_limit_lines = []
            if not self.profile.has_market_stats:
                data_limit_lines.append("- 该市场暂无涨跌家数、涨跌停、成交额汇总、参与度或资金流信号。")
            if not self.profile.has_sector_rankings:
                data_limit_lines.append("- 该市场暂无行业板块/概念题材涨跌榜。")
            if data_limit_lines:
                data_limits_block = "## 数据边界\n" + "\n".join(data_limit_lines)

        data_no_indices_hint = (
            "注意：由于行情数据获取失败，请主要根据【市场新闻】进行定性分析和总结，不要编造具体的指数点位。"
            if not indices_text
            else ""
        )
        if review_language == "en":
            data_no_indices_hint = (
                "Note: Market data fetch failed. Rely mainly on [Market News] for qualitative analysis. Do not invent index levels."
                if not indices_text
                else ""
            )
            indices_placeholder = indices_text if indices_text else "No index data (API error)"
            news_placeholder = news_text if news_text else "No relevant news"
            data_boundary_requirement = (
                "- Respect Data Limits: do not invent or over-interpret unsupported breadth, fund-flow, turnover, participation, or sector-ranking data.\n"
                if data_limits_block
                else ""
            )
            market_summary_hint = (
                "2-3 sentences summarizing overall market tone, index moves, and liquidity."
                if self.profile.has_market_stats
                else "2-3 sentences summarizing overall market tone, index moves, and available news context."
            )
        else:
            indices_placeholder = indices_text if indices_text else "暂无指数数据（接口异常）"
            news_placeholder = news_text if news_text else "暂无相关新闻"
            data_boundary_requirement = (
                "- 严格遵守数据边界：未提供涨跌家数、资金流、成交额汇总或板块榜时，不要编造或过度解读。\n"
                if data_limits_block
                else ""
            )
            market_summary_hint = (
                "2-3句话概括指数、涨跌家数、成交额和情绪温度，明确“强势/偏暖/震荡/偏弱”判断"
                if self.profile.has_market_stats
                else "2-3句话概括指数表现、新闻线索和整体风险状态，不要补写未提供的市场宽度或资金流数据"
            )

        output_template_sections = self._build_output_template_sections(review_language)
        zh_market_scope_name = self._get_market_scope_name("zh")
        zh_report_title = f"{overview.date} 大盘复盘"
        if self.region == "us":
            zh_report_title = f"{overview.date} 美股早报"
        if self.region in ("jp", "kr"):
            zh_report_title = f"{overview.date} {zh_market_scope_name}大盘复盘"
        workflow_hint = (
            "报告要像交易员盘后工作台：先给结论，再按数据表、主线、催化、计划展开"
            if self.profile.has_market_stats or self.profile.has_sector_rankings
            else "报告要像交易员盘后工作台：先给结论，再按指数、新闻催化和计划展开"
        )
        zh_region_context_requirement = (
            "- 本报告是北京时间早间美股早报，分析对象是上一交易日/昨夜美股；正文优先使用“上一交易日”“昨夜美股”“美股早报”等表述，避免误写成 A 股盘后或当天 A 股行情。\n"
            if self.region == "us"
            else ""
        )

        if review_language == "en":
            report_title = self._get_review_title(overview.date).removeprefix("## ").strip()
            return f"""You are a professional {self._get_market_scope_name('en')} analyst. Please produce a concise market recap report based on the data below.

[Requirements]
- Output pure Markdown only
- No JSON
- No code blocks
- Use emoji sparingly in headings (at most one per heading)
- The entire fixed shell, headings, guidance, and conclusion must be in English
- Start with a short "Today at a Glance" block of 3 bullets: market regime, strongest theme, watch/avoid.
- Keep the report compact and scannable. Prefer short bullets over long paragraphs.
- If no market news is provided, explicitly say "No verified news catalyst provided" and do not infer unnamed macro events.
- Never invent exact individual-stock prices, support levels, moving-average values, or indicator readings unless those exact stock data points are present in the input.
- Treat all stock names as an observation pool for faster understanding, not as buy recommendations.
{data_boundary_requirement}

---

# Today's Market Data

## Date
{overview.date}

## Major Indices
{indices_placeholder}

{stats_block}

{sector_block}

{data_limits_block}

## Market News
{news_placeholder}

{data_no_indices_hint}

{self._get_strategy_prompt_block()}

{self._get_hot_watchlist_prompt_block()}

---

# Output Template (follow this structure)

## {report_title}

### Today at a Glance
(- Market regime; strongest theme; watch/avoid.)

### 1. Market Summary
({market_summary_hint})

### 2. Index Commentary
({self._get_index_hint()})

{output_template_sections}

---

Output the report content directly, no extra commentary.
"""

        # A 股场景使用中文提示语
        return f"""你是一位专业的{self._get_market_scope_name('zh')}分析师，请根据以下数据生成一份结构化的{self._get_market_scope_name('zh')}大盘复盘报告。

【重要】输出要求：
- 必须输出纯 Markdown 文本格式
- 禁止输出 JSON 格式
- 禁止输出代码块
- emoji 仅在标题处少量使用（每个标题最多1个）
- {workflow_hint}
- 不要重复列出已由系统注入的表格数据；正文负责解释表格背后的含义
- 开头必须有“今日速读”三条：市场状态、最强主线、观察/回避方向。
- “今日速读”后必须紧跟“30秒行动卡”，把最重要的观察和纪律放在飞书第一屏。
- 内容要适合飞书快速阅读：多用短句和要点，避免长篇段落；中文报告建议控制在 1200-1800 字，最多 2 张表。
- 报告必须围绕“快速观察、理解市场、再考虑是否入手”，不要制造确定性。
- 每个重点结论尽量按“证据 -> 判断 -> 动作”表达；没有数据支撑时必须标注“需复核”，不要用空泛形容词代替筛选理由。
- 推送优先级必须固定为：市场状态 -> 明日动作 -> 候选池 -> 风险否决；背景解释放在后面，不要把候选和风险埋到长段落里。
{zh_region_context_requirement}
- 如果市场新闻为空，必须明确写“暂无已验证消息催化”，不得反推出未给出的宏观事件。
- 没有个股行情数据时，禁止编造具体股价、支撑价、均线数值、MACD 等指标数值；只能写“站上短期均线”“突破前高”“回踩支撑企稳”等相对触发条件。
- AI选股池仅用于帮助快速观察和理解市场，不得直接写成买入指令。
{data_boundary_requirement}

---

# 今日市场数据

## 日期
{overview.date}

## 主要指数
{indices_placeholder}

{stats_block}

{sector_block}

{limit_up_block}

{data_limits_block}

## 市场新闻
{news_placeholder}

{data_no_indices_hint}

{self._get_strategy_prompt_block()}

{self._get_hot_watchlist_prompt_block()}

---

# 输出格式模板（请严格按此格式输出）

## {zh_report_title}

### 今日速读
- 市场状态：
- 最强主线：
- 观察/回避：

> 一句话给出今日市场状态、核心矛盾和明日优先观察方向。

### 30秒行动卡
| 项目 | 证据 | 判断 | 动作 |
|------|------|------|------|
| 市场温度 |  |  |  |
| 最强主线 |  |  |  |
| 优先观察 |  |  |  |
| 暂时回避 |  |  |  |

### 明日执行清单
- 仓位姿态：
- 只看方向：
- 不碰方向：
- 触发加关注：
- 触发撤退：

### 一、盘面总览
（{market_summary_hint}）

### 二、指数结构
（{self._get_index_hint()}，说明谁在护盘、谁在拖累，以及关键支撑/压力）

{output_template_sections}

---

请直接输出复盘报告内容，不要输出其他说明文字。
"""
    
    def _generate_template_review(self, overview: MarketOverview, news: List) -> str:
        """使用模板生成复盘报告（无大模型时的备选方案）"""
        template_language = self._get_template_review_language()
        mood_code = self.profile.mood_index_code
        # 根据 mood_index_code 查找对应指数
        # cn: mood_code="000001"，idx.code 可能为 "sh000001"（以 mood_code 结尾）
        # us: mood_code="SPX"，idx.code 直接为 "SPX"
        mood_index = next(
            (
                idx
                for idx in overview.indices
                if idx.code == mood_code or idx.code.endswith(mood_code)
            ),
            None,
        )
        if mood_index:
            if mood_index.change_pct > 1:
                market_mood = self._get_market_mood_text("strong_up", template_language)
            elif mood_index.change_pct > 0:
                market_mood = self._get_market_mood_text("mild_up", template_language)
            elif mood_index.change_pct > -1:
                market_mood = self._get_market_mood_text("mild_down", template_language)
            else:
                market_mood = self._get_market_mood_text("strong_down", template_language)
        else:
            market_mood = self._get_market_mood_text("range", template_language)
        
        # 指数行情（简洁格式）
        indices_text = ""
        for idx in overview.indices[:4]:
            direction = "↑" if idx.change_pct > 0 else "↓" if idx.change_pct < 0 else "-"
            indices_text += f"- **{idx.name}**: {idx.current:.2f} ({direction}{abs(idx.change_pct):.2f}%)\n"
        
        # 板块信息
        separator = ", " if template_language == "en" else "、"
        top_text = separator.join([s['name'] for s in overview.top_sectors[:3]])
        bottom_text = separator.join([s['name'] for s in overview.bottom_sectors[:3]])
        top_concept_text = separator.join([s['name'] for s in overview.top_concepts[:3]])
        bottom_concept_text = separator.join([s['name'] for s in overview.bottom_concepts[:3]])

        if template_language == "en":
            stats_section = ""
            if self.profile.has_market_stats:
                stats_section = f"""
### 3. Breadth & Liquidity
| Metric | Value |
|--------|-------|
| Advancers | {overview.up_count} |
| Decliners | {overview.down_count} |
| Limit-up | {overview.limit_up_count} |
| Limit-down | {overview.limit_down_count} |
| Turnover ({self._get_turnover_unit_label()}) | {overview.total_amount:.0f} |
"""
            sector_section = ""
            if self.profile.has_sector_rankings and (top_text or bottom_text or top_concept_text or bottom_concept_text):
                sector_section = f"""
### 4. Sector / Theme Highlights
- **Industry Leaders**: {top_text or "N/A"}
- **Industry Laggards**: {bottom_text or "N/A"}
- **Concept Leaders**: {top_concept_text or "N/A"}
- **Concept Laggards**: {bottom_concept_text or "N/A"}
"""
            market_names = {
                "us": "US Market Recap",
                "hk": "HK Market Recap",
                "jp": "Japan Market Recap",
                "kr": "Korea Market Recap",
            }
            market_name = market_names.get(self.region, "A-share Market Recap")
            ai_stock_selection_block = self._build_ai_stock_selection_template_block(template_language)
            report = f"""## {overview.date} {market_name}

### 1. Market Summary
Today's {self._get_market_scope_name(template_language)} showed **{market_mood}**.

### 2. Major Indices
{indices_text or "- No index data available"}
{stats_section}
{sector_section}
{ai_stock_selection_block}

### Risk Alerts
Market conditions can change quickly. The data above is for reference only and does not constitute investment advice.

{self._get_strategy_markdown_block(template_language)}

---
*Review Time: {datetime.now().strftime('%H:%M')}*
"""
            return report

        market_labels = {"cn": "A股", "us": "美股", "hk": "港股", "jp": "日股", "kr": "韩股"}
        market_label = market_labels.get(self.region, "A股")
        dashboard_block = self._build_stats_block(overview) if self.profile.has_market_stats else ""
        indices_block = self._build_indices_block(overview)
        sector_block = self._build_sector_block(overview) if self.profile.has_sector_rankings else ""
        summary_focus = (
            "指数承接、成交额变化和板块持续性"
            if self.profile.has_market_stats and self.profile.has_sector_rankings
            else "指数承接、消息催化和整体风险状态"
        )
        market_summary_block = (
            dashboard_block
            if dashboard_block
            else (
                "暂无市场宽度数据。"
                if self.profile.has_market_stats
                else "- 当前以主要指数与可用新闻线索评估整体风险状态。"
            )
        )
        sector_section = (
            f"""
### 三、板块主线
{sector_block or "- 暂无板块涨跌榜数据。"}
"""
            if self.profile.has_sector_rankings
            else ""
        )
        ai_stock_selection_block = self._build_ai_stock_selection_template_block(template_language)
        limit_up_funnel_block = self._build_limit_up_funnel_template_block(overview)
        funds_section = (
            """
### 资金与情绪
- 结合成交额和涨跌家数看，当前更适合等待确认，避免仅凭单一热点追高。
"""
            if self.profile.has_market_stats
            else ""
        )
        return f"""## {overview.date} 大盘复盘

> 今日{market_label}市场整体呈现**{market_mood}**态势，优先观察{summary_focus}。

### 一、盘面总览
{market_summary_block}

### 二、指数结构
{indices_block or indices_text or "暂无指数数据。"}
{sector_section}
{limit_up_funnel_block}
{ai_stock_selection_block}

{funds_section}

### 消息催化
- 暂无可用新闻时，应降低对题材持续性的确定性判断。

{self._get_strategy_markdown_block(template_language)}

### 风险提示
- 市场有风险，投资需谨慎。以上数据仅供参考，不构成投资建议。

---
*复盘时间: {datetime.now().strftime('%H:%M')}*
"""
    
    def _run_daily_review_parts(self) -> MarketLightReviewResult:
        """Run market review once and keep report/snapshot on the same overview."""
        logger.info("========== 开始大盘复盘分析 ==========")

        # 1. 获取市场概览
        overview = self.get_market_overview()

        # 2. 搜索市场新闻
        news = self.search_market_news()
        news = self._merge_persisted_market_intelligence(news)

        # 3. 生成复盘报告
        report = self.generate_market_review(overview, news)
        snapshot = self.build_market_light_snapshot(overview) if self._supports_market_light() else None
        structured_payload = self.build_market_review_payload(
            overview,
            news,
            report,
            snapshot,
        )

        logger.info("========== 大盘复盘分析完成 ==========")

        return MarketLightReviewResult(
            overview=overview,
            report=report,
            market_light_snapshot=snapshot,
            structured_payload=structured_payload,
        )

    def _merge_persisted_market_intelligence(self, news: List) -> List:
        """Merge local persisted market intelligence and search news with bounded prompt/payload slot preservation."""
        search_news = list(news or [])
        merged_local = []
        seen_urls = {
            self._get_news_field(item, "url")
            for item in search_news
            if self._get_news_field(item, "url")
        }
        try:
            service = IntelligenceService()
            payload = service.list_items(
                scope_type="market",
                market=self.region,
                published_days=max(1, int(self.config.get_effective_news_window_days() or 1)),
                page=1,
                page_size=6,
            )
            for item in payload.get("items", []):
                if not isinstance(item, dict):
                    continue
                url = str(item.get("url") or "")
                if url and url in seen_urls:
                    continue
                seen_urls.add(url)
                merged_local.append({
                    "title": item.get("title") or "未命名资讯",
                    "snippet": item.get("summary") or "",
                    "source": item.get("source") or item.get("source_name") or "local-intel",
                    "published_date": item.get("published_at") or "",
                    "url": "" if url.startswith("no-url:intel:") else url,
                })
        except Exception as exc:
            logger.debug("[大盘] %s action=load_local_intelligence status=failed error=%s", self._log_context(), exc)
        merged_news = []
        merged_local_index = 0
        merged_search_index = 0
        while merged_local_index < len(merged_local) or merged_search_index < len(search_news):
            if merged_local_index < len(merged_local):
                merged_news.append(merged_local[merged_local_index])
                merged_local_index += 1
            if merged_search_index < len(search_news):
                merged_news.append(search_news[merged_search_index])
                merged_search_index += 1
        return merged_news

    def run_daily_review(self) -> str:
        """
        执行每日大盘复盘流程

        Returns:
            复盘报告文本
        """
        return self.run_daily_review_with_snapshot().report

    def run_daily_review_with_snapshot(self) -> MarketLightReviewResult:
        """Run daily review and return the report plus its structured Market Light snapshot."""
        return self._run_daily_review_parts()


# 测试入口
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    )
    
    analyzer = MarketAnalyzer()
    
    # 测试获取市场概览
    overview = analyzer.get_market_overview()
    print(f"\n=== 市场概览 ===")
    print(f"日期: {overview.date}")
    print(f"指数数量: {len(overview.indices)}")
    for idx in overview.indices:
        print(f"  {idx.name}: {idx.current:.2f} ({idx.change_pct:+.2f}%)")
    print(f"上涨: {overview.up_count} | 下跌: {overview.down_count}")
    print(f"成交额: {overview.total_amount:.0f}亿")
    
    # 测试生成模板报告
    report = analyzer._generate_template_review(overview, [])
    print(f"\n=== 复盘报告 ===")
    print(report)
