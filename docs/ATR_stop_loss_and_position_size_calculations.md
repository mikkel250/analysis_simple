# Basic ATR Stop-Loss Calculation
Recommended multiplier is 1-3 depending on how volatile the asset is, e.g. BTC might be 1 and memcoins are 3. 

**Long Position**
Stop Loss=Entry Price−(ATR×Multiplier)

**For a short position:**
Stop Loss=Entry Price+(ATR×Multiplier)

ATR-based position sizing is a method that adjusts your trade size according to the current market volatility measured by the Average True Range (ATR). This helps align your risk exposure with how much the price typically moves, improving risk management and preventing outsized losses.

---

## How ATR Is Used for Position Sizing

1. **Determine Your Risk Per Trade**  
   Decide the maximum dollar amount (or percentage of your capital) you are willing to risk on a single trade. For example, you might risk $500 or 2% of your account.

2. **Calculate Per-Unit Risk Using ATR**  
   ATR measures the average price movement (volatility) over a period. By multiplying ATR by a stop loss multiplier (e.g., 1× or 2× ATR), you get the price distance you expect your stop loss to be from your entry.  
   - Example: If ATR = $2 and you use a 2× ATR stop, your stop loss is $4 away from entry.

3. **Calculate Position Size**  
   Divide your risk per trade by the per-unit risk to get the number of units/contracts/shares you can trade without exceeding your risk limit.  
   - Example: Risk per trade = $500, per-unit risk = $4, so position size = 500 / 4 = 125 shares.
---
Here’s a concise summary of ATR-based position sizing and its calculation for your notes:

---

## ATR-Based Position Sizing: Summary

**Concept:**  
ATR (Average True Range) measures an asset’s volatility by averaging its price range over a set period. Using ATR for position sizing helps adjust trade size according to current market volatility, ensuring consistent risk exposure regardless of how volatile the asset is.

**Why Use ATR for Position Sizing?**  
- Adapts position size to market volatility: smaller size in high volatility, larger size in low volatility.  
- Keeps your risk per trade consistent in dollar terms.  
- Helps avoid emotional or arbitrary sizing decisions.

---

## Basic Calculation Steps

1. **Determine your maximum risk per trade** (e.g., 2% of your account equity).  
2. **Calculate the ATR value** for your asset (commonly 14-period ATR).  
3. **Choose an ATR multiplier** for your stop loss distance (often 1.5 to 3× ATR).  
4. x
   {Stop Loss Distance} = {ATR} * {Multiplier}
5. **Calculate position size** by dividing your risk per trade by the stop loss distance:  
   Position Size = Risk per Trade/Stop Loss Distance

---

## Example

- Account size: $10,000  
- Risk per trade: 2% → $200  
- ATR: $5  
- ATR multiplier: 2  
- Stop loss distance = $5 × 2 = $10  
- Position size = $200 ÷ $10 = 20 units (shares, contracts, etc.)

This means you buy 20 units so that if the price moves $10 against you, your loss is limited to $200 (2% of your account).

---

## Key Takeaways

- ATR-based sizing dynamically adjusts for volatility, protecting you from oversized losses in choppy markets.  
- Regularly update ATR values to reflect changing market conditions.  
- Combine ATR sizing with other risk management tools like stop losses and position limits.  
- This method is especially useful in volatile markets like crypto, where price swings can be large and unpredictable.

---

**Sources:**  
LuxAlgo (2025), Tiomarkets (2024), TheRobustTrader (2025), Altrady, UEEX Technology

---

This summary captures the core idea and practical calculation of ATR-based position sizing for effective risk management.

Citations:
[1] https://www.luxalgo.com/blog/5-position-sizing-methods-for-high-volatility-trades/
[2] https://tiomarkets.com/ja/article/average-true-range-in-position-trading-guide
[3] https://blog.ueex.com/strategies-for-position-sizing-in-crypto-trading/
[4] https://www.altrady.com/crypto-trading/technical-analysis/average-true-range
[5] https://therobusttrader.com/how-to-use-atr-in-position-sizing/
[6] https://tiomarkets.com/en/article/average-true-range-in-position-trading-guide
[7] https://biyond.co/blog/guides/the-art-of-position-sizing-in-crypto-trading.html
[8] https://www.investopedia.com/terms/a/atr.asp

---
Answer from Perplexity: pplx.ai/share
---

## Why Use ATR for Position Sizing?

- **Adapts to Market Volatility:**  
  In highly volatile markets (high ATR), price swings are larger, so you reduce position size to keep risk constant. In calmer markets (low ATR), you can take larger positions.

- **Improves Risk Consistency:**  
  Using ATR-based sizing helps maintain a consistent dollar risk across trades, regardless of the asset’s price or volatility.

- **Reduces Emotional Trading:**  
  Systematic position sizing based on ATR removes guesswork and helps avoid overexposure in choppy markets.

---
Using ATR for position sizing helps you tailor trade sizes to market conditions, keeping risk consistent and improving your overall trading discipline.

---

**Sources:**  
[Altrady](https://www.altrady.com/crypto-trading/technical-analysis/average-true-range)  
[Oanda](https://www.oanda.com/us-en/trade-tap-blog/analysis/technical/how-to-use-average-true-range-atr/)  
[IG](https://www.ig.com/en/trading-strategies/what-is-the-average-true-range--atr--indicator-and-how-do-you-tr-240905)

Citations:
[1] https://tiomarkets.com/ja/article/average-true-range-in-position-trading-guide
[2] https://www.altrady.com/crypto-trading/technical-analysis/average-true-range
[3] https://www.bitget.com/wiki/how-to-atr
[4] https://www.oanda.com/us-en/trade-tap-blog/analysis/technical/how-to-use-average-true-range-atr/
[5] https://www.ig.com/en/trading-strategies/what-is-the-average-true-range--atr--indicator-and-how-do-you-tr-240905
[6] https://www.youtube.com/watch?v=sC6bpah9eRs
[7] https://www.schwab.com/learn/story/average-true-range-indicator-and-volatility
[8] https://www.avatrade.com/education/technical-analysis-indicators-strategies/atr-indicator-strategies

---
Answer from Perplexity: pplx.ai/share