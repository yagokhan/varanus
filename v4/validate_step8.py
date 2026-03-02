import pandas as pd
import numpy as np

from varanus.risk import check_portfolio_health, RISK_CONFIG

def generate_mock_equity(start_capital: float, days: int, crash: bool = False) -> pd.Series:
    """Generates a dummy equity curve for testing circuit breakers."""
    base_time = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
    times = [base_time + pd.Timedelta(hours=4 * i) for i in range(days * 6)]
    
    # Healthy gradual uptrend
    equity_list = [start_capital]
    for _ in range(1, len(times)):
        # Random walk with slight positive drift
        change = equity_list[-1] * np.random.normal(0.001, 0.005)
        equity_list.append(equity_list[-1] + change)
        
    equity = pd.Series(equity_list, index=times)
        
    if crash:
        # Induce a massive -16% crash in the final 24 hours to trip both breakers
        crash_start_idx = -6 
        for i in range(crash_start_idx, 0):
            equity[i] = equity[i-1] * 0.96 # Drop 4% per 4h candle
            
    return pd.Series(equity, index=times)


def run():
    print("=== Testing Varanus Risk Management Layer ===\\n")
    
    # Test 1: Healthy Portfolio
    print("Test 1: Normal Market Conditions")
    healthy_curve = generate_mock_equity(RISK_CONFIG['initial_capital'], days=30, crash=False)
    healthy_health = check_portfolio_health(healthy_curve)
    
    print(f"Current Equity: ${healthy_health['current_equity']:.2f}")
    print(f"Daily Loss:     {healthy_health['daily_loss_pct']}%")
    print(f"Max Drawdown:   {healthy_health['drawdown_pct']}%")
    print(f"Halt Signals:   {healthy_health['halt_signals']}")
    
    assert healthy_health['halt_signals'] is False, "False positive on circuit breaker!"
    print("PASS: Circuit breakers remained open.\\n")
    
    # Test 2: Flash Crash
    print("Test 2: Flash Crash Event (-15%+ Drawdown)")
    crash_curve = generate_mock_equity(RISK_CONFIG['initial_capital'], days=30, crash=True)
    crash_health = check_portfolio_health(crash_curve)
    
    print(f"Current Equity: ${crash_health['current_equity']:.2f}")
    print(f"Daily Loss:     {crash_health['daily_loss_pct']}%")
    print(f"Max Drawdown:   {crash_health['drawdown_pct']}%")
    print(f"Halt Signals:   {crash_health['halt_signals']}")
    
    assert crash_health['halt_signals'] is True, "Circuit breaker FAILED to trip!"
    print("PASS: Circuit breaker tripped correctly.")
    
    print("\\n[+] Step 8 Validation Complete.")

if __name__ == "__main__":
    run()
