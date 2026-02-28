from varanus.alerts import ALERT_FORMAT, REQUIRED_FIELDS

def run():
    print("=== Testing Varanus Alert System ===\\n")
    
    mock_trade = {f: "TEST" for f in REQUIRED_FIELDS}
    mock_trade.update({
        "asset":        "LINK",
        "direction":    "LONG",   
        "confidence":   0.87, 
        "entry_price":  18.45,
        "take_profit":  19.50,
        "stop_loss":    17.90,
        "rr_ratio":     2.08,
        "leverage":     2.0, 
        "rvol":         1.82, 
        "rsi":          38.4, 
        "port_lev":     1.6,
        "position_usd": 1250, 
        "atr_14":       0.0432,
        "mss":          "Bullish",
        "htf_bias":     "Bullish",
    })
    
    print("Test 1: Render Markdown Format String")
    try:
        rendered = ALERT_FORMAT.format(**mock_trade)
        print("PASS: Alert formatted successfully without KeyError:\\n")
        print("-" * 40)
        print(rendered)
        print("-" * 40 + "\\n")
    except KeyError as e:
        print(f"FAIL: Missing Key -> {e}")
        assert False
        
    print("Test 2: Missing Required Field Guard")
    from varanus.alerts import send_alert
    
    # Intentionally break the struct
    del mock_trade["atr_14"]
    
    try:
        # Dummy webhook credentials
        send_alert(mock_trade, "123:dummy", "abc")
        print("FAIL: The sender ignored the missing 'atr_14' field!")
        assert False
    except ValueError as e:
        print(f"PASS: Properly caught missing field -> {e}")
        
    print("\\n[+] Step 9 Validation Complete.")

if __name__ == "__main__":
    run()
