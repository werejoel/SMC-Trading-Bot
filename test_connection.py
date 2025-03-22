import MetaTrader5 as mt5
from config import MT5_CONFIG

def test_mt5_connection():
    if not mt5.initialize(
        path=MT5_CONFIG['path'],
        login=MT5_CONFIG['login'],
        password=MT5_CONFIG['password'],
        server=MT5_CONFIG['server']
    ):
        print("MT5 initialization failed")
        return False
    
    print("MT5 connection successful")
    print(f"Terminal Info: {mt5.terminal_info()}")
    print(f"Account Info: {mt5.account_info()}")
    
    mt5.shutdown()
    return True

if __name__ == "__main__":
    test_mt5_connection()
