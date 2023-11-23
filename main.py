from indi_cal import rsi_calculator
from indi_cal import ema_calculator
from indi_cal import macd_calculator

input_csv = "C:\\Users\\jesse\\Desktop\\PPMLT\\Tests\\APPL\\AAPL.csv"

rsi_calculator.add_rsi(input_csv, input_csv)
macd_calculator.add_macd(input_csv, input_csv)
ema_calculator.add_ema_columns_to_csv(input_csv,input_csv)
