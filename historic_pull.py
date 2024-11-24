from flask import Flask, render_template, request
import yfinance as yf
from datetime import datetime
import pandas as pd
import os
from Analyze import handle
app = Flask(__name__)

# Folder to store CSV files
CSV_FOLDER = os.path.join(os.getcwd(), "csv_files")
if not os.path.exists(CSV_FOLDER):
    os.makedirs(CSV_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_historical_data', methods=['POST'])
def get_historical_data():
    company_symbol = request.form.get('symbol')
    if not company_symbol:
        return render_template('index.html', error='No symbol provided')

    try:
        end_date = datetime.now().date()
        start_date = end_date.replace(year=end_date.year - 7)

        stock = yf.Ticker(company_symbol)
        hist_data = stock.history(start=start_date, end=end_date)

        if hist_data.empty:
            return render_template('index.html', error='No data found for the symbol')

        csv_file_name = f"{company_symbol}_historical_data.csv"
        csv_file_path = os.path.join(CSV_FOLDER, csv_file_name)


        hist_data = hist_data[['Open', 'High', 'Low', 'Close', 'Volume']].reset_index()
        hist_data['Date'] = hist_data['Date'].dt.strftime('%Y-%m-%d')
        hist_data.to_csv(csv_file_path, index=False)
        handle(csv_file_name)
        
        return render_template('index.html', data=hist_data.to_dict(orient='records'), symbol=company_symbol, success=f"Data saved successfully as {csv_file_name}!")

    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
