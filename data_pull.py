from flask import Flask, render_template, request, jsonify
import yfinance as yf

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_stock_price', methods=['GET'])
def get_stock_price():
    company_symbol = request.args.get('symbol')
    if not company_symbol:
        return jsonify('index.html', error= 'No symbol provided'), 400
    
    # Fetch stock data using yfinance
    stock = yf.Ticker(company_symbol)
    stock_info = stock.history(period="1d")
    
    if not stock_info.empty:
        # Get the last available closing price
        price = stock_info['Close'].iloc[-1]
        return jsonify({'symbol': company_symbol, 'price': round(price, 2)})
    else:
        return jsonify({'error': 'Could not fetch data or invalid symbol'}), 500

if __name__ == '__main__':
    app.run(debug=True)
