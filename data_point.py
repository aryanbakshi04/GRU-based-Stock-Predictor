import requests
import pandas as pd
from bs4 import BeautifulSoup


# URL of your Flask app's endpoint
url = 'http://localhost:5000/get_historical_data'

# Parameters for the POST request
params = {'symbol': 'AAPL'}  # Replace 'AAPL' with the desired company symbol

# Make a POST request to retrieve the data
response = requests.post(url, data=params)

# Check if the request was successful
if response.status_code == 200:
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all the table rows in the HTML
    rows = soup.find_all('tr')

    # Extract the data from each row
    data = []
    for row in rows:
        cols = row.find_all('td')
        cols = [col.text.strip() for col in cols]
        data.append(cols)

    # Convert the data to a pandas DataFrame
    df = pd.DataFrame(data, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])

    # Define the filename for the CSV
    csv_filename = f"{params['symbol']}_historical_data.csv"

    # Save the DataFrame to a CSV file
    df.to_csv(csv_filename, index=False)

    print(f"Data saved to {csv_filename}")
else:
    print(f"Failed to retrieve data. Status Code: {response.status_code}")
