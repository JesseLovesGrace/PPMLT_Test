import pandas as pd


def split_csv_by_date(input_file, output_file_before, output_file_after, split_date):
    # Read the CSV file
    df = pd.read_csv(input_file)

    # Convert the 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Split the DataFrame based on the specified date
    df_before = df[df['Date'] < split_date]
    df_after = df[df['Date'] >= split_date]

    # Write the results to new CSV files
    df_before.to_csv(output_file_before, index=False)
    df_after.to_csv(output_file_after, index=False)


# Specify file paths
input_csv_path = "C:\\Users\\jesse\\Desktop\\PPMLT\\Tests\\APPL\\AAPL.csv"
output_csv_path_before = "C:\\Users\\jesse\\Desktop\\PPMLT\\Tests\\APPL\\AAPL_train.csv"
output_csv_path_after = "C:\\Users\\jesse\\Desktop\\PPMLT\\Tests\\APPL\\AAPL_test.csv"

# Specify the split date
split_date = pd.to_datetime('2014-01-02')

# Call the function to split the CSV file
split_csv_by_date(input_csv_path, output_csv_path_before, output_csv_path_after, split_date)

print(f"CSV files have been created: {output_csv_path_before} and {output_csv_path_after}")
