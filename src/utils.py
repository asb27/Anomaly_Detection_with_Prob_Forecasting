import pandas as pd

df_ex_factors_qxgboost = pd.DataFrame()
df_ex_factors_quantile_regression = pd.DataFrame()
detection_results = {}

for model_type, df in detection_results.items():
   globals()[f'df_{model_type}'] = df




import pandas as pd

def count_ones_in_column(df: pd.DataFrame, column_name: str) -> int:
    if column_name in df.columns:
        return (df[column_name] == 1).sum()
    else:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")


def get_rows_with_ones(df, column_name):

    if column_name not in df.columns:
        raise ValueError(f"{column_name} sütunu DataFrame'de bulunamadı.")

    filtered_df = df[df[column_name] == 1]

    return filtered_df
df_x = get_rows_with_ones(df_ex_factors_qxgboost,"det_quantile_upper_quantile_0.99_lower_quantile_0.0005")


def plot_row_data_in_date_range(df: pd.DataFrame, row_name: str, start_date: str, end_date: str):

    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise ValueError("DataFrame index must be of DateTime type.")

    filtered_df = df.loc[start_date:end_date]

    if row_name not in filtered_df.columns:
        raise ValueError(f"Row '{row_name}' does not exist in the DataFrame.")

    plt.figure(figsize=(10, 6))
    plt.plot(filtered_df.index, filtered_df[row_name], marker='o', linestyle='-')
    plt.title(f"Data for {row_name} from {start_date} to {end_date}")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()#




def remove_rows_in_date_range(dfs_dict: dict, start_date: str, end_date: str) -> dict:
    for key, df in dfs_dict.items():
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            raise ValueError(f"DataFrame {key} index must be of DateTime type.")

        # Filter out rows within the specified date range
        dfs_dict[key] = df.loc[~df.index.isin(pd.date_range(start_date, end_date))]

    return dfs_dict


import pandas as pd

def count_values_based_on_threshold(df: pd.DataFrame, column_name: str, threshold: int) -> tuple:
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

    greater_than_threshold = (df[column_name] > threshold).sum()
    print("greater_than_threshold:: " , greater_than_threshold)
    less_than_threshold = (df[column_name] < threshold).sum()
    print("less_than_threshold:: " , less_than_threshold)

    return greater_than_threshold, less_than_threshold



def sort_dataframe_by_column(df, column_name):
    sorted_df = df.sort_values(by=column_name)
    return sorted_df


import pandas as pd
import matplotlib.pyplot as plt


def plot_columns_with_quantiles(df: pd.DataFrame, column1: str, column2: str, quantile1: str, quantile2: str,
                                date_range: tuple = None):

    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise ValueError("DataFrame index must be of DateTime type.")

    if column1 not in df.columns or column2 not in df.columns or quantile1 not in df.columns or quantile2 not in df.columns:
        raise ValueError(
            f"One or more of the columns '{column1}', '{column2}', '{quantile1}', '{quantile2}' do not exist in the DataFrame.")

    if date_range:
        start_date, end_date = date_range
        df = df.loc[start_date:end_date]

    plt.figure(figsize=(10, 6))

    plt.plot(df.index, df[column1], label=column1, marker='o', linestyle='-')
    plt.plot(df.index, df[column2], label=column2, marker='x', linestyle='-')

    plt.fill_between(df.index, df[quantile1], df[quantile2], color='gray', alpha=0.3,
                     label=f"{quantile1} - {quantile2} range")

    plt.title(f"Line Graph of {column1} and {column2} with Quantile Ranges")
    plt.xlabel("Date")
    plt.ylabel("Values")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_roc_curve(fprr, tprr):
    from sklearn.metrics import auc

    auc_value = auc(fprr, tprr)

    plt.figure()
    plt.plot(fprr, tprr, color='blue', label=f'ROC curve (AUC = {auc_value:.2f})')
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')  # Reference line for random classification
    plt.xlim([-0.1, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xticks([-0.1, 0.0, 0.5, 1.0], fontsize=20)  # Set font size for X-axis ticks
    plt.yticks([0.2, 0.5, 1.0], fontsize=20)  # Set font size for Y-axis ticks
    plt.xlabel('False Positive Rate (FPR)', fontsize=20)
    plt.ylabel('True Positive Rate (TPR)', fontsize=20)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=20)
    plt.legend(loc="lower right", fontsize=20)
    plt.show()


