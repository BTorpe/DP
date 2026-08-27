import pandas as pd
from dstapi import DstApi


def load_dst(table, variables, value_name):
    """Download and clean a table from Statistics Denmark.

    Downloads the requested variables from the given DST table,
    replaces missing values ('..') with NaN, converts types, and
    returns a tidy DataFrame indexed by year.

    Args:
        table (str): DST table name, e.g. 'IFOR41'.
        variables (list): list of variable dicts for the API,
            e.g. [{'code': 'ULLIG', 'values': ['70']}, ...].
            Must include a 'Tid' variable.
        value_name (str): name to give the value column
            (INDHOLD), e.g. 'gini'.

    Returns:
        pd.DataFrame: cleaned data. Columns that are constant
            across all rows are dropped; 'TID' is renamed 'year'
            and set as a sorted index.
    """

    # a. download
    params = {
        'table': table,
        'format': 'BULK',
        'lang': 'en',
        'variables': variables,
    }
    df = DstApi(table).get_data(params=params)

    # b. convert INDHOLD to float; '..' becomes NaN
    df['INDHOLD'] = pd.to_numeric(df['INDHOLD'], errors='coerce')

    # c. year to int
    df['TID'] = df['TID'].astype(int)

    # d. drop columns that carry no information (same value in every row)
    for col in df.columns:
        if col not in ('TID', 'INDHOLD') and df[col].nunique() == 1:
            df = df.drop(columns=col)

    # e. rename and index
    df = df.rename(columns={'TID': 'year', 'INDHOLD': value_name})
    df = df.set_index('year').sort_index()

    return df