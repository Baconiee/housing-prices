from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def get_preprocessor(categorical_cols_with_na):
    """
    Returns the preprocessor for the Housing dataset.
    """

    num_median_imputer = SimpleImputer(strategy='median')
    num_zero_imputer = SimpleImputer(strategy='constant', fill_value=0)

    cat_none_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    cat_generic_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num_med', num_median_imputer, ['LotFrontage']),
            ('num_zero', num_zero_imputer, ['MasVnrArea', 'GarageYrBlt']),
            ('cat_none', cat_none_pipeline, categorical_cols_with_na),
            ('cat_gen', cat_generic_pipeline, make_column_selector(dtype_include=['object']))
        ],
        remainder='passthrough'
    )

    preprocessor.set_output(transform="pandas")
    return preprocessor