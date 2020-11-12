"""
This document stores all the data preprocessing and models I value
"""
#-------------------------------------------------------------------------------
# Le Wagon data processing & model
col_num = []
col_bool =[]
col_object =[]

for col in df:
    if df[col].dtype == "float64":
        col_num.append(col)
    if df[col].dtype == "int64":
        col_num.append(col)
    if df[col].dtype == 'bool':
        col_bool.append(col)
    if df[col].dtype == 'object':
        col_object.append(col)

col_bool.remove('target')

numeric_transformer = make_pipeline(SimpleImputer(), MinMaxScaler())

categorical_transformer = make_pipeline(SimpleImputer(strategy='most_frequent'),
                                        OneHotEncoder(drop= 'first', handle_unknown='error'))
feateng_blocks = [
            ('num', numeric_transformer, col_num),
            ('cat', categorical_transformer, col_object),
        ]
features_encoder = ColumnTransformer(feateng_blocks, n_jobs=None, remainder="passthrough")

pipeline = Pipeline([
            ('features', features_encoder)])

X = df.drop(columns=['target'])
X = pipeline.fit_transform(X)

y = df.summit_success

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state= 1)

# Model Traning
boost = XGBClassifier()
boost.fit(X_train, y_train)

# Save trained model
model_name = 'XGB_model.joblib'
joblib.dump(boost, model_name)

pipe_name = "pipe_transformation.joblib"
joblib.dump(pipeline, pipe_name)

# Export pipeline as pickle file
with open("pipeline.pkl", "wb") as file:
    dump(pipeline, file)

#-------------------------------------------------------------------------------
# Imputer. Here it does nothing. Just return the same value
class BoolImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        return None
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X

#-------------------------------------------------------------------------------
# Pipeline for features preprocessing
preprocessor = ColumnTransformer([
    ('median', SimpleImputer(strategy='median'), ['col1']),
    ('frequency', SimpleImputer(strategy='most_frequent'), ['col2', 'col3']),
    ('ohe', OneHotEncoder(drop= 'first'), ['col4'])])

final_pipe = Pipeline([('preprocessing', preprocessor)])

data_trans = final_pipe.fit_transform(data)

#-------------------------------------------------------------------------------
# Find features'names after OHE in a pipeline
ohe_col = list(clf.named_steps['preprocessor'].transformers_[2][1]\
   .named_steps['onehot'].get_feature_names(col_object))

feature_names = col_num + col_bool + ohe_col

#-------------------------------------------------------------------------------
# map numerical values to text
df['season'] = df['season'].map({
                0 : 'Unknown',
                1 : 'Spring',
                2 : 'Summer',
                3 : 'Autumn',
                4 : 'Winter'})


#-------------------------------------------------------------------------------
# function to get the data of multiple .xsls files stored in one folder
def get_data(self):
    """
    This function get the data from the xls file and return a DataFrame.
    """
    root_dir = os.path.abspath('')
    xls_path = os.path.join(root_dir, 'data')
    file_names = [f for f in os.listdir(xls_path) if f.endswith('.xls')]

    def key_from_file_name(f):
        if f[-4:] == '.xls':
            return f[:-4]
    data = {}
    for f in file_names:
        data[key_from_file_name(f)] = pd.read_excel(os.path.join(xls_path, f))

    data = data['expeds']

    return data
