import pandas as pd
data=pd.DataFrame([[0]],columns=['counter'])
data=data.set_index('counter')
data.to_csv('./data/counter')
