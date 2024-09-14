import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

warnings.filterwarnings('ignore')
st.title("Telco analysis")
st.header("Welcome to our company :sunglasses:")
data = pd.read_csv('C:\\Users\\RAJ\\OneDrive\\Desktop\\project_env\\data\\clean_data.csv')
user_engagement = pd.read_csv('C:\\Users\\RAJ\\OneDrive\\Desktop\\project_env\\data\\user_engagement.csv')
final_data = pd.read_csv('C:\\Users\\RAJ\\OneDrive\\Desktop\\project_env\\data\\final_data.csv')

model = joblib.load(open('C:\\Users\\RAJ\\OneDrive\\Desktop\\KNN_model.sav','rb'))

nav = st.sidebar.radio("Navigation", ["Data", "Visualization", "Prediction"])  
exp_eng = ['Engagement Score' , 'Experience Score']
scatter_data = final_data[exp_eng]
scatter_data['Engagement Score'] = scatter_data['Engagement Score'].abs()
cleaned_data = ['MSISDN/Number' , 'Social Media DL (Bytes)' , 'Social Media UL (Bytes)' ,
              'Google DL (Bytes)', 'Google UL (Bytes)', 'Email DL (Bytes)',
              'Email UL (Bytes)', 'Youtube DL (Bytes)', 'Youtube UL (Bytes)',
              'Netflix DL (Bytes)', 'Netflix UL (Bytes)', 'Gaming DL (Bytes)',
              'Gaming UL (Bytes)', 'Other DL (Bytes)', 'Other UL (Bytes)']
agg_app = data[cleaned_data]

agg_app.head()
agg_app['Total_Social_Media_Traffic'] = agg_app['Social Media DL (Bytes)'] + agg_app['Social Media UL (Bytes)']
agg_app['Total_Google_Traffic'] = agg_app['Google DL (Bytes)'] + agg_app['Google UL (Bytes)']
agg_app['Total_Email_Traffic'] = agg_app['Email DL (Bytes)'] + agg_app['Email UL (Bytes)']
agg_app['Total_Youtube_Traffic'] = agg_app['Youtube DL (Bytes)'] + agg_app['Youtube UL (Bytes)']
agg_app['Total_Netflix_Traffic'] = agg_app['Netflix DL (Bytes)'] + agg_app['Netflix UL (Bytes)']
agg_app['Total_Gaming_Traffic'] = agg_app['Gaming DL (Bytes)'] + agg_app['Gaming UL (Bytes)']
agg_app['Total_Other_Traffic'] = agg_app['Other DL (Bytes)'] + agg_app['Other UL (Bytes)']

#Aggregating user total traffic application wise
agg_app_traffic = agg_app.groupby('MSISDN/Number').agg(
    Total_Social_Media_Traffic = ('Total_Social_Media_Traffic' ,sum),
    Total_Google_Traffic = ('Total_Google_Traffic' ,sum),
    Total_Email_Traffic = ('Total_Email_Traffic' ,sum),
    Total_Youtube_Traffic = ('Total_Youtube_Traffic' ,sum),
    Total_Netflix_Traffic = ('Total_Netflix_Traffic' ,sum),
    Total_Gaming_Traffic = ('Total_Gaming_Traffic' ,sum),
    Total_Other_Traffic = ('Total_Other_Traffic' ,sum)
).reset_index()
#Total data consumption by Social Media
Total_data_volume_by_Social_Media = agg_app['Social Media DL (Bytes)'].sum() + agg_app['Social Media UL (Bytes)'].sum()

#Total data consumption by Email
Total_data_volume_by_Email = agg_app['Email DL (Bytes)'].sum() + agg_app['Email UL (Bytes)'].sum()

#Total data consumption by Youtube
Total_data_volume_by_Youtube = agg_app['Youtube DL (Bytes)'].sum() + agg_app['Youtube UL (Bytes)'].sum()

#Total data consumption by Netfilx
Total_data_volume_by_Netflix = agg_app['Netflix DL (Bytes)'].sum() + agg_app['Netflix UL (Bytes)'].sum()

#Total data consumption by Gaming
Total_data_volume_by_Gaming = agg_app['Gaming DL (Bytes)'].sum() + agg_app['Gaming UL (Bytes)'].sum()

#Total data consumption by Other
Total_data_volume_by_Other = agg_app['Other DL (Bytes)'].sum() + agg_app['Other UL (Bytes)'].sum()

#Total data consumption by Google
Total_data_volume_by_Google = agg_app['Google DL (Bytes)'].sum() + agg_app['Google UL (Bytes)'].sum()

Total_data_vol_by_each_app = [Total_data_volume_by_Social_Media,Total_data_volume_by_Youtube,
                          Total_data_volume_by_Netflix,Total_data_volume_by_Google,
                          Total_data_volume_by_Email,Total_data_volume_by_Gaming,Total_data_volume_by_Other]

total_data_df = pd.DataFrame({
    'Application': ['Social_Media', 'Youtube', 'Netflix', 'Google', 'Email', 'Gaming', 'Other'],
    'Total_data_volume_by_each_app': Total_data_vol_by_each_app
})

top_3_most_used_apps = total_data_df.nlargest(3, 'Total_data_volume_by_each_app')

#Splitting data into training and testing data
X = final_data[['Engagement Score', 'Experience Score']]
y = final_data['Satisfaction Score']

X_train, X_test ,y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)
X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
# Remove rows with NaN values
X_train.dropna(inplace=True)

y_train.dropna(inplace=True)

# Apply StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = KNeighborsRegressor()
model.fit(X_train,y_train)
prediction = model.predict(X_test)
joblib.dump(model,'satisfaction.model')
loaded_model = joblib.load('satisfaction.model')

if nav == "Data":
    st.image("C:\\Users\\RAJ\\OneDrive\\Desktop\\project_env\\image\\telco_logo.png", width = 500)
    st.subheader('This is our clean data')
    st.dataframe(data)
    st.subheader(' This is engagement data')
    st.dataframe(user_engagement)
    st.subheader('This is final data')
    st.dataframe(final_data)
if nav == "Visualization":
    Handset_count = data['Handset Type'].value_counts().head(10)
    fig = px.bar(Handset_count )
    
    
    fig.update_layout(
        title='Top 10 handset used by customers',
        xaxis_title='Handset name',
        yaxis_title='Count',
        template='plotly_dark' ,
        
    )
    st.plotly_chart(fig, use_container_width=True)
    
    #Create the bar chart
    fig1 = px.bar(x = data['Handset Manufacturer'].value_counts().head(3).index, y = data['Handset Manufacturer'].value_counts().head(3).values)   
    fig1.update_layout(
        title='Top 3 handset manufacture company',
        xaxis_title='Handset Manufacture',
        yaxis_title='Count',
        template='plotly_dark' ,
        width = 1200
    )
    st.plotly_chart(fig1, use_container_width=True)

#   st.pyplot(plt.gcf())
#Bar chart
    
    fig2 = px.bar(x =total_data_df['Application'],y=total_data_df['Total_data_volume_by_each_app'])
    fig2.update_layout(
        title='Total data volume by application ',
        xaxis_title='Application',
        yaxis_title='Total data volume',
        template='plotly_dark' ,
        width = 1200
    )
    st.plotly_chart(fig2, use_container_width=True)

#    st.pyplot(plt.gcf())

# Create the pie chart
    top_3_most_used_apps = total_data_df.nlargest(3, 'Total_data_volume_by_each_app')
    
    fig3 =px.pie(
        top_3_most_used_apps,
        names='Application',
        values='Total_data_volume_by_each_app',
        color='Application',  # Optional: color by application
        color_discrete_sequence=px.colors.qualitative.Plotly
        
    )
    fig3.update_layout(
      title='Top 3 app volume by application ',  
      template='plotly_dark' ,
      width = 1200
    )
    st.plotly_chart(fig3, use_container_width=True)

#Create scatter plot
    fig4 = px.scatter(
        scatter_data,
        x = 'Engagement Score',
        y = 'Experience Score'
        
    )
    fig4.update_layout(
        title = 'Relation between experience and engagement',
        template='plotly_dark',  # Optional: dark theme
        width=1200,  # Adjust width as needed
        
    )
    st.plotly_chart(fig4, use_container_width=True)
    
#Create bar chart
    top_10_satisfied_customers = final_data.nlargest(10 , 'Satisfaction Score')
    
    fig5 = px.bar(top_10_satisfied_customers,x='MSISDN/Number' , y='Satisfaction Score' )
    fig5.update_layout(
        title = 'Top 10 satisfied customers',
        template='plotly_dark',  # Optional: dark theme
        width=1200, 
    )
    st.plotly_chart(fig5, use_container_width=True)
if nav == "Prediction":
    st.subheader("Prediction")
    x = float(st.number_input("Engagement Score"))
    y = float(st.number_input("Experience Score"))
    btn = st.button("Predict")
    if btn:
        st.text('Satisfaction Score')
        input_features = np.array([x, y]).reshape(1, -1)
        pred = loaded_model.predict(input_features)
        st.subheader(pred)