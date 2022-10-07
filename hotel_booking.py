from matplotlib.pyplot import subplots
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go 
import plotly.subplots as subplots
import pycountry as pc
import datetime

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression

st.set_page_config(
    page_title="Data Driven Hotel Marketing",
    page_icon="✅",
    layout="centered"
)

###################################################################################################################################################################
# Creating the Side Bar Navigation Panel
navigate = st.sidebar.radio('Navigation Side Bar',
                 ('Home Page', 'Overview', 'Customer Segmentation',
                  'Time Series Analysis', 'Cancelation Prediction'))

# Updating the Datset if needed
uploaded_file = st.file_uploader("Upload the Updated Dataset")
if uploaded_file is None:
    df = pd.read_csv("hotel_bookings.csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if st.button('Show a sample of the data'):
        st.write(df.head())
#####################################################################################################################################################################
        
# Creating the Home Page

if navigate == 'Home Page':
    # adding an aligned title without using CSS
    title_col = st.columns(4)
    title_col[0].title("Data")
    title_col[1].title("Driven")
    title_col[2].title("Hotel")
    title_col[3].title("Marketing")
    # adding the home page image
    image_col = st.columns(3)
    image_col[1].image("A-definitive-guide-to-hotel-booking-app-development_.png")
    # dashboard description
    st.header("Context")
    st.markdown("""Because the hospitality industry focuses heavily on creating experiences and relationships with customers and patrons, marketing that inspires customer loyalty while also reaching out to new customers is an important part of ensuring a member of the hospitality industry's success.
    The Aim of this Dashboard is to help the Marketing Agency in making data driven marketing decisons.
    """)
    # dataset info
    st.header("Dataset Information")
    st.markdown("This data is about hotels demand and bookings. One of the hotels (H1) is a resort hotel and the other is a city hotel (H2). This dataset includes 31 variables describing the 40,060 observations of H1 and 79,330 observations of H2. Each observation represents a hotel booking. It comprehends bookings due to arrive between the 1st of July of 2015 and the 31st of August 2017, including bookings that effectively arrived and bookings that were canceled. Since this is hotel real data, all data elements pertaining hotel or costumer identification were deleted. Due to the scarcity of real business data for scientific and educational purposes, this dataset can have an important role for research and education in revenue management, machine learning, or data mining, but most importantly hotel marketing.")

# Preparing the KPIs to use in the next page

# Percentage of Canceled Bookings

number_canceled_bookings = len(df[df['is_canceled'] == 1])/len(df['is_canceled'])
percentage_canceled_bookings = number_canceled_bookings * 100
percent_canceled_bookings = str(int(percentage_canceled_bookings)) + "%"

# Percentage of Successful Bookings

percentage_uncanceled_bookings = 100 - percentage_canceled_bookings
percent_uncanceled_bookings =  str(int(percentage_uncanceled_bookings)) + "%"

# Number of different hotel types

hotel_types = list(df['hotel'])
number_hotel_types = len(set(hotel_types))

# Create Booking Per Year bar chart 
df_not_canceled = df[df['is_canceled'] == 0]
bookings_15 = df_not_canceled.loc[df_not_canceled['arrival_date_year'] == 2015, 'hotel'].count()
bookings_16 = df_not_canceled.loc[df_not_canceled['arrival_date_year'] == 2016, 'hotel'].count()
bookings_17 = df_not_canceled.loc[df_not_canceled['arrival_date_year'] == 2017, 'hotel'].count()


bookings_data = {'Year': ['2015', '2016', '2017'], '# Bookings': [bookings_15, bookings_16, bookings_17]}
bookings_df = pd.DataFrame(bookings_data)
annual_bookings_bar_chart = px.bar(bookings_df, x='Year', y='# Bookings', color='Year', title='Annual Bookings')
annual_bookings_bar_chart['layout'].update(height=400, width=400)
annual_bookings_bar_chart.update_layout(showlegend=False)

# Bookings Count based on the Hotel Type Plot

custom_aggregation = {}
custom_aggregation["arrival_date_day_of_month"] = "count"
data2 = df_not_canceled.groupby("hotel").agg(custom_aggregation)
data2.columns = ["Booking Count"]
data2['Hotel'] = data2.index

bookings_hotel_count = px.bar(data2, x='Hotel', y="Booking Count", color="Hotel")
bookings_hotel_count['layout'].update(height=400, width=400, title='Bookings Count based on the Hotel Type', boxmode='group', showlegend=False)

# Sub Plot: Cancelation per hotel type, and Repeated Guest by Hotel Type

feature = ["hotel", 'is_canceled']
data2 = pd.crosstab(df[feature[0]], df[feature[1]])
data2['Hotel'] = data2.index


_0 = go.Bar(
            x = data2['Hotel'].index.values,
            y = data2[0],
            name='Uncanceled')

_1 = go.Bar(
            x = data2['Hotel'].index.values,
            y = data2[1],
            name='Is Canceled')


feature = ["hotel", 'is_repeated_guest']
data2 = pd.crosstab(df[feature[0]], df[feature[1]])
data2['Hotel'] = data2.index

_0a = go.Bar(
            x = data2['Hotel'].index.values,
            y = data2[0],
            name='New Guest')

_1a = go.Bar(
            x = data2['Hotel'].index.values,
            y = data2[1],
            name='Repeated Guest')


fig = subplots.make_subplots(rows=1, 
                          cols=2, 
                          #specs=[[{}, {}], [{'colspan': 1}, None]],
                          subplot_titles=('Is Canceled based on Hotel Type',
                                          'Guest based on Hotel Type'))

fig.append_trace(_0, 1, 1)
fig.append_trace(_1, 1, 1)

fig.append_trace(_0a, 1, 2)
fig.append_trace(_1a, 1, 2)

fig['layout'].update(height=600, width=800, title=' ', boxmode='group')


# Converting string month to numerical one (Dec = 12, Jan = 1, etc.)
df1 = df.copy()
datetime_object = df1['arrival_date_month'].str[0:3]
month_number = np.zeros(len(datetime_object))

# Creating a new column based on numerical representation of the months
for i in range(0, len(datetime_object)):
    datetime_object[i] = datetime.datetime.strptime(datetime_object[i], "%b")
    month_number[i] = datetime_object[i].month

# Float to integer conversion
month_number = pd.DataFrame(month_number).astype(int)

# 3 columns are merged into one
df1['arrival_date'] = df1['arrival_date_year'].map(str) + '-' + month_number[0].map(str) + '-' \
                       + df1['arrival_date_day_of_month'].map(str)

# Converting wrong datatype columns to correct type (object to datetime)
df1['arrival_date'] = pd.to_datetime(df1['arrival_date'])
df1['reservation_status_date'] = pd.to_datetime(df1['reservation_status_date'])

# Creating two dataframes include only discrete hotel type

dataResort = df1[df1['hotel'] == 'Resort Hotel']
dataCity = df1[df1['hotel'] == 'City Hotel']


dataResortMonthly = dataResort['arrival_date'].value_counts()
dataResortMonthly = dataResortMonthly.resample('m').sum().to_frame()

dataCityMonthly = dataCity['arrival_date'].value_counts()
dataCityMonthly = dataCityMonthly.resample('m').sum().to_frame()

# Booking By Hotel Type Line Chart

bookings_line_chart = go.Figure()
bookings_line_chart.add_trace(go.Scatter(x=dataResortMonthly.index, y=dataResortMonthly['arrival_date'], name="Resort Hotel",
                         hovertext=dataResortMonthly['arrival_date']))
bookings_line_chart.add_trace(go.Scatter(x=dataCityMonthly.index, y=dataCityMonthly['arrival_date'], name="City Hotel",
                         hovertext=dataCityMonthly['arrival_date']))
bookings_line_chart.update_layout(title_text='Number of Monthly Booking by Hotel',
                  title_x=0.5, title_font=dict(size=30))  # Location and the font size of the main title
bookings_line_chart.update_layout(
    xaxis_title="Arrival Date",
    yaxis_title="Number of Bookings", width = 800)
########################################################################################################################################################################
# Creating the Overview Page

if navigate == 'Overview':
    
    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Total Number of Bookings", value = len(df['hotel']))
    kpi_cols[1].metric("% of Canceled Bookings", value = percent_canceled_bookings)
    kpi_cols[2].metric("% of Successful Bookings", value = percent_uncanceled_bookings)
    kpi_cols[3].metric("Hotel Types", value = number_hotel_types)
    
    st.plotly_chart(bookings_line_chart)

    overview_visuals_1 = st.columns(2)
    overview_visuals_1[0].plotly_chart(annual_bookings_bar_chart)
    overview_visuals_1[1].plotly_chart(bookings_hotel_count)

    st.plotly_chart(fig)
########################################################################################################################################################################
# Creating the Time Series Analysis Page

# Creating the Time Series Plots

# Bookings time Series
bookings_time_series = subplots.make_subplots(rows=2, 
                    cols=1, 
                    subplot_titles=('Weekly Count of Bookings',
                                    'Monthly Count of Booking'))

custom_aggregation = {}
custom_aggregation["reservation_status_date"] = "count"
df = df.set_index(pd.DatetimeIndex(df['reservation_status_date']))
data2 = df.resample('W').agg(custom_aggregation)
data2.columns = ["Booking Count"]
data2['Date'] = data2.index

x = data2['Date'].tolist()
y = data2['Booking Count'].tolist()


bookings_time_series.add_trace(go.Scatter(x=x, y=y,name='Weekly Nb. of Book'), 1, 1)

custom_aggregation = {}
custom_aggregation["reservation_status_date"] = "count"
df = df.set_index(pd.DatetimeIndex(df['reservation_status_date']))
data2 = df.resample('M').agg(custom_aggregation)
data2.columns = ["Booking Count"]
data2['Date'] = data2.index

x = data2['Date'].tolist()
y = data2['Booking Count'].tolist()

bookings_time_series.add_trace(go.Scatter(x=x, y=y, mode='lines+markers',name='Monthly Nb. of Book'), 2, 1)

bookings_time_series['layout'].update(height=700, width=800, title='Booking Time Series')

# Guest Arrival Time Series

guest_arrival_time_series = subplots.make_subplots(rows=2, 
                    cols=1, 
                    subplot_titles=('Weekly Count of Guest Arrival by Day',
                                    'Monthly Count of Guest Arrival by Day'))

custom_aggregation = {}
custom_aggregation["arrival_date_day_of_month"] = "count"
df = df.set_index(pd.DatetimeIndex(df['reservation_status_date']))
data2 = df.resample('W').agg(custom_aggregation)
data2.columns = ["Nb. of Arrival Guest"]
data2['Date'] = data2.index

x = data2['Date'].tolist()
y = data2['Nb. of Arrival Guest'].tolist()


guest_arrival_time_series.add_trace(go.Scatter(x=x, y=y,name='Weekly Nb. of Arrival Guest'), 1, 1)

custom_aggregation = {}
custom_aggregation["arrival_date_day_of_month"] = "count"
df = df.set_index(pd.DatetimeIndex(df['reservation_status_date']))
data2 = df.resample('M').agg(custom_aggregation)
data2.columns = ["Nb. of Arrival Guest"]
data2['Date'] = data2.index

x = data2['Date'].tolist()
y = data2['Nb. of Arrival Guest'].tolist()

guest_arrival_time_series.add_trace(go.Scatter(x=x, y=y, mode='lines+markers',name='Monthly Nb. of Arrival Guest'), 2, 1)

guest_arrival_time_series['layout'].update(height=700, width=800, title='Arrival Guest Time Series')

# Average Daily Rate Time Series

adr_time_series = subplots.make_subplots(rows=2, 
                    cols=1, 
                    subplot_titles=('Weekly Average Rate',
                                    'Monthly Average Rate'))

custom_aggregation = {}
custom_aggregation["adr"] = "mean"
df = df.set_index(pd.DatetimeIndex(df['reservation_status_date']))
data2 = df.resample('W').agg(custom_aggregation)
data2.columns = ["ADR"]
data2['Date'] = data2.index

x = data2['Date'].tolist()
y = data2['ADR'].tolist()


adr_time_series.add_trace(go.Scatter(x=x, y=y, name='Weekly Average Rate'), 1, 1)

custom_aggregation = {}
custom_aggregation["adr"] = "mean"
df = df.set_index(pd.DatetimeIndex(df['reservation_status_date']))
data2 = df.resample('M').agg(custom_aggregation)
data2.columns = ["ADR"]
data2['Date'] = data2.index

x = data2['Date'].tolist()
y = data2['ADR'].tolist()

adr_time_series.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name='Monthly Average Rate'), 2, 1)

adr_time_series['layout'].update(height=700, width=800, title='Average Rate Time Series')

# Total Stays Time Series

total_stays_time_series = subplots.make_subplots(rows=2, 
                    cols=1, 
                    subplot_titles=('Weekly Total Stays',
                                    'Monthly Total Stays'))

custom_aggregation = {}
custom_aggregation["stays_in_weekend_nights"] = "sum"
custom_aggregation["stays_in_week_nights"] = "sum"
df = df.set_index(pd.DatetimeIndex(df['reservation_status_date']))
data2 = df.resample('W').agg(custom_aggregation)
data2.columns = ["Stays Weekend", 'Stays Weekdays']
data2['Date'] = data2.index

x = data2['Date'].tolist()
y = data2['Stays Weekend'].tolist()
z = data2['Stays Weekdays'].tolist()

total_stays_time_series.add_trace(go.Scatter(x=x, y=y, name='Weekly Stays in Weekend'), 1, 1)
total_stays_time_series.add_trace(go.Scatter(x=x, y=z, name='Weekly Stays in Weekdays'), 1, 1)

custom_aggregation = {}
custom_aggregation["stays_in_weekend_nights"] = "sum"
custom_aggregation["stays_in_week_nights"] = "sum"
df = df.set_index(pd.DatetimeIndex(df['reservation_status_date']))
data2 = df.resample('M').agg(custom_aggregation)
data2.columns = ["Stays Weekend", 'Stays Weekdays']
data2['Date'] = data2.index

x = data2['Date'].tolist()
y = data2['Stays Weekend'].tolist()
z = data2['Stays Weekdays'].tolist()

total_stays_time_series.add_trace(go.Scatter(x=x, y=y, name='Monthly Stays in Weekend', mode='lines+markers'), 2, 1)
total_stays_time_series.add_trace(go.Scatter(x=x, y=z, name='Monthly Stays in Weekdays', mode='lines+markers'), 2, 1)


total_stays_time_series['layout'].update(height=700, width=800, title='Total Stays Time Series')

if navigate == "Time Series Analysis":
    st.markdown("<h3 style='text-align: center; color: black;'>Time Series Analysis Across Multiple Fields</h1>", unsafe_allow_html=True)
    select_group = st.selectbox("Select Field", ('Booking', 'Guest Arrival', 'Average Daily Rate', 'Total Stays'))
    if select_group == 'Booking':
        st.plotly_chart(bookings_time_series)
    elif select_group == 'Guest Arrival':
        st.plotly_chart(guest_arrival_time_series)
    elif select_group == 'Average Daily Rate':
        st.plotly_chart(adr_time_series)
    elif select_group == 'Total Stays':
        st.plotly_chart(total_stays_time_series)
######################################################################################################################################################################

# Creating the KPIs for segmentation page

# Top 10 countries

# Creating a function to get the percentage of Booking by  Each Country

def get_count(series, limit=None):
    
    '''
    INPUT:
        series: Pandas Series (Single Column from DataFrame)
        limit:  If value given, limit the output value to first limit samples.
    OUTPUT:
        x = Unique values
        y = Count of unique values
    '''
    
    if limit != None:
        series = series.value_counts()[:limit]
    else:
        series = series.value_counts()
    
    x = series.index
    y = series/series.sum()*100
    
    return x.values,y.values

x,y = get_count(df_not_canceled['country'], limit=10)

# For each country code convert it to the country name 

country_name = [pc.countries.get(alpha_3=name).name for name in x]

countries_counts = df_not_canceled['country'].value_counts()
countries_codes = countries_counts.index

# Creating the Top 10 Countries Bar Chart 

countries_data = {'Country': country_name, 'Bookings (%)': y}
countries_df = pd.DataFrame(countries_data)
countries_bar_chart = px.bar(countries_df, x='Country', y='Bookings (%)', color='Country', title='Bookings By Top 10 Countries')
countries_bar_chart.update_layout(title_x=0.5, title_font=dict(size=22), showlegend=False, width = 600)

# Creating the Countries Map

country_freq = df_not_canceled['country'].value_counts().to_frame()
country_freq.columns = ['count']
country_map = px.choropleth(country_freq, color='count',
                    locations=country_freq.index,
                    hover_name=country_freq.index,
                    color_continuous_scale=px.colors.sequential.Teal)
country_map.update_traces(marker=dict(line=dict(color='#000000', width=1)))
country_map.update_layout(title_text='Number of Bookings by Countries',
                  title_x=0.5, title_font=dict(size=22), width = 600)


# Customer Category KPI

customer_categories = df1['customer_type'].value_counts()
customer_top_category = customer_categories.index[0]

# Customer Categories Line Chart 

df1['Total Guests'] = df1['adults'] + df1['children']

customerTransient = df1[df1['customer_type'] == 'Transient']
customerContract = df1[df1['customer_type'] == 'Contract']
customerTransientParty = df1[df1['customer_type'] == 'Transient-Party']
customerGroup = df1[df1['customer_type'] == 'Group']

customerTransient = customerTransient.set_index("arrival_date")
customerContract = customerContract.set_index("arrival_date")
customerTransientParty = customerTransientParty.set_index("arrival_date")
customerGroup = customerGroup.set_index("arrival_date")

customerTransientMonthly = customerTransient.resample('m').sum()
customerContract = customerContract.resample('m').sum()
customerTransientParty = customerTransientParty.resample('m').sum()
customerGroup = customerGroup.resample('m').sum()

customer_categories_line_chart = go.Figure()
customer_categories_line_chart.add_trace(go.Scatter(x=customerTransientMonthly.index, y=customerTransientMonthly['Total Guests'],
                         name="Transient Guests",
                         ))
customer_categories_line_chart.add_trace(go.Scatter(x=customerContract.index, y=customerContract['Total Guests'],
                         name="Contract Guests",
                         ))
customer_categories_line_chart.add_trace(go.Scatter(x=customerTransientParty.index, y=customerTransientParty['Total Guests'],
                         name="Transient-Party Guests",
                         ))
customer_categories_line_chart.add_trace(go.Scatter(x=customerGroup.index, y=customerGroup['Total Guests'],
                         name="Group Guests",
                         ))
customer_categories_line_chart.update_layout(title_text='Bookings by Customer Type',
                  title_x=0.5, title_font=dict(size=30))  # Location and the font size of the main title
customer_categories_line_chart.update_layout(
    xaxis_title="Arrival Date",
    yaxis_title="Bookings", 
    width = 750, height = 500)

####################################################################################################################################################################
# Creating the Market Segmentation Page 

if navigate == "Customer Segmentation":
    st.markdown("<h3 style='text-align: center; color: black;'>Customer Segmentation</h1>", unsafe_allow_html=True)
    select_segmentation = st.selectbox("Select Segmentation type", ('Geographic Segmentation', 'Guest Category Segmentation'))
    if select_segmentation == 'Geographic Segmentation':
        country_col = st.columns(2)
        country_col[0].metric("# Unique Countries", value = len(countries_codes))
        country_col[1].metric("Top Country", value = country_name[0])
        #geo_cols = st.columns(2)
        #geo_cols[0].plotly_chart(countries_bar_chart)
        #geo_cols[1].plotly_chart(country_map)
        st.plotly_chart(countries_bar_chart)
        st.plotly_chart(country_map)
    elif select_segmentation == 'Guest Category Segmentation':
        category_cols = st.columns(2)
        category_cols[0].metric("Guest Categories", value = len(df['customer_type'].value_counts()))
        category_cols[1].metric("Top Category", value = customer_top_category)
        st.plotly_chart(customer_categories_line_chart)
###################################################################################################################################################################
        
# Creating the Cancelation Prediction Page 

if navigate == "Cancelation Prediction":
    
    #identify the features and the target
    X = df[['lead_time', 'total_of_special_requests', 
               'booking_changes', 'hotel', 'assigned_room_type',
               'market_segment', 'distribution_channel', 'deposit_type']]
    y = df['is_canceled']
    
    #identify the numerical and categorical features
    num_vars = X.select_dtypes(include=['float', 'int']).columns.tolist()
    cat_vars = X.select_dtypes(include=['object']).columns.tolist()
    
    #Defining categorical features and numerical features separately
    X_cat = X[[c for c in X.columns if c in cat_vars]]
    X_num = X[[c for c in X.columns if c in num_vars]]
    
    #splitting the data  into train and test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, shuffle = True)
    
    # deal with missing values in categorical columns
    # fill these with 'Missing'
    def fill_cat_na(X):
        cat_vars_with_na = [col for col in X if (X[col].dtypes == "object") & (X[col].isna().sum() > 0)]
        for var in cat_vars_with_na:
            X[var] = X[var].fillna('Missing')
        return X
    fill_cat = FunctionTransformer(fill_cat_na, validate=False)
    
    #Encoding categorical features with OneHotEncoding
    cat_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore', drop = 'first')
    
    #Pipeline for categorical features
    cat_pipeline = Pipeline([
        ('imputer', fill_cat),
        ('encoding', cat_encoder),
    ])
    
    #Defining lower and upper limits to detect outliers
    def outliers(col):
        q1 = np.nanpercentile(col, 25)
        q3 = np.nanpercentile(col, 75)
        iqr = q3 - q1
        lower = q1 - (1.5 * iqr)
        upper = q3 + (1.5 * iqr)
        return lower , upper
    
    #Imputing missing values and outliers in the numerical columns with the median of each
    from numpy import percentile
    def impute_outliers_and_missing(X):
        num_vars = X.select_dtypes(include=['float', 'int']).columns.tolist() 
        for var in num_vars:
            median = X[var].median()
            X[var].fillna(median, inplace=True) 
            lower, upper = outliers(X[var])
            outliers_list = [x for x in X[var] if x < lower or x > upper]
            X[var][X[var].isin(outliers_list)] = median
        return X

    outliers_and_missing_impute = FunctionTransformer(impute_outliers_and_missing, validate=False)

    #Pipeline for numerical features
    num_pipeline = Pipeline([
        ('imputer', outliers_and_missing_impute),
        ('scaler', StandardScaler())])
    #Pipeline to apply on all columns
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_vars),
        ("cat", cat_pipeline, cat_vars),
    ])
    #Logistic Regression Model
    log_reg = LogisticRegression()

    pipeline = Pipeline(steps=[('i', full_pipeline), ('m', log_reg)])

    def display_scores(scores):
        print("Scores:", scores)
        print(f'Mean: {scores.mean():.3f}')
        print(f'Stdev: {scores.std():.3f}')
    
    # Train the model
    pipeline.fit(X_train,y_train)
    
    # Web App
    st.markdown("<h1 style='text-align: center; color: black;'>Reservation Cancelation Prediction</h1>", unsafe_allow_html=True)
    st.markdown("This Simple & Fast Reservation Cancelation Prediction tool will help the hotel in predicting the guest behavior with 80% accurecy.")
    st.markdown("Instructions: Simply change the values of the below predictors according to your preference and you're all set up")
    # User Data Entry
    def user_report():
        hotel = st.selectbox('Hotel Type',('City Hotel','Resort Hotel'))
        lead_time= st.slider('Lead Time',min_value =0,max_value=100,step=1)
        market_segment = st.selectbox("Market Segment", ('Online TA', 'Offline TA/TO', 'Groups', 'Direct', 'Corporate', 'Complementary', 'Aviation', 'Undefined'))
        distribution_channel = st.selectbox("Distribution Channel", ('TA/TO', 'Direct', 'Corporate', 'GDS', 'Undefined'))
        assigned_room_type = st.selectbox("Room Type", ("A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "P"))
        booking_changes= st.slider('Booking Changes',min_value=0,max_value=20,step =1)
        deposit_type = st.selectbox("Deposit Type", ('No Deposit', 'Non Refund', 'Refundable'))
        total_of_special_requests   = st.slider('Tota Special Requests',min_value=0,max_value=5,step=1)
        
        user_report_data = {
        'hotel': hotel,
        'lead_time': lead_time,
        'market_segment': market_segment,
        'distribution_channel': distribution_channel,
        'assigned_room_type': assigned_room_type,
        'booking_changes': booking_changes,
        'deposit_type' : deposit_type,
        'total_of_special_requests': total_of_special_requests
        }
        report_data =pd.DataFrame(user_report_data, index = [0])
        return report_data
    user_data = user_report()
    
    st.header('Guest Data')
    st.table(user_data)
    
    # Predict the result
    prediction = pipeline.predict(user_data)
    st.subheader('The Guest is more likely to: ')
    if prediction == 0:
        st.subheader("Keep the reservation")
    else:
        st.subheader("Cancel")
