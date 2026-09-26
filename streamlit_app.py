import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px

st.set_page_config(page_title="Nassau Candy Factory Optimization", layout="wide")

FACTORIES = {
    "Lot's O' Nuts": (32.881893, -111.768036),
    "Wicked Choccy's": (32.076176, -81.088371),
    "Sugar Shack": (48.11914, -96.18115),
    "Secret Factory": (41.446333, -90.565487),
    "The Other Factory": (35.1175, -89.971107)
}
PRODUCT_FACTORY = {
"Wonka Bar - Nutty Crunch Surprise":"Lot's O' Nuts","Wonka Bar - Fudge Mallows":"Lot's O' Nuts",
"Wonka Bar -Scrumdiddlyumptious":"Lot's O' Nuts","Wonka Bar - Milk Chocolate":"Wicked Choccy's",
"Wonka Bar - Triple Dazzle Caramel":"Wicked Choccy's","Laffy Taffy":"Sugar Shack",
"SweeTARTS":"Sugar Shack","Nerds":"Sugar Shack","Fun Dip":"Sugar Shack",
"Fizzy Lifting Drinks":"Sugar Shack","Everlasting Gobstopper":"Secret Factory",
"Hair Toffee":"The Other Factory","Lickable Wallpaper":"Secret Factory",
"Wonka Gum":"Secret Factory","Kazookles":"The Other Factory"
}
LAT = {"Alabama":32.8067,"Alaska":61.3707,"Arizona":33.7298,"Arkansas":34.9697,"California":36.1162,"Colorado":39.0598,"Connecticut":41.5978,"Delaware":39.3185,"Florida":27.7663,"Georgia":33.0406,"Hawaii":21.0943,"Idaho":44.2405,"Illinois":40.3495,"Indiana":39.8494,"Iowa":42.0115,"Kansas":38.5266,"Kentucky":37.6681,"Louisiana":31.1695,"Maine":44.6939,"Maryland":39.0639,"Massachusetts":42.2302,"Michigan":43.3266,"Minnesota":45.6945,"Mississippi":32.7416,"Missouri":38.4561,"Montana":47.0529,"Nebraska":41.1254,"Nevada":38.3135,"New Hampshire":43.4525,"New Jersey":40.2989,"New Mexico":34.8405,"New York":42.1657,"North Carolina":35.6301,"North Dakota":47.5289,"Ohio":40.3888,"Oklahoma":35.5653,"Oregon":44.5720,"Pennsylvania":40.5908,"Rhode Island":41.6809,"South Carolina":33.8569,"South Dakota":44.2998,"Tennessee":35.7478,"Texas":31.0545,"Utah":40.15,"Vermont":44.0459,"Virginia":37.7693,"Washington":47.4009,"West Virginia":38.4912,"Wisconsin":44.2685,"Wyoming":42.7559,"Ontario":43.7,"Quebec":46.8,"British Columbia":53.7,"Alberta":53.9,"Manitoba":50.6,"Saskatchewan":52.9,"New Brunswick":46.6,"Nova Scotia":44.7,"Newfoundland and Labrador":53.1,"Prince Edward Island":46.3,"Northwest Territories":61.1,"Yukon":64.3,"Nunavut":70.3}
LON = {"Alabama":-86.8,"Alaska":-152.4,"Arizona":-111.4,"Arkansas":-92.4,"California":-119.7,"Colorado":-105.5,"Connecticut":-72.7,"Delaware":-75.5,"Florida":-81.7,"Georgia":-83.6,"Hawaii":-157.5,"Idaho":-114.5,"Illinois":-88.99,"Indiana":-86.25,"Iowa":-93.2,"Kansas":-98.8,"Kentucky":-84.7,"Louisiana":-91.9,"Maine":-69.4,"Maryland":-76.8,"Massachusetts":-71.5,"Michigan":-84.5,"Minnesota":-94.6,"Mississippi":-89.7,"Missouri":-92.2,"Montana":-109.6,"Nebraska":-99.8,"Nevada":-117.1,"New Hampshire":-71.6,"New Jersey":-74.7,"New Mexico":-106.2,"New York":-74.9,"North Carolina":-79.8,"North Dakota":-99.8,"Ohio":-82.8,"Oklahoma":-97.5,"Oregon":-120.5,"Pennsylvania":-77.2,"Rhode Island":-71.5,"South Carolina":-80.9,"South Dakota":-99.4,"Tennessee":-86.7,"Texas":-99.9,"Utah":-111.9,"Vermont":-72.7,"Virginia":-78.2,"Washington":-120.5,"West Virginia":-80.7,"Wisconsin":-89.6,"Wyoming":-107.3,"Ontario":-79.4,"Quebec":-71.2,"British Columbia":-124.9,"Alberta":-114.1,"Manitoba":-97.0,"Saskatchewan":-106.0,"New Brunswick":-66.5,"Nova Scotia":-63.6,"Newfoundland and Labrador":-57.7,"Prince Edward Island":-63.1,"Northwest Territories":-117.0,"Yukon":-135.0,"Nunavut":-95.0}

def haversine(a,b,c,d):
    R=6371; p1,p2=np.radians(a),np.radians(c); dp=np.radians(c-a); dl=np.radians(d-b)
    x=np.sin(dp/2)**2+np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2*R*np.arcsin(np.sqrt(x))

@st.cache_data
def load_data(file):
    d=pd.read_csv(file)
    d["OrderDateParsed"]=pd.to_datetime(d["Order Date"],dayfirst=True,errors="coerce")
    d["ShipDateParsed"]=pd.to_datetime(d["Ship Date"],dayfirst=True,errors="coerce")
    d["LeadTimeDays"]=(d["ShipDateParsed"]-d["OrderDateParsed"]).dt.days
    d["CurrentFactory"]=d["Product Name"].map(PRODUCT_FACTORY)
    d["DestLat"]=d["State/Province"].map(LAT); d["DestLon"]=d["State/Province"].map(LON)
    d["DistanceKM"]=[haversine(FACTORIES[f][0],FACTORIES[f][1],la,lo) if f and pd.notna(la) and pd.notna(lo) else np.nan for f,la,lo in zip(d["CurrentFactory"],d["DestLat"],d["DestLon"])]
    d["OrderYear"]=d["OrderDateParsed"].dt.year; d["OrderMonth"]=d["OrderDateParsed"].dt.month
    return d

@st.cache_resource
def train_model(d):
    feats=["Product Name","Division","Region","Ship Mode","CurrentFactory","Units","Sales","Cost","DistanceKM","OrderYear","OrderMonth"]
    cat=["Product Name","Division","Region","Ship Mode","CurrentFactory"]; num=["Units","Sales","Cost","DistanceKM","OrderYear","OrderMonth"]
    x=d[feats]; y=d["LeadTimeDays"]; good=x.notna().all(axis=1)&y.notna()
    x=x.loc[good]; y=y.loc[good]
    xtr,xte,ytr,yte=train_test_split(x,y,test_size=.2,random_state=42)
    pre=ColumnTransformer([("cat",OneHotEncoder(handle_unknown="ignore"),cat),("num",StandardScaler(),num)])
    models={"Linear Regression":LinearRegression(),"Random Forest":RandomForestRegressor(n_estimators=160,random_state=42,n_jobs=-1,min_samples_leaf=3),"Gradient Boosting":GradientBoostingRegressor(random_state=42,n_estimators=160,max_depth=3,learning_rate=.05,loss="huber")}
    scores=[]; fitted={}
    for name,m in models.items():
        p=Pipeline([("prep",pre),("model",m)]); p.fit(xtr,ytr); pred=p.predict(xte)
        scores.append([name,mean_squared_error(yte,pred)**0.5,mean_absolute_error(yte,pred),r2_score(yte,pred)]); fitted[name]=p
    s=pd.DataFrame(scores,columns=["Model","RMSE","MAE","R2"]).sort_values("RMSE")
    return fitted[s.iloc[0].Model],s

uploaded=st.sidebar.file_uploader("Optional CSV",type=["csv"])
file=uploaded if uploaded else "Nassau_Candy_cleaned.csv"
df=load_data(file)
model,model_scores=train_model(df)

st.title("Factory Reallocation & Shipping Optimization")
st.caption("Nassau Candy Distributor • predictive scenario screening, not automatic execution")

if df["LeadTimeDays"].median()>90:
    st.warning("Data-quality warning: supplied Order Date → Ship Date gaps are unusually large. Validate date alignment before using recommendations operationally.")

c1,c2,c3,c4=st.columns(4)
c1.metric("Records",f"{len(df):,}")
c2.metric("Unique Orders",f"{df['Order ID'].nunique():,}")
c3.metric("Products",f"{df['Product Name'].nunique():,}")
c4.metric("Avg Lead Time",f"{df['LeadTimeDays'].mean():,.1f} days")

st.sidebar.header("Filters")
region=st.sidebar.selectbox("Region",["All"]+sorted(df.Region.dropna().unique()))
ship=st.sidebar.selectbox("Ship Mode",["All"]+sorted(df["Ship Mode"].dropna().unique()))
division=st.sidebar.selectbox("Division",["All"]+sorted(df.Division.dropna().unique()))
view=df.copy()
if region!="All": view=view[view.Region==region]
if ship!="All": view=view[view["Ship Mode"]==ship]
if division!="All": view=view[view.Division==division]

tabs=st.tabs(["Overview","Factory Simulator","Recommendations","Risk & Model"])
with tabs[0]:
    a,b=st.columns(2)
    with a:
        ch=view.groupby("Ship Mode").agg(Orders=("Order ID","count"),AvgLead=("LeadTimeDays","mean")).reset_index()
        st.plotly_chart(px.bar(ch,x="Ship Mode",y="Orders",title="Orders by Ship Mode"),use_container_width=True)
    with b:
        rh=view.groupby("Region").agg(Orders=("Order ID","count"),AvgLead=("LeadTimeDays","mean")).reset_index()
        st.plotly_chart(px.bar(rh,x="Region",y="AvgLead",title="Average Lead Time by Region"),use_container_width=True)
    st.dataframe(view.groupby(["CurrentFactory","Region"]).agg(Orders=("Order ID","count"),AvgLeadTime=("LeadTimeDays","mean"),Sales=("Sales","sum"),GrossProfit=("Gross Profit","sum")).reset_index(),use_container_width=True)

with tabs[1]:
    product=st.selectbox("Product",sorted(df["Product Name"].dropna().unique()))
    prod=df[df["Product Name"]==product].copy()
    row=prod.iloc[0]
    st.write(f"Current factory: **{row['CurrentFactory']}**")
    priority=st.slider("Optimization priority: Speed ↔ Profit preservation",0,100,75)
    scenarios=[]
    for f in FACTORIES:
        s=pd.DataFrame([{
            "Product Name":product,"Division":row["Division"],"Region":row["Region"],"Ship Mode":row["Ship Mode"],
            "CurrentFactory":f,"Units":prod["Units"].mean(),"Sales":prod["Sales"].mean(),"Cost":prod["Cost"].mean(),
            "DistanceKM":haversine(FACTORIES[f][0],FACTORIES[f][1],row["DestLat"],row["DestLon"]),
            "OrderYear":prod["OrderYear"].mode().iloc[0],"OrderMonth":prod["OrderMonth"].mode().iloc[0]
        }])
        pred=model.predict(s)[0]
        scenarios.append([f,pred])
    s=pd.DataFrame(scenarios,columns=["Factory","Predicted Lead Time"])
    current=float(s.loc[s.Factory==row["CurrentFactory"],"Predicted Lead Time"].iloc[0])
    s["Lead Reduction Days"]=current-s["Predicted Lead Time"]
    s["Lead Reduction %"]=100*s["Lead Reduction Days"]/current
    st.plotly_chart(px.bar(s,x="Factory",y="Predicted Lead Time",title="Predicted Lead Time by Factory"),use_container_width=True)
    st.dataframe(s.round(2),use_container_width=True)
    best=s.sort_values("Predicted Lead Time").iloc[0]
    if best.Factory!=row["CurrentFactory"]:
        st.info(f"Scenario shortlist: {best.Factory} has the lowest predicted lead time for this selected context. Validate capacity, freight cost and service levels before changing assignment.")
    else:
        st.success("The current factory has the lowest predicted lead time in this scenario.")

with tabs[2]:
    rec=pd.read_csv("factory_recommendations.csv")
    rec2=rec.copy()
    if region!="All":
        # recommendations are product-level; region is a filter only for the underlying context
        st.caption("Recommendations are product-level scenario outputs; region filtering affects the simulator and overview.")
    st.dataframe(rec2[["Product Name","CurrentFactory","ScenarioFactory","LeadReductionDays","LeadReductionPct","Recommendation","RiskFlag"]].round(2),use_container_width=True)
    st.plotly_chart(px.bar(rec2.sort_values("LeadReductionDays"),x="LeadReductionDays",y="Product Name",orientation="h",title="Modelled Lead-Time Change by Product"),use_container_width=True)

with tabs[3]:
    st.subheader("Model comparison")
    st.dataframe(model_scores.round(3),use_container_width=True)
    st.write(f"Selected model: **{model_scores.iloc[0].Model}**")
    st.metric("Recommendation coverage",f"{(rec.LeadReductionDays>0).mean()*100:.1f}%")
    st.metric("Positive-scenario average reduction",f"{rec.loc[rec.LeadReductionDays>0,'LeadReductionDays'].mean():.1f} days")
    st.warning("Profit impact is a proxy because the dataset does not contain factory-specific freight cost, capacity, inventory, or service-level constraints.")
