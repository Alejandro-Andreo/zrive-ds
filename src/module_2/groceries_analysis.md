```python
%reset -f
```

# Importing data

Data has been stored in local using aws cli. 


```python
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from pathlib import Path
from IPython.display import Markdown, display

DATA_PATH = "/home/alejandro/Zrive"

abandoned_carts_df = pd.read_parquet(Path(DATA_PATH, "abandoned_carts.parquet"))
inventory_df = pd.read_parquet(Path(DATA_PATH, "inventory.parquet"))
orders_df = pd.read_parquet(Path(DATA_PATH, "orders.parquet"))
regulars_df = pd.read_parquet(Path(DATA_PATH, "regulars.parquet"))
users_df = pd.read_parquet(Path(DATA_PATH, "users.parquet"))

feature_frame = pd.read_csv(Path(DATA_PATH, "feature_frame.csv"))


def printmd(string):
    display(Markdown(string))
```

# Part 1: Understanding the problem space

The data is partitioned over multiple dataset and comes from a groceries ecommerce
platform selling products directly to consumers (think of it as an online
supermarket):

**orders.parquet**: An order history of customers. Each row is an order and the
item_ids for the order are stored as a list in the item_ids column

**regulars.parquet**: Users are allowed to specify items that they wish to buy
regularly. This data gives the items each user has asked to get regularly, along
with when they input that information.

**abandoned_cart.parquet**: If a user has added items to their basket but not
bought them, we capture that information. Items that were abandoned are stored
as a list in item_ids.

**inventory.parquet**: Some information about each item_id

**users.parquet**: Information about users.


```python
# Quick check on each dataset to understand their structure and identify potential issues
datasets = {
    "Abandoned Carts": abandoned_carts_df,
    "Inventory": inventory_df,
    "Orders": orders_df,
    "Regulars": regulars_df,
    "Users": users_df,
}

# Displaying the first few rows of each dataset and their info to identify potential issues
for name, df in datasets.items():
    printmd(f"# Dataset: {name}")
    display(df.head())
    df.info()
    print("\nMising values(NAs):")
    display(df.isna().sum())
```


# Dataset: Abandoned Carts



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>user_id</th>
      <th>created_at</th>
      <th>variant_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12858560217220</td>
      <td>5c4e5953f13ddc3bc9659a3453356155e5efe4739d7a2b...</td>
      <td>2020-05-20 13:53:24</td>
      <td>[33826459287684, 33826457616516, 3366719212762...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>20352449839236</td>
      <td>9d6187545c005d39e44d0456d87790db18611d7c7379bd...</td>
      <td>2021-06-27 05:24:13</td>
      <td>[34415988179076, 34037940158596, 3450282236326...</td>
    </tr>
    <tr>
      <th>45</th>
      <td>20478401413252</td>
      <td>e83fb0273d70c37a2968fee107113698fd4f389c442c0b...</td>
      <td>2021-07-18 08:23:49</td>
      <td>[34543001337988, 34037939372164, 3411360609088...</td>
    </tr>
    <tr>
      <th>50</th>
      <td>20481783103620</td>
      <td>10c42e10e530284b7c7c50f3a23a98726d5747b8128084...</td>
      <td>2021-07-18 21:29:36</td>
      <td>[33667268116612, 34037940224132, 3443605520397...</td>
    </tr>
    <tr>
      <th>52</th>
      <td>20485321687172</td>
      <td>d9989439524b3f6fc4f41686d043f315fb408b954d6153...</td>
      <td>2021-07-19 12:17:05</td>
      <td>[33667268083844, 34284950454404, 33973246886020]</td>
    </tr>
  </tbody>
</table>
</div>


    <class 'pandas.core.frame.DataFrame'>
    Index: 5457 entries, 0 to 70050
    Data columns (total 4 columns):
     #   Column      Non-Null Count  Dtype         
    ---  ------      --------------  -----         
     0   id          5457 non-null   int64         
     1   user_id     5457 non-null   object        
     2   created_at  5457 non-null   datetime64[us]
     3   variant_id  5457 non-null   object        
    dtypes: datetime64[us](1), int64(1), object(2)
    memory usage: 213.2+ KB
    
    Mising values(NAs):



    id            0
    user_id       0
    created_at    0
    variant_id    0
    dtype: int64



# Dataset: Inventory



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variant_id</th>
      <th>price</th>
      <th>compare_at_price</th>
      <th>vendor</th>
      <th>product_type</th>
      <th>tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39587297165444</td>
      <td>3.09</td>
      <td>3.15</td>
      <td>heinz</td>
      <td>condiments-dressings</td>
      <td>[table-sauces, vegan]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>34370361229444</td>
      <td>4.99</td>
      <td>5.50</td>
      <td>whogivesacrap</td>
      <td>toilet-roll-kitchen-roll-tissue</td>
      <td>[b-corp, eco, toilet-rolls]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>34284951863428</td>
      <td>3.69</td>
      <td>3.99</td>
      <td>plenty</td>
      <td>toilet-roll-kitchen-roll-tissue</td>
      <td>[kitchen-roll]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33667283583108</td>
      <td>1.79</td>
      <td>1.99</td>
      <td>thecheekypanda</td>
      <td>toilet-roll-kitchen-roll-tissue</td>
      <td>[b-corp, cruelty-free, eco, tissue, vegan]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>33803537973380</td>
      <td>1.99</td>
      <td>2.09</td>
      <td>colgate</td>
      <td>dental</td>
      <td>[dental-accessories]</td>
    </tr>
  </tbody>
</table>
</div>


    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1733 entries, 0 to 1732
    Data columns (total 6 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   variant_id        1733 non-null   int64  
     1   price             1733 non-null   float64
     2   compare_at_price  1733 non-null   float64
     3   vendor            1733 non-null   object 
     4   product_type      1733 non-null   object 
     5   tags              1733 non-null   object 
    dtypes: float64(2), int64(1), object(3)
    memory usage: 81.4+ KB
    
    Mising values(NAs):



    variant_id          0
    price               0
    compare_at_price    0
    vendor              0
    product_type        0
    tags                0
    dtype: int64



# Dataset: Orders



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>user_id</th>
      <th>created_at</th>
      <th>order_date</th>
      <th>user_order_seq</th>
      <th>ordered_items</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>2204073066628</td>
      <td>62e271062eb827e411bd73941178d29b022f5f2de9d37f...</td>
      <td>2020-04-30 14:32:19</td>
      <td>2020-04-30</td>
      <td>1</td>
      <td>[33618849693828, 33618860179588, 3361887404045...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2204707520644</td>
      <td>bf591c887c46d5d3513142b6a855dd7ffb9cc00697f6f5...</td>
      <td>2020-04-30 17:39:00</td>
      <td>2020-04-30</td>
      <td>1</td>
      <td>[33618835243140, 33618835964036, 3361886244058...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2204838822020</td>
      <td>329f08c66abb51f8c0b8a9526670da2d94c0c6eef06700...</td>
      <td>2020-04-30 18:12:30</td>
      <td>2020-04-30</td>
      <td>1</td>
      <td>[33618891145348, 33618893570180, 3361889766618...</td>
    </tr>
    <tr>
      <th>34</th>
      <td>2208967852164</td>
      <td>f6451fce7b1c58d0effbe37fcb4e67b718193562766470...</td>
      <td>2020-05-01 19:44:11</td>
      <td>2020-05-01</td>
      <td>1</td>
      <td>[33618830196868, 33618846580868, 3361891234624...</td>
    </tr>
    <tr>
      <th>49</th>
      <td>2215889436804</td>
      <td>68e872ff888303bff58ec56a3a986f77ddebdbe5c279e7...</td>
      <td>2020-05-03 21:56:14</td>
      <td>2020-05-03</td>
      <td>1</td>
      <td>[33667166699652, 33667166699652, 3366717122163...</td>
    </tr>
  </tbody>
</table>
</div>


    <class 'pandas.core.frame.DataFrame'>
    Index: 8773 entries, 10 to 64538
    Data columns (total 6 columns):
     #   Column          Non-Null Count  Dtype         
    ---  ------          --------------  -----         
     0   id              8773 non-null   int64         
     1   user_id         8773 non-null   object        
     2   created_at      8773 non-null   datetime64[us]
     3   order_date      8773 non-null   datetime64[us]
     4   user_order_seq  8773 non-null   int64         
     5   ordered_items   8773 non-null   object        
    dtypes: datetime64[us](2), int64(2), object(2)
    memory usage: 479.8+ KB
    
    Mising values(NAs):



    id                0
    user_id           0
    created_at        0
    order_date        0
    user_order_seq    0
    ordered_items     0
    dtype: int64



# Dataset: Regulars



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>variant_id</th>
      <th>created_at</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>68e872ff888303bff58ec56a3a986f77ddebdbe5c279e7...</td>
      <td>33618848088196</td>
      <td>2020-04-30 15:07:03</td>
    </tr>
    <tr>
      <th>11</th>
      <td>aed88fc0b004270a62ff1fe4b94141f6b1db1496dbb0c0...</td>
      <td>33667178659972</td>
      <td>2020-05-05 23:34:35</td>
    </tr>
    <tr>
      <th>18</th>
      <td>68e872ff888303bff58ec56a3a986f77ddebdbe5c279e7...</td>
      <td>33619009208452</td>
      <td>2020-04-30 15:07:03</td>
    </tr>
    <tr>
      <th>46</th>
      <td>aed88fc0b004270a62ff1fe4b94141f6b1db1496dbb0c0...</td>
      <td>33667305373828</td>
      <td>2020-05-05 23:34:35</td>
    </tr>
    <tr>
      <th>47</th>
      <td>4594e99557113d5a1c5b59bf31b8704aafe5c7bd180b32...</td>
      <td>33667247341700</td>
      <td>2020-05-06 14:42:11</td>
    </tr>
  </tbody>
</table>
</div>


    <class 'pandas.core.frame.DataFrame'>
    Index: 18105 entries, 3 to 37720
    Data columns (total 3 columns):
     #   Column      Non-Null Count  Dtype         
    ---  ------      --------------  -----         
     0   user_id     18105 non-null  object        
     1   variant_id  18105 non-null  int64         
     2   created_at  18105 non-null  datetime64[us]
    dtypes: datetime64[us](1), int64(1), object(1)
    memory usage: 565.8+ KB
    
    Mising values(NAs):



    user_id       0
    variant_id    0
    created_at    0
    dtype: int64



# Dataset: Users



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>user_segment</th>
      <th>user_nuts1</th>
      <th>first_ordered_at</th>
      <th>customer_cohort_month</th>
      <th>count_people</th>
      <th>count_adults</th>
      <th>count_children</th>
      <th>count_babies</th>
      <th>count_pets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2160</th>
      <td>0e823a42e107461379e5b5613b7aa00537a72e1b0eaa7a...</td>
      <td>Top Up</td>
      <td>UKH</td>
      <td>2021-05-08 13:33:49</td>
      <td>2021-05-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1123</th>
      <td>15768ced9bed648f745a7aa566a8895f7a73b9a47c1d4f...</td>
      <td>Top Up</td>
      <td>UKJ</td>
      <td>2021-11-17 16:30:20</td>
      <td>2021-11-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1958</th>
      <td>33e0cb6eacea0775e34adbaa2c1dec16b9d6484e6b9324...</td>
      <td>Top Up</td>
      <td>UKD</td>
      <td>2022-03-09 23:12:25</td>
      <td>2022-03-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>675</th>
      <td>57ca7591dc79825df0cecc4836a58e6062454555c86c35...</td>
      <td>Top Up</td>
      <td>UKI</td>
      <td>2021-04-23 16:29:02</td>
      <td>2021-04-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4694</th>
      <td>085d8e598139ce6fc9f75d9de97960fa9e1457b409ec00...</td>
      <td>Top Up</td>
      <td>UKJ</td>
      <td>2021-11-02 13:50:06</td>
      <td>2021-11-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


    <class 'pandas.core.frame.DataFrame'>
    Index: 4983 entries, 2160 to 3360
    Data columns (total 10 columns):
     #   Column                 Non-Null Count  Dtype  
    ---  ------                 --------------  -----  
     0   user_id                4983 non-null   object 
     1   user_segment           4983 non-null   object 
     2   user_nuts1             4932 non-null   object 
     3   first_ordered_at       4983 non-null   object 
     4   customer_cohort_month  4983 non-null   object 
     5   count_people           325 non-null    float64
     6   count_adults           325 non-null    float64
     7   count_children         325 non-null    float64
     8   count_babies           325 non-null    float64
     9   count_pets             325 non-null    float64
    dtypes: float64(5), object(5)
    memory usage: 428.2+ KB
    
    Mising values(NAs):



    user_id                     0
    user_segment                0
    user_nuts1                 51
    first_ordered_at            0
    customer_cohort_month       0
    count_people             4658
    count_adults             4658
    count_children           4658
    count_babies             4658
    count_pets               4658
    dtype: int64


## Check Orders 

With the orders dataframe, we could try to identify the purchasing patterns of customers. Both, the number of orders per user and the number of items per order are important metrics to understand the purchasing patterns of customers. 



Let's first check the frequency of orders per user.


```python
orders_frequency = (
    orders_df["user_id"]
    .value_counts()
    .reset_index()
    .rename(columns={"count": "order_count"})
)
display(orders_frequency["order_count"].describe())
plt.hist(
    orders_frequency["order_count"],
    bins=len(orders_frequency["order_count"].unique()),
    edgecolor="k",
)
plt.title("Distribution of Order Frequencies per User")
plt.xlabel("Number of Orders")
plt.ylabel("Number of Users")
plt.xlim(min(orders_frequency["order_count"] - 1), max(orders_frequency["order_count"]))
plt.grid(axis="y", alpha=0.75)
plt.show()
```


    count    4983.000000
    mean        1.760586
    std         1.936537
    min         1.000000
    25%         1.000000
    50%         1.000000
    75%         2.000000
    max        25.000000
    Name: order_count, dtype: float64



    
![png](groceries_analysis_files/groceries_analysis_9_1.png)
    


Most of users just make 1 order, which indicates that the platform has a large base of "one time" users. Let's now check the distribution of order sizes.


```python
# Calculate order size as the length of the ordered_items list for each order
orders_df["order_size"] = orders_df["ordered_items"].apply(len)
# Analyze the distribution of order sizes
order_size_distribution = orders_df["order_size"].describe()
display(order_size_distribution)
printmd(f"Most common order size: {orders_df['order_size'].mode()[0]}")
plt.hist(orders_df["order_size"], bins=50, edgecolor="k")
plt.title("Distribution of Orders Sizes")
plt.xlabel("Order Size (Number of Items)")
plt.ylabel("Frequency")
plt.show()
```


    count    8773.000000
    mean       12.305711
    std         6.839507
    min         1.000000
    25%         8.000000
    50%        11.000000
    75%        15.000000
    max       114.000000
    Name: order_size, dtype: float64



Most common order size: 10



    
![png](groceries_analysis_files/groceries_analysis_11_2.png)
    


As we can see, most orders have 10 items, but there are users who make orders with lots of items creating a long right tail. 
Let's now check the most ordered items. 
As each item has a unique id, we can't know what each item is. However, we can see that all items fall into a specific category, thus ,we can show the most ordered items by mapping its id to its category.


```python
item_frequencies = (
    orders_df.explode("ordered_items")["ordered_items"]
    .value_counts()
    .reset_index()
    .rename(columns={"ordered_items": "variant_id", "count": "item_count"})
)

# merge with inventory to get category information
common_items = pd.merge(item_frequencies, inventory_df, on="variant_id", how="inner")

# plot the top 10 most ordered items
plt.bar(
    common_items["variant_id"][:10].astype(str),
    common_items["item_count"][:10],
    edgecolor="k",
)
plt.title("Top 10 Most Ordered Items")
plt.xlabel("Item")
plt.ylabel("Frequency")
plt.xticks(
    ticks=range(0, 10),
    labels=common_items["product_type"][:10],
    rotation=45,
    ha="right",
)
plt.show()
```


    
![png](groceries_analysis_files/groceries_analysis_13_0.png)
    


There is a catch here, as we don't have the category of each item, when we merged the datasets, item_frequencies(n=2117) and inventory(n=1733), we lost some items. This is because the inventory dataset has less items than the orders dataset.


```python
# Determine the most commonly ordered categories
common_categories = (
    common_items.groupby("product_type")["item_count"]
    .sum()
    .reset_index()
    .sort_values("item_count", ascending=False)
)

plt.bar(
    x=common_categories["product_type"][:10],
    height=common_categories["item_count"][:10],
    edgecolor="k",
)
plt.title("Top 10 Most Ordered Categories")
plt.xlabel("Product Category")
plt.ylabel("Frecuency")
plt.xticks(rotation=45, ha="right")
plt.show()
```


    
![png](groceries_analysis_files/groceries_analysis_15_0.png)
    



```python
# Top 10 most ordered brands
common_brands = (
    common_items.groupby("vendor")["item_count"]
    .sum()
    .reset_index()
    .sort_values("item_count", ascending=False)
)

plt.bar(
    x=common_brands["vendor"][:10],
    height=common_categories["item_count"][:10],
    edgecolor="k",
)
plt.title("Top 10 Most Ordered Brands")
plt.xlabel("Product brand")
plt.ylabel("Frecuency")
plt.xticks(rotation=45, ha="right")
plt.show()
```


    
![png](groceries_analysis_files/groceries_analysis_16_0.png)
    



```python
# Determine the items and categories which bring the most revenue
# Calculate the revenue for each item by multiplying the price of each item by the quantity ordered
common_items["revenue"] = common_items["item_count"] * common_items["price"]
common_items = common_items.sort_values("revenue", ascending=False)

plt.bar(
    common_items["variant_id"][:10].astype(str),
    common_items["revenue"][:10],
    edgecolor="k",
)
plt.title("Top 10 Items by Revenue")
plt.xlabel("Item")
plt.ylabel("Revenue")
plt.xticks(
    ticks=range(0, 10),
    labels=common_items["product_type"][:10],
    rotation=45,
    ha="right",
)
plt.show()
```


    
![png](groceries_analysis_files/groceries_analysis_17_0.png)
    


As we can see, the product which produces the most revenue by far, is part of the milk category. This might indicate that the platform relies too much in selling this kind of milk-related product. To check this, let's compare the revenue of the top ten products against the revenue of the rest of the products.


```python
# Calculate the total revenue of the rest of the dataset
rest_revenue = common_items["revenue"][10:].sum()

# Create a new dataframe to hold the data for the bar plot
revenue_data = pd.DataFrame(
    {
        "Product": ["Top 10 Items", "Rest of Dataset"],
        "Revenue": [common_items["revenue"][:10].sum(), rest_revenue],
    }
)

# Plot the bar chart
plt.bar(revenue_data["Product"], revenue_data["Revenue"], edgecolor="k")
plt.title("Revenue Comparison")
plt.xlabel("Product")
plt.ylabel("Revenue")
plt.show()

# Calculate percentage of revenue from the top 10 items
top_10_revenue = common_items["revenue"][:10].sum()
total_revenue = common_items["revenue"].sum()
top_10_revenue_percentage = top_10_revenue / total_revenue * 100
printmd(
    f"Percentage of revenue from the top 10 items: {top_10_revenue_percentage:.2f}%"
)
# Calculate percentage of revenue from the rest of the dataset
rest_revenue = common_items["revenue"][10:].sum()
rest_revenue_percentage = rest_revenue / total_revenue * 100
printmd(
    f"Percentage of revenue from the rest of the dataset: {rest_revenue_percentage:.2f}%"
)
# Calculate percentage of revenue from the top product
top_product_revenue = common_items["revenue"][0]
top_product_revenue_percentage = top_product_revenue / total_revenue * 100
printmd(
    f"Percentage of revenue from the top product: {top_product_revenue_percentage:.2f}%"
)
```


    
![png](groceries_analysis_files/groceries_analysis_19_0.png)
    



Percentage of revenue from the top 10 items: 20.68%



Percentage of revenue from the rest of the dataset: 79.32%



Percentage of revenue from the top product: 9.88%


The platform's product portfolio doesn't appear to be well balanced in terms of its item revenues, with the top ten products accounting for 20% of the total revenue, while the other 1467 products account for the remaining 80% . Moreover, the top product represents 10% of the total revenue, which is a lot for a single product. Let's now check revenue by general categories.


```python
category_revenue = (
    common_items.groupby("product_type")["revenue"]
    .sum()
    .reset_index()
    .sort_values("revenue", ascending=False)
)
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].bar(
    x=category_revenue["product_type"],
    height=category_revenue["revenue"],
    edgecolor="k",
)
ax[0].set_title("Categories by Revenue")
ax[0].set_xlabel("Category")
ax[0].set_ylabel("Revenue")
ax[0].set_xticks(range(0, len(category_revenue)))
ax[0].set_xticklabels(category_revenue["product_type"], rotation=45, ha="right")


ax[1].bar(
    x=category_revenue["product_type"][:10],
    height=category_revenue["revenue"][:10],
    edgecolor="k",
)
ax[1].set_title("Top 10 Categories by Revenue")
ax[1].set_xlabel("Category")
ax[1].set_ylabel("Revenue")
ax[1].set_xticks(range(0, 10))
ax[1].set_xticklabels(category_revenue["product_type"][:10], rotation=45, ha="right")

plt.tight_layout()
plt.show()
```

    /tmp/ipykernel_4591/1866416911.py:7: UserWarning: set_ticklabels() should only be used with a fixed number of ticks, i.e. after set_ticks() or using a FixedLocator.
      ax[0].set_xticklabels(category_revenue['product_type'],rotation=45, ha='right')
    /tmp/ipykernel_4591/1866416911.py:14: UserWarning: set_ticklabels() should only be used with a fixed number of ticks, i.e. after set_ticks() or using a FixedLocator.
      ax[1].set_xticklabels(category_revenue['product_type'][:10],rotation=45, ha='right')



    
![png](groceries_analysis_files/groceries_analysis_21_1.png)
    


Key Insights:

- User Behavior: Most users make only one order, indicating a large base of one-time users. Strategies could be developed to encourage repeat purchases.

- Order Size: Most orders contain around 10 items, but there is a long tail of orders with many items. This suggests a diverse range of customer shopping behaviors.

- Most Ordered Items: The most ordered items fall into categories like milk substitutes, toilet & kitchen roll tissue, dishwashing products, and fabric softeners. These are everyday household items, suggesting that users are using the platform for their regular grocery shopping.

- Most Ordered Brands: 4 out 5 of the most ordered brands are ecofriendly/organic/vegan brands, which suggests that the platform's user base may be environmentally conscious.

- Revenue Concentration: The top 10 items account for over 20% of total revenue, and the top product alone accounts for nearly 10%. This indicates a high concentration of revenue in a small number of products.

- Product Portfolio Balance: The product portfolio may not be well balanced, with a small number of products accounting for a large proportion of revenue. Diversifying the product portfolio could help reduce risk and increase potential for growth.

- Category Revenue: The milk category generates the most revenue, suggesting that dairy products are a key area for the platform. However, reliance on a single category could pose a risk if demand changes or supply issues arise.

- These insights can help inform strategies for product assortment, marketing, and customer retention.



## Check Users

Let's know try to combine the regulars dataframe with the users dataframe, to understand better the consumers behavior.


```python
# Calculate number of regulars by user
regulars_by_user = (
    regulars_df.groupby("user_id")["variant_id"]
    .nunique()
    .reset_index()
    .rename(columns={"variant_id": "regulars_count"})
)
# Merge with users to get user information
users = pd.merge(users_df, regulars_by_user, on="user_id", how="left").fillna(
    {"regulars_count": 0}
)
users.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count_people</th>
      <th>count_adults</th>
      <th>count_children</th>
      <th>count_babies</th>
      <th>count_pets</th>
      <th>regulars_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>325.000000</td>
      <td>325.000000</td>
      <td>325.000000</td>
      <td>325.000000</td>
      <td>325.000000</td>
      <td>4983.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.787692</td>
      <td>2.003077</td>
      <td>0.707692</td>
      <td>0.076923</td>
      <td>0.636923</td>
      <td>2.481437</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.365753</td>
      <td>0.869577</td>
      <td>1.026246</td>
      <td>0.289086</td>
      <td>0.995603</td>
      <td>8.890588</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>8.000000</td>
      <td>7.000000</td>
      <td>6.000000</td>
      <td>2.000000</td>
      <td>6.000000</td>
      <td>320.000000</td>
    </tr>
  </tbody>
</table>
</div>



Most people don't have a regular list, which is expected, as it's most likely voluntary info. Let's check the number of regulars by user_segment


```python
# distribution of user_segment values
users["user_segment"].value_counts().plot.pie(y="user_segment", autopct="%1.1f%%")
```




    <Axes: ylabel='count'>




    
![png](groceries_analysis_files/groceries_analysis_27_1.png)
    



```python
# Distribution by user_segment of the number of regulars
users.groupby("user_segment").sum().plot.pie(y="regulars_count", autopct="%1.1f%%")
```




    <Axes: ylabel='regulars_count'>




    
![png](groceries_analysis_files/groceries_analysis_28_1.png)
    



```python
# Check the average number of regulars_count by user segment
users.groupby("user_segment")["regulars_count"].mean().plot.bar(edgecolor="k")
plt.title("Average Number of Regulars by User Segment")
plt.xlabel("User Segment")
plt.ylabel("Average Number of Regulars")
plt.show()
```


    
![png](groceries_analysis_files/groceries_analysis_29_0.png)
    



```python
(users["count_people"] > 1).sum() / len(users["count_people"].dropna())
```




    0.8184615384615385




```python
pd.crosstab(users["user_nuts1"], users["user_segment"]).plot.bar(
    stacked=True, edgecolor="k"
)
```




    <Axes: xlabel='user_nuts1'>




    
![png](groceries_analysis_files/groceries_analysis_31_1.png)
    


Insights:
- Most users don't have a regulars items, which is expected, as it's most likely voluntary info.
- The Users segments distribution is fairly similar: Top Up (53%), Proposition (47%)
- Users in the "Proposition" segment are more likely to have a regular list, which is expected, as they are more likely to be frequent users of the platform.
- From the users who have a regulars items, The proposition segment has the highest average number of regulars items (3.5), and the "Top Up" segment has an average of 1.5 regulars items.
- Most households who answered the survey live in a family(80%) while the rest live alone(20%).
- Most users come from Greater London (UKI), followed up by the South East (UKJ) and the South West (UKI). Which indicates that the platform is more popular in the south of the UK. The low number of users from the North West (UKD) could indicate that the platform is not as popular in the north of the UK considering it's the third most populous region in the UK.

## Check regulars


```python
regulars = regulars_df.merge(inventory_df, on="variant_id", how="inner")
```


```python
# Top 10 items with the most regulars
regulars.groupby("variant_id")["user_id"].nunique().sort_values(ascending=False).head(
    10
).plot.bar(edgecolor="k")
plt.title("Top 10 Items with the Most Regulars")
plt.xlabel("Item")
plt.ylabel("Number of Regulars")
plt.xticks(
    ticks=range(0, 10), labels=regulars["product_type"][:10], rotation=45, ha="right"
)
plt.show()
```


    
![png](groceries_analysis_files/groceries_analysis_35_0.png)
    


Curious, the most regular items don't match the most ordered item's categories nor the most ordered categories.


```python
sns.kdeplot(regulars["price"], label="regular_price")
sns.kdeplot(inventory_df["price"], label="inventory_price")
sns.kdeplot(common_items["price"], label="common_items_price")
```




    <Axes: xlabel='price', ylabel='Density'>




    
![png](groceries_analysis_files/groceries_analysis_37_1.png)
    



```python
orders = (
    (
        orders_df.explode("ordered_items")["ordered_items"]
        .value_counts(normalize=True)
        .reset_index()
        .rename(
            columns={"ordered_items": "variant_id", "proportion": "orders_prevalence"}
        )
        .merge(inventory_df, on="variant_id")
    )
    .groupby("product_type")["orders_prevalence"]
    .sum()
    .reset_index()
    .sort_values("orders_prevalence", ascending=False)
)
orders.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>product_type</th>
      <th>orders_prevalence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12</th>
      <td>cleaning-products</td>
      <td>0.089627</td>
    </tr>
    <tr>
      <th>51</th>
      <td>tins-packaged-foods</td>
      <td>0.082986</td>
    </tr>
    <tr>
      <th>29</th>
      <td>long-life-milk-substitutes</td>
      <td>0.061478</td>
    </tr>
    <tr>
      <th>52</th>
      <td>toilet-roll-kitchen-roll-tissue</td>
      <td>0.058486</td>
    </tr>
    <tr>
      <th>20</th>
      <td>dishwashing</td>
      <td>0.040405</td>
    </tr>
    <tr>
      <th>45</th>
      <td>soft-drinks-mixers</td>
      <td>0.035273</td>
    </tr>
    <tr>
      <th>44</th>
      <td>snacks-confectionery</td>
      <td>0.035078</td>
    </tr>
    <tr>
      <th>15</th>
      <td>cooking-ingredients</td>
      <td>0.029178</td>
    </tr>
    <tr>
      <th>10</th>
      <td>cereal</td>
      <td>0.027918</td>
    </tr>
    <tr>
      <th>14</th>
      <td>condiments-dressings</td>
      <td>0.026131</td>
    </tr>
  </tbody>
</table>
</div>




```python
diff_prevalence = (
    inventory_df["product_type"]
    .value_counts(normalize=True)
    .rename("inventory_prevalence")
    .reset_index()
    .merge(
        regulars["product_type"]
        .value_counts(normalize=True)
        .rename("regulars_prevalence")
        .reset_index()
    )
    .merge(orders[["product_type", "orders_prevalence"]], how="left")
    .assign(inventory_rank=lambda x: x["inventory_prevalence"].rank(ascending=False))
    .assign(regulars_rank=lambda x: x["regulars_prevalence"].rank(ascending=False))
    .assign(orders_rank=lambda x: x["orders_prevalence"].rank(ascending=False))
)
```


```python
diff_prevalence.sort_values("regulars_prevalence", ascending=False).head(15)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>product_type</th>
      <th>inventory_prevalence</th>
      <th>regulars_prevalence</th>
      <th>orders_prevalence</th>
      <th>inventory_rank</th>
      <th>regulars_rank</th>
      <th>orders_rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>cleaning-products</td>
      <td>0.092325</td>
      <td>0.124850</td>
      <td>0.089627</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tins-packaged-foods</td>
      <td>0.072129</td>
      <td>0.093255</td>
      <td>0.082986</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>dishwashing</td>
      <td>0.015580</td>
      <td>0.055474</td>
      <td>0.040405</td>
      <td>22.0</td>
      <td>3.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>31</th>
      <td>toilet-roll-kitchen-roll-tissue</td>
      <td>0.010387</td>
      <td>0.053346</td>
      <td>0.058486</td>
      <td>32.5</td>
      <td>4.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>cooking-ingredients</td>
      <td>0.042123</td>
      <td>0.052148</td>
      <td>0.029178</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>snacks-confectionery</td>
      <td>0.070398</td>
      <td>0.043900</td>
      <td>0.035078</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>long-life-milk-substitutes</td>
      <td>0.013849</td>
      <td>0.037648</td>
      <td>0.061478</td>
      <td>25.5</td>
      <td>7.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>condiments-dressings</td>
      <td>0.030006</td>
      <td>0.034655</td>
      <td>0.026131</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>soft-drinks-mixers</td>
      <td>0.027698</td>
      <td>0.032061</td>
      <td>0.035273</td>
      <td>11.0</td>
      <td>9.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>cereal</td>
      <td>0.029429</td>
      <td>0.031329</td>
      <td>0.027918</td>
      <td>8.0</td>
      <td>10.0</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>pasta-rice-noodles</td>
      <td>0.038084</td>
      <td>0.030664</td>
      <td>0.022629</td>
      <td>5.0</td>
      <td>11.0</td>
      <td>12.5</td>
    </tr>
    <tr>
      <th>13</th>
      <td>cooking-sauces</td>
      <td>0.024812</td>
      <td>0.029533</td>
      <td>0.024371</td>
      <td>14.0</td>
      <td>12.0</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>dental</td>
      <td>0.024235</td>
      <td>0.028602</td>
      <td>0.020017</td>
      <td>15.5</td>
      <td>13.0</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>home-baking</td>
      <td>0.014426</td>
      <td>0.026806</td>
      <td>0.016747</td>
      <td>24.0</td>
      <td>14.0</td>
      <td>18.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>hand-soap-sanitisers</td>
      <td>0.013849</td>
      <td>0.025808</td>
      <td>0.018350</td>
      <td>25.5</td>
      <td>15.0</td>
      <td>16.0</td>
    </tr>
  </tbody>
</table>
</div>



Insights:
- Regulars product's rank categories are fairly similar to the most ordered product's categories. However, there are some major differences with the inventory product's rank categories. Diswashing, toilet & kitchen roll tissue, and milk substitutes products may be lacking. In any case, the stock level of each product is unknown, so no conclusions can be drawn.

# Part 2: EDA


```python
display(feature_frame.head())
feature_frame.info()
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variant_id</th>
      <th>product_type</th>
      <th>order_id</th>
      <th>user_id</th>
      <th>created_at</th>
      <th>order_date</th>
      <th>user_order_seq</th>
      <th>outcome</th>
      <th>ordered_before</th>
      <th>abandoned_before</th>
      <th>...</th>
      <th>count_children</th>
      <th>count_babies</th>
      <th>count_pets</th>
      <th>people_ex_baby</th>
      <th>days_since_purchase_variant_id</th>
      <th>avg_days_to_buy_variant_id</th>
      <th>std_days_to_buy_variant_id</th>
      <th>days_since_purchase_product_type</th>
      <th>avg_days_to_buy_product_type</th>
      <th>std_days_to_buy_product_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2807985930372</td>
      <td>3482464092292</td>
      <td>2020-10-05 16:46:19</td>
      <td>2020-10-05 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808027644036</td>
      <td>3466586718340</td>
      <td>2020-10-05 17:59:51</td>
      <td>2020-10-05 00:00:00</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>2</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808099078276</td>
      <td>3481384026244</td>
      <td>2020-10-05 20:08:53</td>
      <td>2020-10-05 00:00:00</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808393957508</td>
      <td>3291363377284</td>
      <td>2020-10-06 08:57:59</td>
      <td>2020-10-06 00:00:00</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>4</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808429314180</td>
      <td>3537167515780</td>
      <td>2020-10-06 10:37:05</td>
      <td>2020-10-06 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>


    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2880549 entries, 0 to 2880548
    Data columns (total 27 columns):
     #   Column                            Dtype  
    ---  ------                            -----  
     0   variant_id                        int64  
     1   product_type                      object 
     2   order_id                          int64  
     3   user_id                           int64  
     4   created_at                        object 
     5   order_date                        object 
     6   user_order_seq                    int64  
     7   outcome                           float64
     8   ordered_before                    float64
     9   abandoned_before                  float64
     10  active_snoozed                    float64
     11  set_as_regular                    float64
     12  normalised_price                  float64
     13  discount_pct                      float64
     14  vendor                            object 
     15  global_popularity                 float64
     16  count_adults                      float64
     17  count_children                    float64
     18  count_babies                      float64
     19  count_pets                        float64
     20  people_ex_baby                    float64
     21  days_since_purchase_variant_id    float64
     22  avg_days_to_buy_variant_id        float64
     23  std_days_to_buy_variant_id        float64
     24  days_since_purchase_product_type  float64
     25  avg_days_to_buy_product_type      float64
     26  std_days_to_buy_product_type      float64
    dtypes: float64(19), int64(4), object(4)
    memory usage: 593.4+ MB



```python
info_columns = ["variant_id", "order_id", "user_id", "created_at", "order_date"]
target = "outcome"
features_cols = [
    col for col in feature_frame.columns if col not in info_columns and col != target
]
categorical_cols = ["product_type", "vendor"]
binary_cols = ["ordered_before", "abandoned_before", "active_snoozed", "set_as_regular"]
numerical_cols = [
    col
    for col in features_cols
    if col not in categorical_cols and col not in binary_cols
]

print(f"Number of categorical variables : {len(categorical_cols)}")
print(f"Number of binary variables : {len(binary_cols)}")
print(f"Number of numerical variables : {len(numerical_cols)}")
print(f"Number of info variables : {len(info_columns)}")
```

    Number of categorical variables : 2
    Number of binary variables : 4
    Number of numerical variables : 15
    Number of info variables : 5



```python
feature_frame[target].value_counts()
for col in binary_cols:
    print(f"Value counts for {col}: {feature_frame[col].value_counts().to_dict()}")
    print(
        f"Mean outcome by {col} value : {feature_frame.groupby(col)[target].mean().to_dict()}"
    )
    print("\n")
```

    Value counts for ordered_before: {0.0: 2819658, 1.0: 60891}
    Mean outcome by ordered_before value : {0.0: 0.008223337723936732, 1.0: 0.1649669080816541}
    
    
    Value counts for abandoned_before: {0.0: 2878794, 1.0: 1755}
    Mean outcome by abandoned_before value : {0.0: 0.011106039542947498, 1.0: 0.717948717948718}
    
    
    Value counts for active_snoozed: {0.0: 2873952, 1.0: 6597}
    Mean outcome by active_snoozed value : {0.0: 0.011302554809544488, 1.0: 0.1135364559648325}
    
    
    Value counts for set_as_regular: {0.0: 2870093, 1.0: 10456}
    Mean outcome by set_as_regular value : {0.0: 0.010668992259135854, 1.0: 0.24971308339709258}
    
    



```python
# Correlation betweeen outcome and numerical features
correlation = feature_frame[numerical_cols + [target]].corr()
plt.figure(figsize=(20, 10))
sns.heatmap(correlation, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()
```


    
![png](groceries_analysis_files/groceries_analysis_46_0.png)
    



```python
# Distribution of numerical features by outcome
rows = int(np.ceil(len(numerical_cols) / 3))
fig, ax = plt.subplots(rows, 3, figsize=(20, 5 * rows))
ax = ax.flatten()
for i, col in enumerate(numerical_cols):
    sns.kdeplot(feature_frame.loc[lambda x: x[target] == 0, col], label="Class 0", ax=ax[i])
    sns.kdeplot(feature_frame.loc[lambda x: x[target] == 1, col], label="Class 1", ax=ax[i])
    ax[i].set_title(col)
    ax[i].legend()

plt.tight_layout()
```


    
![png](groceries_analysis_files/groceries_analysis_47_0.png)
    



```python
feature_frame[categorical_cols].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>product_type</th>
      <th>vendor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2880549</td>
      <td>2880549</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>62</td>
      <td>264</td>
    </tr>
    <tr>
      <th>top</th>
      <td>tinspackagedfoods</td>
      <td>biona</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>226474</td>
      <td>146828</td>
    </tr>
  </tbody>
</table>
</div>



Insights:
- There are 62 unique product_types and 264 brands, making the categorical encoding of these features a challenge, as it would create a large number of dummy variables in case of using one-hot encoding. Label or frecuencia encoding could be used instead.
- The majority of the products have not been ordered or abandoned before, are not snoozed, and have not been set as regulars. However, when they are set as regulars or abandoned before, there's a noticeable increase in the mean outcome.
- Regarding numerical variables, some of them have high spikes in the distribution, which most likely are the result of univariate imputations: mean,median,etc.
- There are multiple corralated features, so we should be careful when using them in a model, specially if we are using a regression model.


