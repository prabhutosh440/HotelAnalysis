import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

data_folder = './'
read_dir = data_folder + 'Consolidated_5D_NY_5Star.xlsx'
output_dir = data_folder + 'output_plots/'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir+'Rate_Analysis/', exist_ok=True)
os.makedirs(output_dir+'Inventory_Analysis/', exist_ok=True)
os.makedirs(output_dir+'Additional_Analysis/', exist_ok=True)

def convert_to_miles(distance_str):
    # Conversion factors
    conversion_factors = {
        'miles': 1,  # Miles to miles
        'km': 0.621371,  # Kilometers to miles
        'm' : 0.000621371,
        'yards': 0.000568182,  # Yards to miles
        'feet': 0.000189394  # Feet to miles
    }

    # Extract the numeric value and the unit using regex
    match = re.search(r"([0-9.]+)\s*(miles|km|m|yards|feet)", distance_str)

    if match:
        value = float(match.group(1))  # Extract numeric part
        unit = match.group(2)  # Extract unit part

        # Convert to miles using the conversion factor
        return value * conversion_factors[unit]
    else:
        return None  # Return None if no valid pattern is found

### Function 1: Rate Comparison
def rate_comparison(df1, df2):

    # Group by 'Date' and calculate the mean 'Prices'
    df1_grouped = df1.groupby('Checkin')['Price'].mean().reset_index(name='Avg_Price_1')
    df2_grouped = df2.groupby('Checkin')['Price'].mean().reset_index(name='Avg_Price_2')
    merged_df = pd.merge(df1_grouped,df2_grouped,on='Checkin')
    merged_df['Price_Difference_All'] = merged_df['Avg_Price_1'] - merged_df['Avg_Price_2']

    df1_common_grouped = df1[df1['Name'].isin(set(df2['Name']))].groupby('Checkin')['Price'].mean().reset_index(name='Avg_Price_1')
    df2_common_grouped = df2[df2['Name'].isin(set(df1['Name']))].groupby('Checkin')['Price'].mean().reset_index(name='Avg_Price_2')
    merged_common_df = pd.merge(df1_common_grouped, df2_common_grouped, on='Checkin')
    merged_common_df['Price_Difference_Common'] = merged_common_df['Avg_Price_1'] - merged_common_df['Avg_Price_2']

    # Plot the differences in average prices
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    plt.suptitle('Price Differences in $ (Booking - Expedia)', fontsize=16, fontweight='bold')

    # First subplot: Price difference for all hotels
    plt.subplot(1, 2, 1)
    sns.barplot(x='Checkin', y='Price_Difference_All', data=merged_df, palette='Blues_d')
    plt.title('All Hotels'); plt.xlabel('Date'); plt.ylabel('Price Difference'); plt.ylim(-150, 150)

    # Second subplot: Price difference for common hotels
    plt.subplot(1, 2, 2)
    sns.barplot(x='Checkin', y='Price_Difference_Common', data=merged_common_df, palette='Blues_d')
    plt.title('Common Hotels'); plt.xlabel('Date'); plt.ylabel('Price Difference'); plt.ylim(-150, 150)

    # Adjust layout and show plot
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(output_dir+'Rate_Analysis/'+'Price Differences in $ (Booking - Expedia)')

    plt.figure(figsize=(12, 6))
    plt.suptitle('Avg Price ($) over Days (Booking vs Expedia)', fontsize=16, fontweight='bold')
    sns.lineplot(x='Checkin', y='Avg_Price_1', data=merged_df, markers=True,marker="o", color='r', dashes=True, markersize=10, label='Booking')
    sns.lineplot(x='Checkin', y='Avg_Price_2', data=merged_df, markers=True,marker="o", color='b', dashes=True, markersize=10, label='Expedia')
    plt.xlabel('Date'); plt.ylabel('Average Price'); plt.legend()
    # plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(output_dir + 'Rate_Analysis/' + 'Avg Price ($) over Days (Booking vs Expedia)')

    df1_grouped = df1.groupby('Checkin')['Cancellation'].value_counts(normalize=True).reset_index(name='Booking')
    df1_grouped = df1_grouped[df1_grouped['Cancellation'] == True].filter(['Checkin','Booking'])
    df2_grouped = df2.groupby('Checkin')['Cancellation'].value_counts(normalize=True).reset_index(name='Expedia')
    df2_grouped = df2_grouped[df2_grouped['Cancellation'] == True].filter(['Checkin','Expedia'])
    df1_merged = df1_grouped.merge(df2_grouped,how='left',on='Checkin')

    df1_merged['Expedia'] = df1_merged['Expedia'].fillna(df1_merged['Expedia'].mean())

    df_melted = df1_merged.melt(id_vars='Checkin', value_vars=['Booking', 'Expedia'], var_name='Site', value_name='Free_Cancellation%')

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Checkin', y='Free_Cancellation%', hue='Site', data=df_melted, palette='coolwarm')

    # Add labels and title
    plt.title('Comparison of Free Cancellation Availability', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12); plt.ylabel('Free Cancellation %', fontsize=12); plt.ylim(0, 1)

    # Show plot
    plt.tight_layout()
    plt.savefig(output_dir+'Rate_Analysis/'+'Free_Cancellation% (Booking vs Expedia)')


    df1_grouped = df1.groupby('Checkin')['Discount%'].mean().reset_index(name='Discount_1')
    df2_grouped = df2.groupby('Checkin')['Discount%'].mean().reset_index(name='Discount_2')
    merged_df = pd.merge(df1_grouped, df2_grouped, on='Checkin')
    merged_df['Discount_Difference_All'] = merged_df['Discount_2'] - merged_df['Discount_1']

    df1_common_grouped = df1[df1['Name'].isin(set(df2['Name']))].groupby('Checkin')['Discount%'].mean().reset_index(name='Discount_1')
    df2_common_grouped = df2[df2['Name'].isin(set(df1['Name']))].groupby('Checkin')['Discount%'].mean().reset_index(name='Discount_2')
    merged_common_df = pd.merge(df1_common_grouped, df2_common_grouped, on='Checkin')
    merged_common_df['Discount_Difference_Common'] = merged_common_df['Discount_1'] - merged_common_df['Discount_2']

    # Plot the differences in average prices
    plt.figure(figsize=(12, 6))
    plt.suptitle('Discount Differences in $ (Booking - Expedia)', fontsize=16, fontweight='bold')

    # First subplot: Price difference for all hotels
    plt.subplot(1, 2, 1)
    sns.barplot(x='Checkin', y='Discount_Difference_All', data=merged_df, palette='Blues_d')
    plt.title('All Hotels'); plt.xlabel('Date'); plt.ylabel('Discount Difference');

    # Second subplot: Price difference for common hotels
    plt.subplot(1, 2, 2)
    sns.barplot(x='Checkin', y='Discount_Difference_Common', data=merged_common_df, palette='Blues_d')
    plt.title('Common Hotels'); plt.xlabel('Date'); plt.ylabel('Discount Difference');

    # Adjust layout and show plot
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(output_dir +'Rate_Analysis/'+ 'Discount Differences in $ (Expedia - Booking)')


### Function 2: Inventory Comparison
def inventory_comparison(df1, df2):
    df1_grouped = df1.groupby('Checkin')['No of Rooms Left'].mean().reset_index(name='Avg_Room_Left_1')
    df2_grouped = df2.groupby('Checkin')['No of Rooms Left'].mean().reset_index(name='Avg_Room_Left_2')
    merged_df = pd.merge(df1_grouped, df2_grouped, on='Checkin')
    merged_df['Avg_Room_Left_All'] = merged_df['Avg_Room_Left_1'] - merged_df['Avg_Room_Left_2']

    df1_common_grouped = df1[df1['Name'].isin(set(df2['Name']))].groupby('Checkin')['No of Rooms Left'].mean().reset_index(name='Avg_Room_Left_1')
    df2_common_grouped = df2[df2['Name'].isin(set(df1['Name']))].groupby('Checkin')['No of Rooms Left'].mean().reset_index(name='Avg_Room_Left_2')
    merged_common_df = pd.merge(df1_common_grouped, df2_common_grouped, on='Checkin')
    merged_common_df['Avg_Room_Left_Common'] = merged_common_df['Avg_Room_Left_1'] - merged_common_df['Avg_Room_Left_2']

    # Plot the differences in average prices
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")

    # First subplot: Price difference for all hotels
    plt.subplot(1, 2, 1)
    plt.suptitle('Average Room Left (Booking - Expedia)', fontsize=16, fontweight='bold')
    sns.barplot(x='Checkin', y='Avg_Room_Left_All', data=merged_df, palette='Blues_d')
    plt.title('All Hotels'); plt.xlabel('Date'); plt.ylabel('Avg_Room_Left'); plt.ylim(-50, 50)

    # Second subplot: Price difference for common hotels
    plt.subplot(1, 2, 2)
    sns.barplot(x='Checkin', y='Avg_Room_Left_Common', data=merged_common_df, palette='Blues_d')
    plt.title('Common Hotels'); plt.xlabel('Date'); plt.ylabel('Avg_Room_Left'); plt.ylim(-50, 50)

    # Adjust layout and show plot
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(output_dir + 'Inventory_Analysis/' + 'Average Room Left (Booking - Expedia)')


### Function 3: Impact of Different Parameters
def impact_of_parameters(df1, df2):
    df1['Website'] = 'Booking'
    df2['Website'] = 'Expedia'
    data = pd.concat([df1, df2], ignore_index=True)

    # Define Segmentation Criteria:

    # 1. Review Score Segmentation (above 8 vs below 8)
    data['Review Score Group'] = pd.cut(data['Review Score'], bins=[0, 8, 10],
                                        labels=['Low Score (<8.5)', 'High Score (>8.5)'])

    # 2. Distance from Centre Segmentation (within 2 km vs farther)
    data['Distance Group'] = pd.cut(data['Distance from Centre'], bins=[0, 2, 100],
                                    labels=['Close to Centre (<2 mile)', 'Far from Centre (>2 mile)'])

    data1 = data[data['Discount%']>1]
    data1['Review Score Group'] = pd.cut(data1['Review Score'], bins=[0, 8, 10],
                                        labels=['Low Score (<8.5)', 'High Score (>8.5)'])



    ### Analysis: Price, Cancellation, and Inventory based on Segments
    # 1. Price vs Review Score Group
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Review Score Group', y='Price', hue='Website', data=data, palette='Set1')
    plt.title('Price Comparison Based on Review Score Group')
    plt.ylabel('Price ($)')
    plt.savefig(f'{output_dir+'Additional_Analysis/'}/price_vs_review_score_group.png')

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Review Score Group', y='No of Rooms Left', hue='Website', data=data, palette='Set1')
    plt.title('Rooms available Based on Review Score Group')
    plt.ylabel('Price ($)')
    plt.savefig(f'{output_dir+'Additional_Analysis/'}/rooms_left_vs_review_score_group.png')

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Review Score Group', y='Discount%', hue='Website', data=data1, palette='Set1')
    plt.title('Discount% Based on Review Score Group')
    plt.ylabel('Discount %')
    plt.savefig(f'{output_dir+'Additional_Analysis/'}/discount_vs_review_score_group.png')

    # 5. Price vs Distance Group
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Distance Group', y='Price', hue='Website', data=data, palette='cool')
    plt.title('Price Comparison Based on Distance from Centre')
    plt.ylabel('Price ($)')
    plt.savefig(f'{output_dir+'Additional_Analysis/'}/price_vs_distance_group.png')

    data1 = df1.filter(['Checkin','Price','Review Score','Distance from Centre']).groupby(['Checkin']).mean()
    data2 = df2.filter(['Checkin', 'Price', 'Review Score', 'Distance from Centre']).groupby(['Checkin']).mean()
    sizes = [100]*len(data1)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data1['Price'], data1['Review Score'], data1['Distance from Centre'], color='b', marker='o', label='Booking',s=sizes)
    ax.scatter(data2['Price'], data2['Review Score'], data2['Distance from Centre'], color='r', marker='^', label='Expedia',s=sizes)
    ax.set_xlabel('Price'); ax.set_ylabel('Review Score'); ax.set_zlabel('Distance from Centre')
    ax.set_title('Comparison of Prices, Review Scores, and Distance from Centre'); ax.legend()
    plt.savefig(f'{output_dir + 'Additional_Analysis/'}/price_vs_review_score_vs_distance_from_center.png')


def final_comparison_common_uncommon_hotels(df1, df2):
    # Step 1: Identify common and uncommon hotels based on 'Name'
    common_hotels = df1[df1['Name'].isin(df2['Name'])]
    uncommon_hotels_booking = df1[~df1['Name'].isin(df2['Name'])]
    uncommon_hotels_expedia = df2[~df2['Name'].isin(df1['Name'])]

    # Step 2: Calculate mean for each category
    def calculate_mean(df, columns):
        return df[columns].mean().round(2)

    columns_to_compare = ['Price', 'Discount%', 'Cancellation', 'No of Rooms Left', 'Review Score',
                          'Distance from Centre']

    # Mean for Booking (common and uncommon hotels)
    mean_common_booking = calculate_mean(common_hotels, columns_to_compare)
    mean_uncommon_booking = calculate_mean(uncommon_hotels_booking, columns_to_compare)

    # Mean for Expedia (common and uncommon hotels)
    mean_common_expedia = calculate_mean(df2[df2['Name'].isin(df1['Name'])], columns_to_compare)
    mean_uncommon_expedia = calculate_mean(uncommon_hotels_expedia, columns_to_compare)

    # Step 3: Create the comparison table
    comparison_table = pd.DataFrame({
        'Avg of common hotels (Booking)': mean_common_booking,
        'Avg of uncommon hotels (Booking)': mean_uncommon_booking,
        'Avg of common hotels (Expedia)': mean_common_expedia,
        'Avg of uncommon hotels (Expedia)': mean_uncommon_expedia
    })
    print(comparison_table)



def clean_data():
    # Generate dummy data for two hotel listing websites
    np.random.seed(42)  # For reproducibility
    n_hotels = 35

    data_website_1 = pd.DataFrame({
        'Price': np.random.randint(200, 800, n_hotels),
        'Cancellation': np.random.choice(['Yes', 'No'], n_hotels),
        'Discount%': np.random.randint(0, 30, n_hotels),
        'Review Count': np.random.randint(50, 500, n_hotels),
        'Review Score': np.random.uniform(2.0, 5.0, n_hotels),
        'Rooms Available': np.random.choice(['Yes', 'No'], n_hotels),
        'No of Rooms Left': np.random.randint(0, 15, n_hotels),
        'Distance from Centre': np.random.uniform(0.5, 10.0, n_hotels)
    })

    data_website_2 = pd.DataFrame({
        'Price': np.random.randint(250, 850, n_hotels),
        'Cancellation': np.random.choice(['Yes', 'No'], n_hotels),
        'Discount%': np.random.randint(5, 25, n_hotels),
        'Review Count': np.random.randint(40, 600, n_hotels),
        'Review Score': np.random.uniform(2.5, 5.0, n_hotels),
        'Rooms Available': np.random.choice(['Yes', 'No'], n_hotels),
        'No of Rooms Left': np.random.randint(1, 20, n_hotels),
        'Distance from Centre': np.random.uniform(1.0, 15.0, n_hotels)
    })

    df = pd.read_excel(read_dir)

    df = df.rename(columns={'Reviews rating': 'Review Score', 'Number of reviews': 'Review Count',
                            'Base Price per night': 'Price', 'Cancellation charge': 'Cancellation',
                            'Offers': 'Discount', 'Offers %': 'Discount%',
                            'Distance from the city center': 'Distance from Centre'})
    relevant_cols = ['Checkin', 'CheckOut', 'Name', 'Price', 'Cancellation', 'Discount','Discount%', 'Review Count',
                     'Review Score', 'Rooms Available', 'No of Rooms Left', 'Distance from Centre']

    booksrcDf1 = df[df['Website'] == 'B1']
    booksrcDf2 = df[df['Website'] == 'B2']
    expsrcDf = df[df['Website'] == 'Ex']

    booksrcDf1 = booksrcDf1.drop(columns=['Distance from Centre', 'Discount'])
    booksrcDf1 = pd.merge(booksrcDf1, booksrcDf2.rename(columns={'Price': 'Price2'}).
                          filter(['Name', 'Distance from Centre', 'Discount', 'Price2', 'Checkin', 'CheckOut']),
                          on=['Name', 'Checkin', 'CheckOut'], how='left')
    booksrcDf1['Price2'] = booksrcDf1['Price2'].fillna(0)
    booksrcDf1['Price'] = booksrcDf1.apply(lambda x: x['Price2'] if (np.isnan(x['Price']) or x['Price'] == 0) else x['Price'], axis=1)
    booksrcDf1['Price'] = booksrcDf1['Price'].replace(0, pd.NA).dropna()
    booksrcDf1['Cancellation'] = booksrcDf1['Cancellation'].apply(lambda x: True if (x == 'free_cancellation') else False)
    booksrcDf1['Discount'] = booksrcDf1['Discount'].fillna(0)
    booksrcDf1['Discount%'] = (booksrcDf1['Discount'] / booksrcDf1['Price']) * 100
    booksrcDf1['Distance from Centre'] = booksrcDf1['Distance from Centre'].apply(convert_to_miles)
    booksrcDf1['No of Rooms Left'] = booksrcDf1.filter(like='RoomCount', axis=1).replace(' ', pd.NA).fillna(0).apply(lambda x: x.unique().sum(), axis=1)
    booksrcDf1['Rooms Available'] = booksrcDf1['No of Rooms Left'].apply(lambda x: True if x > 0 else False)
    booksrcDf1 = booksrcDf1.drop_duplicates(['Checkin', 'CheckOut', 'Name'])
    booksrcDf1 = booksrcDf1.filter(relevant_cols)

    expsrcDf['Discount'] = expsrcDf['Discount'].fillna(0).apply(
        lambda x: float(x[1:].replace(",", "")) if (isinstance(x, str) and '$' in x) else x)
    expsrcDf['Price'] = expsrcDf['Price'].replace(0, pd.NA).dropna()
    expsrcDf['Price'] = expsrcDf['Price'].apply(lambda x: float(x[1:].replace(",", "")))
    expsrcDf['Discount'] = expsrcDf.apply(
        lambda x: float(x['Discount'] - x['Price']) if (x['Discount'] > 0 and (x['Discount'] - x['Price']) > 0) else x[
            'Discount'], axis=1)
    expsrcDf['Discount%'] = (expsrcDf['Discount'] / expsrcDf['Price']) * 100
    expsrcDf['Cancellation'] = expsrcDf['Cancellation'].apply(
        lambda x: True if (x == 'REFUNDABLE_WITH_NO_PENALTY') else False)
    expsrcDf['No of Rooms Left'] = expsrcDf.filter(like='RoomCount').replace(' ', pd.NA).fillna(0).apply(
        lambda x: x.unique().sum(), axis=1)
    expsrcDf['Rooms Available'] = expsrcDf['No of Rooms Left'].apply(lambda x: True if x > 0 else False)
    expsrcDf = expsrcDf.drop_duplicates(['Checkin', 'CheckOut', 'Name'])
    expsrcDf = expsrcDf.filter(relevant_cols)

    return booksrcDf1, expsrcDf


if __name__ == "__main__":
    print("Data Cleaning and Fetching:")
    data_website_1, data_website_2 = clean_data()

    print("Rate Comparison:")
    rate_comparison(data_website_1, data_website_2)

    print("\nInventory Comparison:")
    inventory_comparison(data_website_1, data_website_2)

    print("\nImpact of Different Parameters:")
    impact_of_parameters(data_website_1, data_website_2)

    print("\nFinal Comparision Table:")
    final_comparison_common_uncommon_hotels(data_website_1, data_website_2)
