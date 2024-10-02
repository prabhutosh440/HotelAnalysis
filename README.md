# Hotel Listings Data Analysis

## Project Overview

This project involves scraping data from two hotel listing websites (e.g., Booking.com and Expedia) to analyze and compare their hotel offerings. We focus on evaluating **room rates** and **inventory** across these platforms. By using scraped data from both sites, we aim to answer the following key questions:

- Which site offers better rates for hotels?
- Which site has a superior inventory?
- Do these answers vary when considering factors like hotel location, star ratings, or other parameters?

The analysis leverages data on room prices, hotel ratings, and other relevant factors to provide insights into which platform may have a competitive advantage in terms of rates and inventory.

## Data Collection

Data was collected for **Booking.com** and **Expedia.com** using Apify. Due to limitation of getting data, the below analysis was done over 5 star rated hotels of New York for 5 days (29th Oct 2024 to 2nd Nov 2024). The data points collected include:

- **Hotel Name**
- **Price**
- **Discount Percentage (if available)**
- **Cancellation Policy**
- **Number of Rooms Left**
- **Review Score**
- **Distance from City Center**

### Steps to Collect Data

1. **Scraping Data**: Web scraping was performed using `Apify`. Both websites were scraped for hotels in **New York City**, specifically focusing on **5-star hotels** for 5 days duration **(29th Oct 2024 to 2nd Nov 2024)**.
2. **Data Storage & Preprocessing**: The raw data was cleaned and processed to standardize fields such as pricing and review scores for analysis.

## Analysis

### 1. Rate Comparison

The first goal of the analysis was to determine which site offers **better rates**. We evaluated this by comparing:

- **Average Room Price**: Calculated the average room price across all hotels for each platform and plotted trends over 5 days.
- **Price Difference**: Visualized the price difference for common and uncommon hotels between both sites over 5 days.
- **Discount Difference**: Segmented the data based on **location** and **hotel star rating** to analyze price trends for specific categories.
- **Free Cancellation**: Percentage of free cancellations available on both sites (Flexible Policy)

#### Key Findings:
- **Average Price**: The average price on Booking was slightly higher compared to Expedia. Also, the rising rates over weekends is clearly visible across websites.
- **Price Difference**: Overall Booking offers more competitive prices however, Expedia wins for common hotels listed on both websites.
- **Discount**: It seems Expedia is offering better discounts on common listings to be able to compete with Booking because, overall Booking is offering better discounts.
- **Free Cancellation**: Both are competitive and none of them seems having an edge on this Free cancellation policy.

### 2. Inventory Comparison

Next, we analyzed which site had a **superior inventory** by evaluating:

- **Number of Hotels with any Availability**: Checked if any types of rooms are available.
- **Number of rooms left**: Assessed the number of rooms available and how many were left at the time of scraping.

#### Key Findings:
- **Number of rooms left**: Booking have more rooms left than Expedia which can imply two things either it is because Booking is negotiaating better with hotels or because it is having higher prices than Expedia.

### 3. Impact of Different Parameters

We segmented the data by **hotel star ratings** and **location** to assess whether the answers to the above questions changed under different conditions:

- **Location-Based Price Comparison**: Prices varied significantly by neighborhood. In some areas, Expedia offered cheaper rates, while Booking.com was more competitive in others.
- **Star Rating Impact**: When focusing on 5-star hotels, Booking.com had a slight edge in terms of pricing flexibility and hotel variety.

#### Key Findings:
- **Price Comparison Based on Review Score Group**:
* Higher-rated hotels (>8.5) are priced similarly between the two platforms, though Booking.com has more high-end outliers.
* Lower-rated hotels (<8.5) are generally less expensive on Booking.com, with more price variance seen for cheaper properties.

-**Rooms Available Based on Review Score Group**:
* Booking.com consistently shows a larger number of rooms left for both high and low review score groups, with particularly high availability in lower-rated hotels (<8.5).
* Expedia has fewer rooms left across both score groups, with the difference more pronounced for highly rated hotels (>8.5), suggesting better booking performance or limited supply.
* The number of rooms left on Booking.com is much higher for high-scored hotels, indicating they may have larger capacities or lower occupancy rates than those on Expedia.

-**Price Comparison Based on Distance from Centre**:
* Hotels closer to the city centre (<2 miles) are priced higher, particularly on Booking.com, with some outliers exceeding $2,000.
* Hotels located farther from the city centre (>2 miles) tend to have more uniform pricing across both platforms, but Expedia generally lists lower-priced options.

-**Discount% Based on Review Score Group**:
* Hotels with higher review scores (>8.5) receive considerably higher discounts on Expedia, while Booking.com maintains lower, more consistent discounts across both score groups.
* There is a wider spread in discounts for higher-scoring hotels on Expedia, with some properties receiving discounts exceeding 50%.
* Lower-rated hotels (<8.5) have very similar discount percentages on both platforms, indicating less promotional effort for these properties.

-**3D plot comparing Booking (blue dots) and Expedia (red triangles) based on price, review scores, and distance from the centre**:
* Price: Booking tends to have lower prices compared to Expedia, with most blue dots clustered around lower price points.
* Review Scores: Booking's review scores are slightly higher, typically around the 9.0 range, whereas Expedia's scores tend to be below 9.0.
* Distance from Centre: Both Booking and Expedia properties are relatively close to the city centre, but Expedia's properties are slightly farther on average, with distances ranging from 1.2 to 1.7 miles, compared to Booking's range of 1.2 to 1.4 miles.

## Visualization

The analysis also includes various plots to visualize the findings:

- **Price Analysis**: /output_plots/Rate_Analysis/
- **Inventory Analysis**: /output_plots/Inventory_Analysis/
- **Additional Analysis**: /output_plots/Additional_Analysis/

## Results Summary

| Average Metric               | Booking.com (Common Hotels) | Booking.com (Uncommon Hotels) | Expedia (Common Hotels) | Expedia (Uncommon Hotels) |
|------------------------------|-----------------------------|-------------------------------|-------------------------|---------------------------|
| Price                        | $1040                       | $833                          | $978                    | $917                      |
| Discount Percentage (%)      | 0.4%                        | 0.5%                          | 2%                      | 0.9%                      |
| Free Cancellation (%)        | 24%                         | 57%                           | 15%                     | 36%                       |
| No of Rooms Available        | 22                          | 17                            | 3                       | 3                         |
| Average Review Score         | 8.2                         | 8.4                           | 9.0                     | 9.1                       |
| Distance from Center (Miles) | 1.7                         | 1.6                           | 1.3                     | 1.4                       |


## Conclusion

Based on the analysis, Booking.com appears to have a more diverse inventory and a slightly higher average price point compared to Expedia. However, Expedia offers more consistent pricing and competitive rates in certain neighborhoods. The impact of location and hotel star ratings plays a significant role in these findings, making it important for customers to consider multiple factors when booking hotels on either platform.

## How to Run the Code

### Requirements

- Python 3.x
- `pandas`, `numpy`, `matplotlib`, `seaborn`, `sqlite3`, `beautifulsoup4`, `selenium`

### Running the Scraper

- Run Hotel_Analysis.py
