
###################################################
# PROJECT: Rating Product & Sorting Reviews in Amazon
###################################################

###################################################
# Business Problem
###################################################

# One of the most important problems in e-commerce is the correct calculation of the points given to the products after sales. 
# The solution to this problem means providing greater customer satisfaction for the e-commerce site, prominence of the product 
# for the sellers and a seamless shopping experience for the buyers. Another problem is the correct ordering of the comments given to the products. 
# Since misleading comments will directly affect the sale of the product, it will cause both financial loss and loss of customers. In the solution of 
# these 2 basic problems, while the e-commerce site and the sellers will increase their sales, the customers will complete the purchasing journey without any problems.



###################################################
The story of the dataset
###################################################

# This dataset, which includes Amazon product data, includes product categories and various metadata. 
# The product with the most reviews in the electronics category has user ratings and reviews.

# Variables:
# reviewerID: User ID
# asin	Product: ID
# reviewerName:	User name
# helpful: Useful rating
# reviewText: Evaluation
# overall: Product rating
# summary: Rating summary
# unixReviewTime: Evaluation time
# reviewTime: Number of days since evaluation
# day_diff: The number of times the evaluation was found useful
# helpful_yes: Number of votes given to the evaluation
# total_vote: Evaluation time Raw



###################################################
# TASK 1: Calculate Average Rating Based on Current Comments and Compare with Existing Average Rating.
###################################################

# In the shared data set, users gave points and comments to a product.
# Our aim in this task is to evaluate the scores given by weighting them by date.
# It is necessary to compare the first average score with the weighted score according to the date to be obtained.

# Library Imports and Dataset Reading

import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

###################################################
# Step 1: Read the Data Set and Calculate the Average Score of the Product.
###################################################

df = pd.read_csv("Rating Product&SortingReviewsinAmazon/amazon_review.csv")

def check_df(df, head=5):
    print("###################### SHAPE #########")
    print(df.shape)
    print("###################### TYPES #########")
    print(df.dtypes)
    print("###################### HEAD #########")
    print(df.head())
    print("###################### TAIL #########")
    print(df.tail())
    print("###################### NA #########")
    print(df.isnull().sum())
    print("###################### SUMMARY STATISTICS #########")
    print(df.describe().T)
check_df(df)


df['overall'].mean


###################################################
# Step 2: Calculate the Weighted Average of Score by Date.
###################################################

df.sort_values('day_diff', ascending=False)

# Average of comments made in the last 30 days:
df.loc[df["day_diff"] <= 30, "overall"].mean()

# Average of comments made between (30,90] days
df.loc[(df["day_diff"] > 30) & (df["day_diff"] <= 90), "overall"].mean()

# Average of comments made between (90,180] days
df.loc[(df["day_diff"] > 90) & (df["day_diff"] <= 180), "overall"].mean()

# Average of comments made over 180 days
df.loc[(df["day_diff"] > 180), "overall"].mean()



# Weighted Rating
df.loc[df["day_diff"] <= 30, "overall"].mean() * 28/100 + \
df.loc[(df["day_diff"] > 30) & (df["day_diff"] <= 90), "overall"].mean() * 26/100 + \
df.loc[(df["day_diff"] > 90) & (df["day_diff"] <= 180), "overall"].mean() * 24/100 + \
df.loc[(df["day_diff"] > 180), "overall"].mean() * 22/100




###################################################
# Task 2: Specify 20 Reviews for the Product to be Displayed on the Product Detail Page.
###################################################


###################################################
# Step 1. Generate the helpful_no variable
###################################################
# Note:
# total_vote is the total number of up-downs given to a comment.
# up means helpful.
# There is no helpful_no variable in the data set, it must be generated over existing variables.

df['helpful_no'] = df['total_vote'] - df['helpful_yes']
df.head()

###################################################
# Step 2. Calculate score_pos_neg_diff, score_average_rating and wilson_lower_bound Scores and add to dataframe
###################################################
df.info()

def score_pos_neg_diff(up, down):
    return up - down

df['score_pos_neg_diff'] = df.apply(lambda x: score_pos_neg_diff(x["helpful_yes"],
                                                                 x["helpful_no"]),
                                    axis=1)

def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)


df['score_average_rating'] = df.apply(lambda x: score_average_rating(x["helpful_yes"],
                                                                 x["helpful_no"]),
                                    axis=1)


def wilson_lower_bound(up, down, confidence=0.95):
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


df['wilson_lower_bound'] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"],
                                                                 x["helpful_no"]),
                                    axis=1)

##################################################
# Adım 3. Step 3. Identify 20 Comments and Interpret Results.
###################################################

df.sort_values("wilson_lower_bound", ascending=False).head(20)


