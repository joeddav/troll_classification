import pandas as pd

print("Loading data...")
non_troll_df = pd.read_csv('data/non_troll_data.csv', index_col='Unnamed: 0')

columns_to_remove = [
    "id_str",
    "geo",
    "coordinates",
    "contributors",
    "in_reply_to_screen_name",
    "in_reply_to_status_id",
    "in_reply_to_status_id_str",
    "in_reply_to_user_id",
    "in_reply_to_user_id_str",
    "favorite_count",
    "favorited",
    "entities",
    "extended_entities",
    "place",
    "quoted_status",
    "quoted_status_id",
    "quoted_status_id_str"
]

def value_from_dict(key):

    def inner(string):
        try:
            return eval(string)[key]
        except TypeError:
            return None
    
    return inner

print("Removing unnecessary columns...")
non_troll_df.drop(columns_to_remove, axis=1, inplace=True)

print("Copying followers and following from user...")
non_troll_df["followers"] = non_troll_df["user"].apply(value_from_dict("followers_count"))
non_troll_df["following"] = non_troll_df["user"].apply(value_from_dict("following"))
del non_troll_df["user"]

print("Creating retweet status feature...")
non_troll_df["is_a_retweet"] = non_troll_df.retweeted_status.notnull()
del non_troll_df["retweeted_status"]

print("Saving simplified dataset...")
non_troll_df.to_csv("data/non_troll_data_simplified.csv")

print("Done!")