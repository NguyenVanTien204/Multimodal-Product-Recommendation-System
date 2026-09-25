import time
import pandas as pd

t0 = time.time()
df = pd.read_parquet("../data/items.parquet")

text_combined = (df["title"].fillna("") + " " + df["category"].fillna("")).str.lower()

is_shoe = text_combined.str.contains("shoe|sneaker|boot|sandal|footwear|loafer|heel|slipper", regex=True)
is_bag = text_combined.str.contains("backpack|tote|handbag|purse|wallet|luggage|duffel|crossbody|bag", regex=True)
is_jewelry = text_combined.str.contains("watch|ring|necklace|bracelet|earring|jewelry|pendant|gold|silver|diamond", regex=True)
is_acc = text_combined.str.contains("sock|belt|hat|cap|sunglasses|glasses|scarf|glove|beanie|tie", regex=True)
is_sport = text_combined.str.contains("swim|running|yoga|gym|athletic|jersey|activewear|cycling", regex=True)
is_men = text_combined.str.contains(r"men's|mens|male|guy|gentleman|polo", regex=True)

cat = pd.Series("thoi-trang-nu", index=df.index)
cat[is_men] = "thoi-trang-nam"
cat[is_sport] = "the-thao-ngoai-troi"
cat[is_acc] = "phu-kien"
cat[is_jewelry] = "dong-ho-trang-suc"
cat[is_bag] = "tui-xach-balo"
cat[is_shoe] = "giay-dep"

print(f"Classification of all {len(df):,} items done in {time.time() - t0:.2f}s")
print(cat.value_counts())
