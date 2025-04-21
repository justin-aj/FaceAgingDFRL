import pandas as pd


df = pd.read_csv("data/csv/utk_train_balanced.csv")

# Add prompt column (FADING style): "a photo of a [age] year old person"
df['prompt'] = df['age'].apply(lambda age: f"a photo of a {age} year old person")


df.to_csv("data/csv/utk_train_balanced_with_prompts.csv", index=False)
print("✅ Prompt column added. Saved as utk_train_balanced_with_prompts.csv")
