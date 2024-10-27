import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

# Data for different words/phrases and their corresponding average scores
data = {
    'Word/Phrase': ['Abysmal', 'Appalling', 'Dreadful', 'Awful', 'Terrible', 'Very bad', 'Really bad', 'Rubbish', 
                    'Unsatisfactory', 'Bad', 'Poor', 'Quite bad', 'Pretty bad', 'Somewhat bad', 'Below average', 
                    'Mediocre', 'Average', 'Not bad', 'Fair', 'Alright', 'OK', 'Satisfactory', 'Fine', 
                    'Somewhat good', 'Quite good', 'Decent', 'Above average', 'Pretty good', 'Good', 
                    'Great', 'Really good', 'Very good', 'Awesome', 'Fantastic', 'Superb', 'Brilliant', 
                    'Incredible', 'Excellent', 'Outstanding', 'Perfect'],
    'Average Score': [1.21, 1.34, 1.44, 1.72, 1.75, 1.76, 1.85, 2.17, 2.48, 2.60, 2.71, 2.80, 2.85, 3.16, 3.38, 
                      4.29, 5.09, 5.13, 5.43, 5.48, 5.51, 5.70, 5.80, 5.82, 6.48, 6.56, 6.62, 6.76, 6.92, 7.76, 
                      7.78, 7.90, 8.44, 8.74, 8.77, 8.77, 8.81, 8.95, 9.11, 9.16]
}

# DataFrame
df = pd.DataFrame(data)

# Generate synthetic distributions based on average scores
x = np.linspace(0, 10, 500)
distributions = {word: gaussian_kde(np.random.normal(avg, 0.5, 1000))(x) for word, avg in zip(df['Word/Phrase'], df['Average Score'])}

# Plot ridgeline plot
plt.figure(figsize=(10, 15))

for i, (word, distribution) in enumerate(distributions.items()):
    plt.fill_between(x, i + distribution * 0.5, i, color=sns.color_palette("Spectral", as_cmap=True)(i / len(distributions)),
                     alpha=0.7)
    plt.text(10.2, i, word, ha='left', va='center', fontsize=8)

plt.xlim([0, 10])
plt.ylim([-1, len(distributions)])
plt.xlabel("Score (0=Very Negative, 10=Very Positive)")
plt.ylabel("Words/Phrases")
plt.title("Sentiment Distribution of Words/Phrases on a Scale of 0 to 10")
plt.gca().axes.get_yaxis().set_visible(False)

plt.show()
