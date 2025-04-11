import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

def draw(importances, feature_names, title, output_file):
    sorted_idx = importances.argsort()
    sorted_importances = importances[sorted_idx]
    sorted_features = [feature_names[i] for i in sorted_idx]
    
    plt.figure(figsize=(8, 6))
    plt.barh(range(len(sorted_features)), sorted_importances, align='center')
    plt.yticks(range(len(sorted_features)), sorted_features)
    plt.xlabel('Feature Importance')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def random_forest_feature_importance(df):
    x = df.drop(columns=['Outcome'])
    y = df['Outcome']
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(x, y)
    
    importances = rf.feature_importances_
    return importances, list(x.columns)

def main():
    input_csv = "test_A/train_data.csv"         
    output_plot = "rf_feature_importance.png"  

    df = pd.read_csv(input_csv)
    
    # Get importance
    importances, feature_names = random_forest_feature_importance(df)
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    print("Importance Rank:")
    print(importance_df)
    
    # draw inmportance ranking picture
    draw(importances, feature_names,
                            "Random Forest Feature Importance", output_plot)

if __name__ == '__main__':
    main()