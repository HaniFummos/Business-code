import numpy as np
import pandas as pd

def rescale_score(score, gold_score, min_possible=0, max_possible=None):
    if max_possible is None:
        if score <= 1.0:
            max_possible = 1.0
        else:
            max_possible = 100.0
    if score >= gold_score:
        return 100.0
    
    return ((score - min_possible) / (gold_score - min_possible)) * 100

def calculate_combined_scores(df, gold_flesch, gold_bert):
    df = df.copy()

    df['Flesch_Rescaled'] = df['Flesch-Douma'].apply(
        lambda x: rescale_score(x, gold_flesch, min_possible=0, max_possible=100)
    )
    df['BERT_Rescaled'] = df['BERTScore'].apply(
        lambda x: rescale_score(x, gold_bert, min_possible=0, max_possible=1.0)
    )
    
    df['Geometric_Mean'] = np.sqrt(df['Flesch_Rescaled'] * df['BERT_Rescaled'])
    
    df['Harmonic_Mean'] = 2 * (df['Flesch_Rescaled'] * df['BERT_Rescaled']) / (
        df['Flesch_Rescaled'] + df['BERT_Rescaled']
    )
    
    df['Better_Than_Gold_Flesch'] = df['Flesch-Douma'] >= gold_flesch
    df['Better_Than_Gold_BERT'] = df['BERTScore'] >= gold_bert
    
    return df

data = {
    'Strategy': ['Zero-shot', 'Zero-shot', 'Zero-shot', 
                 'Persona', 'Persona', 'Persona',
                 'Few-shot', 'Few-shot', 'Few-shot',
                 'Retcon', 'Retcon', 'Retcon',
                 'RetconPos', 'RetconPos', 'RetconPos'],
    'Temp': [0.0, 0.7, 1.0, 0.0, 0.7, 1.0, 0.0, 0.7, 1.0, 0.0, 0.7, 1.0, 0.0, 0.7, 1.0],
    'Flesch-Douma': [89.51, 88.59, 88.16, 89.33, 88.45, 87.53, 91.73, 91.93, 91.80, 91.93, 92.48, 92.62, 92.33, 92.49, 92.83],
    'BERTScore': [0.667, 0.667, 0.670, 0.665, 0.665, 0.664, 0.662, 0.663, 0.664, 0.660, 0.658, 0.655, 0.661, 0.661, 0.658]
}

GOLD_FLESCH = 93.77
GOLD_BERT = 0.6477

df = pd.DataFrame(data)

df_results = calculate_combined_scores(df, GOLD_FLESCH, GOLD_BERT)

df_sorted = df_results.sort_values('Harmonic_Mean', ascending=False)

print("=" * 80)
print("RESULTS WITH RESCALED SCORES (Gold standard = 100%)")
print("=" * 80)
print("\n")

display_cols = ['Strategy', 'Temp', 'Flesch-Douma', 'BERTScore', 
                'Flesch_Rescaled', 'BERT_Rescaled', 
                'Geometric_Mean', 'Harmonic_Mean']

print(df_results[display_cols].round(2).to_string(index=False))
print("\n")

print("=" * 80)
print("TOP 5 PERFORMERS (by Harmonic Mean)")
print("=" * 80)
top_5 = df_sorted.head(5)[['Strategy', 'Temp', 'Harmonic_Mean', 'Geometric_Mean', 
                           'Better_Than_Gold_Flesch', 'Better_Than_Gold_BERT']]
print(top_5.round(2).to_string(index=False))

print("\n")
print("=" * 80)
print("ANALYSIS")
print("=" * 80)

exceeds_gold = df_results[(df_results['Flesch-Douma'] > GOLD_FLESCH) | 
                          (df_results['BERTScore'] > GOLD_BERT)]
if len(exceeds_gold) > 0:
    print("\nWARNING: Some summaries exceed the gold standard:")
    print(exceeds_gold[['Strategy', 'Temp', 'Flesch-Douma', 'BERTScore']].to_string(index=False))
    print("\nThese get capped at 100% in the rescaling, as they're better than gold.")
else:
    print("\nNo summaries exceed the gold standard. All rescaled scores are < 100%.")

print(f"\nFlesch Rescaled Range: {df_results['Flesch_Rescaled'].min():.1f}% - {df_results['Flesch_Rescaled'].max():.1f}%")
print(f"BERT Rescaled Range: {df_results['BERT_Rescaled'].min():.1f}% - {df_results['BERT_Rescaled'].max():.1f}%")
print(f"Geometric Mean Range: {df_results['Geometric_Mean'].min():.1f}% - {df_results['Geometric_Mean'].max():.1f}%")
print(f"Harmonic Mean Range: {df_results['Harmonic_Mean'].min():.1f}% - {df_results['Harmonic_Mean'].max():.1f}%")