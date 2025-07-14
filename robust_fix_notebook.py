#!/usr/bin/env python
"""
Robust fix for the notebook variable conflicts and scope issues
"""
import nbformat

# Read the notebook
with open('pds_consumer_sentiment_analysis_executed_fixed2.ipynb', 'r') as f:
    nb = nbformat.read(f, as_version=4)

print("Fixing notebook issues...")

# Fix 1: Update IRF plotting to use irf_periods instead of periods
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code' and 'periods = range(25)' in cell.source:
        print(f"Found IRF cell at index {i} - updating to use irf_periods")
        cell.source = cell.source.replace('periods = range(25)', 'irf_periods = range(25)')
        cell.source = cell.source.replace('ax.fill_between(periods,', 'ax.fill_between(irf_periods,')
        cell.source = cell.source.replace('ax.plot(periods,', 'ax.plot(irf_periods,')

# Fix 2: Update the period-based analysis cell to properly define periods
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code' and '# Define economic periods' in cell.source:
        print(f"Found economic periods definition at index {i}")
        # Make sure it uses economic_periods
        cell.source = cell.source.replace('periods = {', 'economic_periods = {')
        cell.source = cell.source.replace('for period_name, (start, end) in periods.items():', 
                                          'for period_name, (start, end) in economic_periods.items():')

# Fix 3: Update the problematic RF period analysis cell
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code' and 'period_importance = {}' in cell.source and 'for period_name, (start, end) in periods.items():' in cell.source:
        print(f"Found RF periods cell at index {i} - applying robust fix")
        
        cell.source = """# Random Forest with optimized parameters
rf_optimized = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

# Fit model
rf_optimized.fit(X_train, y_train)

# Predictions
y_pred_rf_train = rf_optimized.predict(X_train)
y_pred_rf_test = rf_optimized.predict(X_test)

# Evaluate
rf_train_r2 = r2_score(y_train, y_pred_rf_train)
rf_test_r2 = r2_score(y_test, y_pred_rf_test)
rf_train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_rf_train))
rf_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf_test))

print("Random Forest Performance:")
print(f"Train R²: {rf_train_r2:.4f}, Test R²: {rf_test_r2:.4f}")
print(f"Train RMSE: {rf_train_rmse:.4f}, Test RMSE: {rf_test_rmse:.4f}")

# Feature importance from optimized model
rf_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_optimized.feature_importances_
}).sort_values('importance', ascending=False)

# Define economic periods for analysis (not to be confused with irf_periods)
economic_periods = {
    '1990s Boom': ('1990-01-01', '2000-12-31'),
    'Pre-Crisis (2001-2008)': ('2001-01-01', '2008-08-31'),
    'Recovery (2009-2019)': ('2009-01-01', '2019-12-31'),
    'Post-COVID (2020-2025)': ('2020-01-01', '2025-05-31')
}

# Ensure we have the required data
try:
    # Check if X_selected exists and has the right structure
    if 'X_selected' not in locals() or not hasattr(X_selected, 'index'):
        print("Recreating X_selected from available data...")
        if 'selected_features' in locals() and 'df_features_clean' in locals():
            X_selected = df_features_clean[selected_features]
        else:
            print("Warning: Could not recreate X_selected. Using X_train features.")
            X_selected = df_features_clean[X_train.columns] if 'df_features_clean' in locals() else None
    
    # Check if y exists
    if 'y' not in locals():
        print("Recreating y from available data...")
        if 'df_features_clean' in locals() and 'UMCSENT' in df_features_clean.columns:
            y = df_features_clean['UMCSENT']
        else:
            print("Warning: Could not recreate y.")
            y = None
    
    # Only proceed if we have valid data
    if X_selected is not None and y is not None:
        # Plot feature importance over time periods
        period_importance = {}
        
        for period_name, (start, end) in economic_periods.items():
            try:
                period_mask = (X_selected.index >= start) & (X_selected.index <= end)
                if period_mask.sum() > 50:  # Need sufficient data
                    X_period = X_selected[period_mask]
                    y_period = y[period_mask]
                    
                    rf_period = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                    rf_period.fit(X_period, y_period)
                    
                    period_importance[period_name] = pd.Series(
                        rf_period.feature_importances_,
                        index=X_period.columns
                    )
                    print(f"✓ Computed importance for {period_name}: {period_mask.sum()} samples")
                else:
                    print(f"✗ Skipping {period_name}: insufficient data ({period_mask.sum()} samples)")
            except Exception as e:
                print(f"✗ Error processing {period_name}: {str(e)}")
        
        # Create heatmap of feature importance evolution
        if len(period_importance) > 0:
            importance_matrix = pd.DataFrame(period_importance)
            
            if not importance_matrix.empty:
                # Get features with highest variance across periods
                feature_variance = importance_matrix.std(axis=1)
                top_evolving_features = feature_variance.nlargest(10).index
                
                plt.figure(figsize=(10, 8))
                sns.heatmap(importance_matrix.loc[top_evolving_features].T, 
                            cmap='YlOrRd', annot=True, fmt='.3f',
                            cbar_kws={'label': 'Feature Importance'})
                plt.title('Evolution of Feature Importance Across Time Periods', fontsize=14)
                plt.xlabel('Features')
                plt.ylabel('Time Period')
                plt.tight_layout()
                plt.savefig('visualizations/feature_importance/importance_evolution_heatmap.png', dpi=300)
                plt.show()
                
                # Save results
                importance_matrix.to_csv('results/summary_tables/feature_importance_by_period.csv')
                print(f"\\nSaved period importance analysis for {len(period_importance)} periods")
            else:
                print("Warning: Importance matrix is empty")
        else:
            print("Warning: No period importance data was computed")
    else:
        print("ERROR: Required data (X_selected or y) not available for period analysis")
        
except Exception as e:
    print(f"ERROR in period importance analysis: {str(e)}")
    import traceback
    traceback.print_exc()

# Save RF results
rf_importance.to_csv('results/summary_tables/rf_feature_importance_optimized.csv', index=False)
print("\\nRandom Forest analysis complete!")"""

# Fix 4: Update any other references to periods that should be economic_periods
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code' and 'for period_name, (start, end) in periods.items():' in cell.source:
        if 'economic_periods' not in cell.source:  # Don't update if already fixed
            print(f"Found periods reference in cell {i} - updating to economic_periods")
            cell.source = cell.source.replace('for period_name, (start, end) in periods.items():', 
                                              'for period_name, (start, end) in economic_periods.items():')

# Save the robustly fixed notebook
output_filename = 'pds_consumer_sentiment_analysis_robust_fixed.ipynb'
with open(output_filename, 'w') as f:
    nbformat.write(nb, f)

print(f"\nSaved robustly fixed notebook as: {output_filename}")
print("\nKey changes made:")
print("1. IRF plotting now uses 'irf_periods' instead of 'periods'")
print("2. Economic period analysis uses 'economic_periods' with proper definition")
print("3. Added validation checks for X_selected and y variables")
print("4. Added detailed error handling and progress messages")
print("5. Ensured all variable names are consistent throughout")