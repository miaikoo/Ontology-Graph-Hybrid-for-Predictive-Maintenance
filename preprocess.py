import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DataPreprocessor:
    """
    Handles data preprocessing including:
    1. Missing value imputation with 0
    2. Removal of constant columns
    3. Min-max normalization
    """
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.constant_columns = []
        self.feature_columns = []
        self.min_values = None
        self.max_values = None
    
    def load_raw_data(self, data_path, labels_path):
        """Load raw data from CSV files"""
        print(">>> Loading raw data...")
        
        # Load features
        df_x = pd.read_csv(data_path, sep=' ', header=None)
        df_x.columns = [f'f_{i}' for i in range(df_x.shape[1])]
        
        # Load labels
        df_y = pd.read_csv(labels_path, sep=' ', header=None)
        df_y.columns = ['Label', 'Timestamp']
        
        # Convert labels: -1 (Pass) -> 0, 1 (Fail) -> 1
        y = df_y['Label'].apply(lambda x: 1 if x == 1 else 0)
        
        print(f"    Loaded {len(df_x)} rows and {df_x.shape[1]} features")
        
        return df_x, y
    
    def handle_missing_values(self, df):
        """Handle missing values with 0 imputation"""
        print(">>> Handling missing values with 0 imputation...")
        
        missing_count = df.isnull().sum().sum()
        print(f"    Found {missing_count} missing values")
        
        df_filled = df.fillna(0)
        
        print(f"    All missing values imputed with 0")
        return df_filled
    
    def remove_constant_columns(self, df):
        """Remove columns with constant values (no variance)"""
        print(">>> Removing constant columns...")
        
        initial_cols = len(df.columns)
        
        # Calculate variance for each column
        variances = df.var()
        
        # Identify constant columns (variance = 0)
        constant_cols = variances[variances == 0].index.tolist()
        
        # Store for reference
        self.constant_columns = constant_cols
        
        # Remove constant columns
        df_filtered = df.drop(columns=constant_cols)
        
        removed_count = len(constant_cols)
        print(f"    Removed {removed_count} constant columns")
        print(f"    Remaining columns: {len(df_filtered)}")
        
        # Store feature column names
        self.feature_columns = df_filtered.columns.tolist()
        
        return df_filtered
    
    def normalize_minmax(self, df, fit=True):
        """Apply min-max normalization (scales to [0, 1])"""
        print(">>> Applying Min-Max normalization...")
        
        if fit:
            # Fit scaler on training data
            df_normalized = pd.DataFrame(
                self.scaler.fit_transform(df),
                columns=df.columns,
                index=df.index
            )
            print(f"    Scaler fitted on {len(df)} samples")
        else:
            # Transform using existing scaler (for test data)
            df_normalized = pd.DataFrame(
                self.scaler.transform(df),
                columns=df.columns,
                index=df.index
            )
            print(f"    Applied existing scaler to {len(df)} samples")
        
        print(f"    Data normalized to range [0, 1]")
        return df_normalized
    
    def preprocess(self, data_path, labels_path):
        """
        Complete preprocessing pipeline:
        1. Load data
        2. Handle missing values
        3. Remove constant columns
        4. Apply Min-Max normalization
        
        Returns:
            df_processed (DataFrame): Preprocessed features
            y (Series): Labels
        """
        print("\n" + "="*60)
        print("STARTING DATA PREPROCESSING PIPELINE")
        print("="*60)
        
        # 1. Load data
        df_x, y = self.load_raw_data(data_path, labels_path)
        
        # 2. Handle missing values
        df_x = self.handle_missing_values(df_x)
        
        # 3. Remove constant columns
        df_x = self.remove_constant_columns(df_x)
        
        # 4. Normalize
        df_x = self.normalize_minmax(df_x, fit=True)
        
        print("="*60)
        print("PREPROCESSING COMPLETE")
        print(f"Final dataset shape: {df_x.shape}")
        print(f"Label distribution: {y.value_counts().to_dict()}")
        print("="*60 + "\n")
        
        return df_x, y
    
    def preprocess_test_data(self, df):
        """
        Preprocess test data using the same transformations as training.
        Use this for test/validation sets after fitting on training data.
        
        Args:
            df (DataFrame): Raw test data
            
        Returns:
            df_processed (DataFrame): Preprocessed test data
        """
        # Handle missing values
        df = df.fillna(0)
        
        # Remove constant columns (same ones as identified in training)
        df = df.drop(columns=[col for col in self.constant_columns if col in df.columns])
        
        # Normalize using the fitted scaler
        df = self.normalize_minmax(df, fit=False)
        
        return df
    
    def get_preprocessing_info(self):
        """Return preprocessing metadata"""
        return {
            'constant_columns_removed': self.constant_columns,
            'final_features': self.feature_columns,
            'num_features': len(self.feature_columns),
            'scaler_min': self.scaler.data_min_.tolist() if hasattr(self.scaler, 'data_min_') else None,
            'scaler_max': self.scaler.data_max_.tolist() if hasattr(self.scaler, 'data_max_') else None,
        }