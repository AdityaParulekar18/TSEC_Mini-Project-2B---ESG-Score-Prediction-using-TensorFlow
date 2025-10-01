# tensorflow_model.py
import tensorflow as tf
import pandas as pd
import numpy as np


# Load CSV containing 16 raw features and overall ESG_Score.
df = pd.read_csv('synthetic_esg_dataset_with_subtargets.csv')
feature_cols = [
    'CO2_Emissions', 'Renewable_Energy', 'Water_Consumption', 'Waste_Management', 'Biodiversity_Impact',
    'Gender_Diversity', 'Employee_Satisfaction', 'Community_Investment', 'Safety_Incidents', 'Labor_Rights',
    'Board_Diversity', 'Executive_Pay_Ratio', 'Transparency', 'Shareholder_Rights', 'Anti_Corruption', 'Political_Donations'
]
X = df[feature_cols].values.astype(np.float32)
y = df['ESG_Score'].values.astype(np.float32)


from sklearn.model_selection import train_test_split  # only for splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define a normalization layer that adapts on the training data.
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(X_train)

# Custom Polynomial Expansion layer with fixed output shape.
class PolynomialExpansion(tf.keras.layers.Layer):
    def __init__(self, n_features, **kwargs):
        super(PolynomialExpansion, self).__init__(**kwargs)
        self.n_features = n_features
        # Total output features: original + squared + interactions.
        self.n_output = n_features + n_features + (n_features * (n_features - 1)) // 2

    def call(self, inputs):
        # inputs shape: (batch, n_features)
        orig = inputs
        sq = tf.square(inputs)
        interactions = []
        # Use a Python loop (for small n_features this is fine)
        for i in range(self.n_features):
            for j in range(i + 1, self.n_features):
                interactions.append(inputs[:, i] * inputs[:, j])
        if interactions:
            # Stack interactions along axis=1: shape becomes (batch, num_interactions)
            interactions = tf.stack(interactions, axis=1)
            return tf.concat([orig, sq, interactions], axis=1)
        else:
            return tf.concat([orig, sq], axis=1)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_output)

# Build the model using the Functional API.
inputs = tf.keras.Input(shape=(16,))
x = normalizer(inputs)
# Use our custom polynomial expansion layer (with n_features = 16)
x = PolynomialExpansion(n_features=16)(x)
# Final Dense layer: For regression, a single unit.
outputs = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
model.summary()


history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, batch_size=32)


loss, mae = model.evaluate(X_test, y_test)
print("Test MSE:", loss)
print("Test MAE:", mae)

# Optionally, display some weights for comparison.
dense_layer = model.layers[-1]
W_tf, b_tf = dense_layer.get_weights()
print("Final Dense Weights shape:", W_tf.shape)
print("Final Dense Bias:", b_tf)
