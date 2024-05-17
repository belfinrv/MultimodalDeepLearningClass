import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import optimizers

def train_model(merged_df):
    """
    Trains a multimodal model combining a 3D CNN and an MLP using the provided dataset.

    Parameters:
    merged_df (DataFrame): DataFrame containing the dataset with features and encoded labels.

    Returns:
    None
    """

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(merged_df, merged_df['Label_Encoded'], test_size=0.2, random_state=42)

    # Process the 3D image data
    X_train_3D = process_image_array(X_train)
    X_test_3D = process_image_array(X_test)

    # Prepare MLP input data by dropping unnecessary columns
    X_train_mlp = X_train.drop(columns=['File_Path', 'Label_Encoded'])
    X_test_mlp = X_test.drop(columns=['File_Path', 'Label_Encoded'])

    # Define input shapes and number of classes
    input_shape_3d = (52, 64, 48, 1)
    input_shape_mlp = X_train_mlp.shape[1]
    num_classes = len(np.unique(y_train))

    # Build the model
    model = build_model(input_shape_3d=input_shape_3d, num_features=input_shape_mlp, num_classes=num_classes)
    model.summary()

    # Compile the model
    optimizer = optimizers.Adam(learning_rate=0.00001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    # Create callbacks for early stopping and model checkpointing
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(
        'best_model.h5',  # Path where the model will be saved
        monitor='val_loss',  # Metric to monitor
        save_best_only=True,  # Save only the best model
        mode='min',  # Minimization problem (i.e., lower val_loss is better)
        verbose=1  # Log a message whenever the model is saved
    )

    # Train the model
    model.fit(
        [X_train_3D, X_train_mlp], y_train,
        epochs=25,
        batch_size=32,
        validation_data=([X_test_3D, X_test_mlp], y_test),
        callbacks=[early_stopping, model_checkpoint]
    )

    # Save the trained model
    model.save('final_model.h5')
