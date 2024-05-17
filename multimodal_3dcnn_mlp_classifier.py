from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D, Dropout, Flatten, Dense, concatenate

def build_model(input_shape_3d, num_features, num_classes):
    # 3D CNN
    input_3d = Input(shape=input_shape_3d)
    cnn = Conv3D(32, (3, 3, 3), padding='same', activation='relu')(input_3d)
    cnn = Conv3D(32, (3, 3, 3), padding='same', activation='relu')(cnn)
    cnn = MaxPooling3D((2, 2, 2), padding='same')(cnn)
    cnn = Dropout(0.1)(cnn)

    cnn = Conv3D(64, (3, 3, 3), padding='same', activation='relu')(cnn)
    cnn = Conv3D(64, (3, 3, 3), padding='same', activation='relu')(cnn)
    cnn = MaxPooling3D((2, 2, 2), padding='same')(cnn)
    cnn = Dropout(0.1)(cnn)

    cnn = Conv3D(128, (3, 3, 3), padding='same', activation='relu')(cnn)
    cnn = Conv3D(128, (3, 3, 3), padding='same', activation='relu')(cnn)
    cnn = MaxPooling3D((2, 2, 2), padding='same')(cnn)
    cnn = Dropout(0.1)(cnn)
    flat = Flatten()(cnn)

    # MLP
    input_mlp = Input(shape=(num_features,))
    mlp = Dense(128, activation='relu')(input_mlp)
    mlp = Dropout(0.1)(mlp)
    mlp = Dense(64, activation='relu')(mlp)

    # Combine
    combined = concatenate([flat, mlp])

    # Classification layers
    output = Dense(64, activation='relu')(combined)
    output = Dense(num_classes, activation='softmax')(output)

    # Create model
    model = Model(inputs=[input_3d, input_mlp], outputs=output)
    return model
