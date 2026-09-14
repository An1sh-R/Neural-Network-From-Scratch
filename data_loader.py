import numpy as np

def load_train_data(path, val_size=5000, seed=42):
    data = np.genfromtxt(path, delimiter=',', skip_header=1)

    labels = data[:, 0].astype(int)
    pixels = data[:, 1:]

    # shuffle rows 
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(labels))
    labels = labels[indices]
    pixels = pixels[indices]

    # Normalize for faster convergence
    pixels = pixels / 255.0

    # split data
    X_val, y_val = pixels[:val_size], labels[:val_size]
    X_train, y_train = pixels[val_size:], labels[val_size:]

    return X_train, y_train, X_val, y_val


if __name__ == '__main__':
    X_train, y_train, X_val, y_val = load_train_data('data/data.csv')

    print('X_train:', X_train.shape, X_train.dtype)
    print('y_train:', y_train.shape, y_train.dtype)
    print('X_val:', X_val.shape, X_val.dtype)
    print('y_val:', y_val.shape, y_val.dtype)
    print('y_train[:10]:', y_train[:10])
    print('pixel range:', X_train.min(), X_train.max())
