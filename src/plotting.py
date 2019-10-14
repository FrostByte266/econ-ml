import matplotlib.pyplot as plt

def plot_train_errors(history, block=True):
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Test Loss')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(loc='upper right')
    plt.show(block=block)

def plot_dataset(train_set, predictions):
    plot_size=len(train_set[0])
    pred_size=len(predictions)
    aa=[x for x in range(10000)]

    plt.plot(aa[:plot_size], train_set[0][:plot_size], label="actual")
    plt.plot(aa[:pred_size], predictions[:,0][:pred_size], 'r', label="prediction")

    plt.legend(fontsize=15)
    plt.show()