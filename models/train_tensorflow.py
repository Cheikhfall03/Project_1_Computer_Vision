import tensorflow as tf
import matplotlib.pyplot as plt

class TrainerTF:
    def __init__(self, model, train_data, val_data, lr, epochs):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.epochs = epochs
        self.train_loss = []
        self.train_acc = []

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self, save=False, plot=False):
        history = self.model.fit(
            self.train_data,
            validation_data=self.val_data,
            epochs=self.epochs,
            verbose=1
        )

        self.train_loss = history.history['loss']
        self.train_acc = [v * 100 for v in history.history['accuracy']]  # convert to %
        
        if save:
            self.model.save("Cheikh_Fall_model.tensorflow")
        if plot:
            self.plot_training_history(history)

    def evaluate(self):
        loss, accuracy = self.model.evaluate(self.val_data, verbose=1)
        print(f"\nTest Accuracy: {accuracy * 100:.2f}%  |  Test Loss: {loss:.4f}")
        return accuracy, loss

    def plot_training_history(self, history):
        epochs = range(1, self.epochs + 1)

        fig, ax1 = plt.subplots(figsize=(8, 5))
        color_loss = 'tab:blue'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color=color_loss)
        ax1.plot(epochs, history.history['loss'], color=color_loss, label='Loss')
        ax1.tick_params(axis='y', labelcolor=color_loss)

        ax2 = ax1.twinx()
        color_acc = 'tab:red'
        ax2.set_ylabel('Accuracy (%)', color=color_acc)
        ax2.plot(epochs, [x * 100 for x in history.history['accuracy']], color=color_acc, label='Accuracy')
        ax2.tick_params(axis='y', labelcolor=color_acc)

        plt.title('Training Loss and Accuracy (TensorFlow)')
        fig.tight_layout()
        plt.show()
