import matplotlib.pyplot as plt

class Visualizer:
    @staticmethod
    def plot_data_and_predictions(data, predictions, model_name, start_index):
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data, label='Actual Price', color='blue')
        predicted_index = data.index[start_index:start_index + len(predictions)]
        plt.plot(predicted_index, predictions, label=f'{model_name} Predictions', color='orange')
        plt.title('Stock Price and Predictions')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.show()
