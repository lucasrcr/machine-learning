import tensorflow as tf
from sklearn.model_selection import train_test_split

'Prepare aqui os dados para treinar sua rede'

# over = RandomOverSampler()
# X, y = over.fit_resample(X, y)
# data = np.hstack((X, np.reshape(y, (-1, 1))))
# transformed_df = pd.DataFrame(data)

'Teste de uma única vez os diversos valores dos parâmetros'

accuracy_list = []

# Definição dos hiperparâmetros a serem testados
test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
layers = [16, 32, 64]
epochs = [10, 20]
learning_rates = [0.001, 0.01, 0.005]
batch_sizes = [16, 32, 64]

# Loop para testar diferentes combinações de hiperparâmetros
for i in range(3):  # Alterando o random_state para i para maior variação
    for test_size in test_sizes:
        for layer in layers:
            for epoch in epochs:
                for learning_rate in learning_rates:
                    for batch_size in batch_sizes:

                        # Divisão dos dados
                        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=i)
                        X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=i)

                        # Definição do modelo
                        model = tf.keras.Sequential([
                            tf.keras.layers.Dense(layer, activation='relu'),
                            tf.keras.layers.Dense(layer, activation='relu'),
                            tf.keras.layers.Dense(layer, activation='relu'),
                            tf.keras.layers.Dense(layer, activation='relu'),
                            tf.keras.layers.Dense(1, activation="sigmoid")
                        ])

                        # Compilação do modelo
                        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                                      loss=tf.keras.losses.BinaryCrossentropy(),
                                      metrics=['accuracy'])

                        # Treinamento do modelo
                        history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epoch,
                                            validation_data=(X_valid, y_valid), verbose=0)

                        # Pegando valores de loss e acurácia
                        initial_val_loss = history.history['val_loss'][0]
                        final_val_loss = history.history['val_loss'][-1]
                        final_val_accuracy = history.history['val_accuracy'][-1]

                        # Condição para salvar o modelo apenas se melhorar na validação
                        if final_val_loss < initial_val_loss and final_val_accuracy > 0.5:
                            print(f"Validation Loss caiu para o Test Size {test_size}: Inicial = {initial_val_loss}, Final = {final_val_loss}")

                            # Avaliação final no conjunto de teste
                            test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
                            accuracy_list.append(test_accuracy)

                            # Nome do modelo salvo
                            model_name = f"models/accuracy_{round(test_accuracy,3)}.h5"
                            model.save(model_name)
                            print(f"Test Size: {test_size}, Layers: {layer}, Epochs: {epoch}, LR: {learning_rate}, Batch: {batch_size}, Test Accuracy: {test_accuracy}, Model Saved as {model_name}")

                        else:
                            print(f"Validation Loss NÃO caiu para o Test Size {test_size}: Inicial = {initial_val_loss}, Final = {final_val_loss}. Modelo NÃO foi salvo.")

# Exibe a melhor acurácia obtida
if accuracy_list:
    print(f"Melhor Test Accuracy: {max(accuracy_list)}")
else:
    print("Nenhum modelo atendeu aos critérios de melhoria e foi salvo.")
