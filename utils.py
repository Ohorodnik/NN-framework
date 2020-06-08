import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# %%
def plot_decision(model, features, labels):
    x_min, x_max = features[:, 0].min() - 0.5, features[:, 0].max() + 0.5
    y_min, y_max = features[:, 1].min() - 0.5, features[:, 1].max() + 0.5
    h = 0.01

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z = model(np.c_[xx.ravel(), yy.ravel()]).numpy().argmax(axis=1)
    Z = Z.reshape(xx.shape)
    
    #figure = plt.figure(figsize=(27, 9))
    ax = plt.subplot()

    ax.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=.8)
    ax.scatter(features[:, 0], features[:, 1], c=labels.ravel(), cmap=plt.cm.Spectral)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    
    plt.show()


# %%
def train(NN, dataset, loss, optimizer, epochs, print_period=1, use_log=False):
    loss_history = []
      
    for epoch in range(epochs):
        for batch_inputs, batch_labels in dataset:
            with tf.GradientTape() as tape:
                predictions = NN(batch_inputs)
                current_loss = loss(y_pred=predictions, y_true=batch_labels)
                loss_history.append(current_loss)
            
            gradients = tape.gradient(target=current_loss, sources=NN.trainable_weights)
            optimizer.apply_gradients(zip(gradients, NN.trainable_weights))
            

        if epoch % print_period == 0:
            print(f"Epoch {epoch}: train cost = {current_loss}")
        else:
            pass

    plt.plot(np.array(loss_history))
    plt.ylabel('Loss')
    plt.xlabel('Batch #')
    plt.ylim([0, 1.3])
    plt.show()
