
def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

def train(network, loss, loss_prime, x_train, y_train, epoch = 1000, learning_rate = 0.01, verbose = True):
    for e in range(epoch):
        error = 0
        for x,y in zip(x_train, y_train):
            # forward pass
            output = predict(network, x)
            # calculating the error
            error += loss(y,output)

            # backward
            grad = loss_prime(y,output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

        error /= len(x_train)
        if verbose:
            print(f"{e + 1}/{epoch}, error={error}")


            

            
    