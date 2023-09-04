import mnist_loader
import network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network.Network([784, 30, 10])

net.SGD(list(training_data), 30, 10, 5.0, test_data=list(test_data))


exit()
a=aplana(Imagen)
resultado = net.fedforward(a)
print (resultado)


