all: AccumLayer.class ActivationFunction.class Conv2DLayer.class DenseLayer.class GRULayer.class LSTMLayer.class NormalLayer.class PoolLayer.class UpresLayer.class NeuralNet.class BuildNeuronModel.class P5Image.class RunNeuronModel.class
.PHONY: all

ActivationFunction.class: ActivationFunction.java
	javac -cp .:./jblas-1.2.4.jar ActivationFunction.java

AccumLayer.class: AccumLayer.java ActivationFunction.java
	javac -cp .:./jblas-1.2.4.jar AccumLayer.java

Conv2DLayer.class: Conv2DLayer.java ActivationFunction.java
	javac -cp .:./jblas-1.2.4.jar Conv2DLayer.java

DenseLayer.class: DenseLayer.java ActivationFunction.java
	javac -cp .:./jblas-1.2.4.jar DenseLayer.java

GRULayer.class: GRULayer.java ActivationFunction.java
	javac -cp .:./jblas-1.2.4.jar GRULayer.java

LSTMLayer.class: LSTMLayer.java ActivationFunction.java
	javac -cp .:./jblas-1.2.4.jar LSTMLayer.java

NeuralNet.class: NeuralNet.java
	javac -cp .:./jblas-1.2.4.jar NeuralNet.java

NormalLayer.class: NormalLayer.java ActivationFunction.java
	javac -cp .:./jblas-1.2.4.jar NormalLayer.java

PoolLayer.class: PoolLayer.java ActivationFunction.java
	javac -cp .:./jblas-1.2.4.jar PoolLayer.java

UpresLayer.class: UpresLayer.java ActivationFunction.java
	javac -cp .:./jblas-1.2.4.jar UpresLayer.java

BuildNeuronModel.class: BuildNeuronModel.java
	javac -cp .:./jblas-1.2.4.jar BuildNeuronModel.java

P5Image.class: P5Image.java
	javac P5Image.java

RunNeuronModel.class: RunNeuronModel.java
	javac -cp .:./jblas-1.2.4.jar RunNeuronModel.java