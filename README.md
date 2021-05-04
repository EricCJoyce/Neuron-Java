# Neuron-Java
## Neural network library written in Java

Want to train networks in Keras/TensorFlow and run them in stand-alone Java applications? Then this is for you.

## Requirements
### Java

New to Java? Get started on Ubuntu 18.04 by issuing the following commands:
```
apt-get update
apt-get upgrade
apt-get install software-properties-common
add-apt-repository ppa:webupd8team/java
apt-get update
apt-get install oracle-java8-installer
apt-get install default-jre
apt-get install default-jdk
```

Check your installation by issuing the following command:
```
javac -version
```

### [jblas](http://jblas.org/)

BLAS is a Fortran library of expertly optimized linear algebraic functions. jBLAS is Dr. Mikio Braun's adaptation of this powerful library to Java. Neuron-Java uses jBLAS to do matrix-matrix and matrix-vector multiplication as quickly as possible. Why depend on jBLAS rather than write these routines myself? Because the underlying BLAS is the bedrock of linear algebraic operations. As [Dr. Shusen Wang](http://wangshusen.github.io/) says, "Do not try to write and optimize these operations yourself. There are just... too many tricks."

For you to use the Neuron-Java library, you'll need to install jBLAS. The following step successfully installed jBLAS version 1.2.4 on Ubuntu 18.04.

Open a command-line terminal and issue the following command:
```
apt-get install jblas
```

## Transfer Trained Weights from Keras

Too long to read? Watch [this]().

(to be continued)

## Citation

If this code was helpful for your research, please consider citing this repository.

```
@misc{neuron_java_2020,
  title={Neuron-Java},
  author={Eric C. Joyce},
  year={2020},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/EricCJoyce/Neuron-Java}}
}
```
