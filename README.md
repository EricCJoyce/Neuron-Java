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

BLAS is a Fortran library of expertly optimized linear algebraic functions. jblas is [Dr. Mikio Braun](https://github.com/mikiobraun)'s adaptation of this powerful library to Java. Neuron-Java uses jblas to do matrix-matrix and matrix-vector multiplication as quickly as possible. Why depend on jblas rather than write these routines myself? Because the underlying BLAS is the bedrock of linear algebraic operations. As [Dr. Shusen Wang](http://wangshusen.github.io/) says, "Do not try to write and optimize these operations yourself. There are just... too many tricks."

For you to use the Neuron-Java library, you'll need to download the jblas JAR and install jblas. The following steps successfully installed jblas version 1.2.4 on Ubuntu 18.04.

You will need the jblas Java archive file. Download it via the command line as follows, but take note of where you download this file; you will need to refer to it explicitly when compiling and running.
```
wget http://jblas.org/jars/jblas-1.2.4.jar
```
Now issue the following command:
```
apt-get install jblas
```
Supposing you've saved the JAR file, this Neuron-Java library's source, and your own code, `MyCode.java`, in the same directory `/my/sourcecode/directory`, call the compiler like this:
```
javac -cp .:/my/sourcecode/directory/jblas-1.2.4.jar MyCode.java
```
Similarly, you'd call the Java runtime environment (JRE) like this:
```
java -cp .:/my/sourcecode/directory/jblas-1.2.4.jar MyCode
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
