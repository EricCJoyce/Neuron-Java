/*
  Eric C. Joyce, Stevens Institute of Technology, 2020

  jblas-1.2.4.jar needs to be in the same directory as all the Neuron *.java files and RunNeuronModel.java.

  Read one of the 28-by-28 MNIST PGM files to be identified.
  Convert its unsigned char values in [0, 255] to a double in [0.0, 1.0].
  Input this floating-point buffer to the neural network.
  Print each of the 10 output values.

  Run by calling:
    java -cp .:./jblas-1.2.4.jar RunNeuronModel mnist.nn samples/sample_1.pgm
*/

public class RunNeuronModel
  {
    public static void main(String[] args)
      {
        NeuralNet nn;
        P5Image img;
        double in[];
        double out[];
        int i;

        nn = new NeuralNet(784);                                    //  Initialize for 28 * 28 images, flattened
        nn.load(args[0]);                                           //  Load given network from file

        img = new P5Image();
        img.read(args[1]);

        in = new double[784];
        for(i = 0; i < 784; i++)
          in[i] = (double)img.buffer[i] / 255.0;

        out = nn.run(in);
        for(i = 0; i < out.length; i++)
          System.out.printf("%d\t%f\n", i, out[i]);

        return;
      }
  }