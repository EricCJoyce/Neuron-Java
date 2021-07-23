/*
jblas-1.2.4.jar needs to be in the same directory as all the Neuron *.java files and xor.java.

This program creates a network file named "xor.nn".
Issue the following commands to compile and run.

Run by calling:
  java -cp .:./jblas-1.2.4.jar XorExample
*/

public class XorExample
  {
    public static void main(String[] args)
      {
        NeuralNet nn;
        NeuralNet nn2;
        double in[];
        double out[];

        nn = new NeuralNet(2);                                      //  Initialize for two inputs
        nn.addDense(2, 2);                                          //  Add dense layer (0)
        nn.dense(0).setName("Dense-1");                             //  Name the first dense layer
        nn.addDense(2, 1);                                          //  Add dense layer (1)
        nn.dense(1).setName("Dense-2");                             //  Name the second dense layer
                                                                    //  Connect input to dense[0]
        nn.linkLayers(NeuralNet.INPUT_ARRAY, 0, 0, 2, NeuralNet.DENSE_ARRAY, 0);
                                                                    //  Connect input to dense[1]
        nn.linkLayers(NeuralNet.DENSE_ARRAY, 0, 0, 2, NeuralNet.DENSE_ARRAY, 1);

        nn.dense(0).setW_ij( 4.169506539262890, 0, 0);              //  Set unit 0, weight 0 of layer 0
        nn.dense(0).setW_ij( 4.175620772246105, 0, 1);              //  Set unit 0, weight 1 of layer 0
        nn.dense(0).setW_ij(-6.399885541033798, 0, 2);              //  Set unit 0, weight 2 (bias) of layer 0
        nn.dense(0).setF_i(ActivationFunction.SIGMOID, 0);          //  Set activation function of unit 0, layer 0

        nn.dense(0).setW_ij( 6.166518349083749, 1, 0);              //  Set unit 1, weight 0 of layer 0
        nn.dense(0).setW_ij( 6.187965760394095, 1, 1);              //  Set unit 1, weight 1 of layer 0
        nn.dense(0).setW_ij(-2.678140646720913, 1, 2);              //  Set unit 1, weight 2 (bias) of layer 0
        nn.dense(0).setF_i(ActivationFunction.SIGMOID, 1);          //  Set activation function of unit 1, layer 0

        nn.dense(0).print();                                        //  Show me the layer we just built

        nn.dense(1).setW_ij(-9.175274710095412, 0, 0);              //  Set unit 0, weight 0 of layer 1
        nn.dense(1).setW_ij( 8.486130185157748, 0, 1);              //  Set unit 0, weight 1 of layer 1
        nn.dense(1).setW_ij(-3.875273098510313, 0, 2);              //  Set unit 0, weight 2 (bias) of layer 1
        nn.dense(1).setF_i(ActivationFunction.SIGMOID, 0);          //  Set activation function of unit 0, layer 1

        nn.dense(1).print();                                        //  Show me the layer we just built
        System.out.println("");

        nn.sortEdges();
        nn.printEdgeList();                                         //  Show me all the connections
        nn.setComment("Two-input network performs XOR");
        nn.print();                                                 //  Summarize the network

        nn.write("xor.nn");                                         //  Write the network to file

        in = new double[2];

        in[0] = 1.0;                                                //  Now let's try the network out.
        in[1] = 0.0;                                                //  This input should produce a signal close to 1.0
        out = nn.run(in);
        System.out.printf("%f %f\n", in[0], in[1]);
        System.out.printf("%f\n\n", out[0]);

        in[0] = 1.0;                                                //  This input should produce a signal close to 0.0
        in[1] = 1.0;
        out = nn.run(in);
        System.out.printf("%f %f\n", in[0], in[1]);
        System.out.printf("%f\n\n", out[0]);

        in[0] = 0.0;                                                //  This input should produce a signal close to 0.0
        in[1] = 0.0;
        out = nn.run(in);
        System.out.printf("%f %f\n", in[0], in[1]);
        System.out.printf("%f\n\n", out[0]);

        in[0] = 0.0;                                                //  This input should produce a signal close to 1.0
        in[1] = 1.0;
        out = nn.run(in);
        System.out.printf("%f %f\n", in[0], in[1]);
        System.out.printf("%f\n\n", out[0]);

        nn = null;                                                  //  Destroy the network
        System.gc();                                                //  Call the garbage collector

        System.out.println(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n");

        nn2 = new NeuralNet(2);                                     //  Initialize
        nn2.load("xor.nn");                                         //  Load the network we just wrote to file
        nn2.dense(0).print();                                       //  Show me the layers retrieved from file
        nn2.dense(1).print();
        nn2.printEdgeList();                                        //  Show me all the connections

        nn2.print();

        in[0] = 1.0;                                                //  Check that these produce the same outputs
        in[1] = 0.0;
        out = nn2.run(in);
        System.out.printf("%f %f\n", in[0], in[1]);
        System.out.printf("%f\n\n", out[0]);

        in[0] = 1.0;
        in[1] = 1.0;
        out = nn2.run(in);
        System.out.printf("%f %f\n", in[0], in[1]);
        System.out.printf("%f\n\n", out[0]);

        in[0] = 0.0;
        in[1] = 0.0;
        out = nn2.run(in);
        System.out.printf("%f %f\n", in[0], in[1]);
        System.out.printf("%f\n\n", out[0]);

        in[0] = 0.0;
        in[1] = 1.0;
        out = nn2.run(in);
        System.out.printf("%f %f\n", in[0], in[1]);
        System.out.printf("%f\n\n", out[0]);

        nn2 = null;                                                 //  Clean up
        System.gc();                                                //  Call the garbage collector

        return;
      }
  }
